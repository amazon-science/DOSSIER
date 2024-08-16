import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import torch
import argparse
import socket
import random
import json
from fact_check import constants
from fact_check.constants import DATA_DIR
from fact_check.prompting import prompter
from fact_check.utils import Tee, path_serial
from fact_check.data import data_select, table_utils
from tqdm import tqdm
import pickle
from fact_check.models.llms import LLM
from fact_check.prompting.prompter import DirectLLMPrompter
import warnings
warnings.simplefilter("ignore", UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type = str)
parser.add_argument('--claim_df_path', default = DATA_DIR/'claims.csv')
parser.add_argument('--llm',  choices = ['medalpaca', 'clinicalcamel', 'llama2', 'clinicalcamel_new', 'asclepius'])
parser.add_argument('--temperature', default = 0.0, type = float)
parser.add_argument('--subset_adms', default = None, help = "take the first k adms instead of using all of them", type = int)
parser.add_argument('--n_samples', default = 1, type = int, help = '# samples in generation')
parser.add_argument('--select_rows',  choices = ['random', 'rag'], default = 'rag', 
    help = 'how to subset tables when the token limit is exceeded')
parser.add_argument('--rag_type',  choices = ['bm25', 'knn'], default = 'bm25', help = '`knn` uses all-MiniLM-L6-v2')
parser.add_argument('--rag_with_kg',  action = 'store_true', help = 'include KG rows in RAG')
# these files are technically not used in this script, but needed for the data processing to run
parser.add_argument('--cui_mapping_path', default = Path(__file__).parent.absolute()/'data/mimic3_cui_mapping.csv')
parser.add_argument('--semmeddb_path', default = DATA_DIR/'semmeddb/semmeddb_processed_10.csv')
parser.add_argument('--generics_path', default = DATA_DIR/'semmeddb/semmedVER43_2023_R_GENERIC_CONCEPT.csv')
parser.add_argument('--subset_predicates', default = ['ISA', 'TREATS', 'PREVENTS'], nargs = '+', type = str)
parser.add_argument('--output_dir', type = str, required = True)
parser.add_argument('--no_checkpoint', action = 'store_true', help = 'Do not checkpoint. Otherwise, checkpoint after every adm')
parser.add_argument('--seed', type = int, default = 42)
parser.add_argument('--debug', action = 'store_true')
args = parser.parse_args()
hparams = vars(args)

hparams['include_all_evidence'] = False
hparams['medcat_model_paths'] = []
hparams['umls_api_n_concepts'] = 1
hparams['prompt'] = 'full'

out_dir = Path(args.output_dir)
out_dir.mkdir(exist_ok = True, parents = True)

if not args.debug:
    sys.stdout = Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = Tee(os.path.join(args.output_dir, 'err.txt'))

print("Environment:")
print("\tPython: {}".format(sys.version.split(" ")[0]))
print("\tPyTorch: {}".format(torch.__version__))
print("\tCUDA: {}".format(torch.version.cuda))
print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
print("\tNumPy: {}".format(np.__version__))
print("\tNode: {}".format(socket.gethostname()))

print('Args:')
for k, v in sorted(hparams.items()):
    print('\t{}: {}'.format(k, v))

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

with open(out_dir/'args.json', 'w') as outfile:
    json.dump(vars(args), outfile, default=path_serial)

ds = data_select(hparams, None)
adms = ds.dfs['adms']
claims = ds.claims

if hparams['subset_adms'] is not None:
    adms = adms.iloc[:hparams['subset_adms']]

if hparams['debug']:
    adms = adms.iloc[:2]

print("Total # of claims: " + str(len(claims[claims.HADM_ID.isin(adms.index)])))

model = LLM('medalpaca_p2' if hparams['llm'] == 'medalpaca' else  hparams['llm'], max_new_tokens = 512, temperature = hparams['temperature'])

if hparams['llm'] == 'llama2':
    token_limit = 20000
elif hparams['llm'] in ['asclepius', 'clinicalcamel_new']:
    token_limit = 3000
else:
    token_limit = 1500

if (out_dir/'checkpoint.pkl').is_file():
    checkpoint = pickle.load((out_dir/'checkpoint.pkl').open('rb'))
    ress = checkpoint['ress']
    completed_hadms = checkpoint['completed_hadms']
    print(f"Loaded checkpoint with {len(completed_hadms)} adms completed!")
else:
    ress = []
    completed_hadms = []

prompt_gen = prompter.DirectLLMPrompter(include_examples = True, include_vitals = True, select_rows = args.select_rows, rag_type = args.rag_type,
                                        rag_with_kg = args.rag_with_kg, kg_path = args.semmeddb_path, kg_emb_path = constants.cache_dir/'kg_embeddings.npy',
                                        kg_subset_predicates = args.subset_predicates)

for hadm_id in tqdm(adms.index):
    if hadm_id in completed_hadms:
        continue
    tables = ds.make_tables(hadm_id) 
    tables = table_utils.subset_to_sql_schema(tables)
    del tables['Global_KG']    
    indices = {}

    claims_i = claims[claims.HADM_ID == hadm_id]
    for idx, row in tqdm(claims_i.iterrows(), total = len(claims_i)):
        claim = row['claim']
        t_C = row['t_C']
        label = row['label']
        prompt, indices = prompt_gen.get_prompt(claim, t_C, tables, model.tokenizer, indices = indices, 
                                                token_limit = token_limit) # we initially pass in an empty index, and use the computed index afterwards

        for it in range(hparams['n_samples']):
            raw_output = model(input = prompt)
            stance = prompt_gen.parse_stance(raw_output)
            pred_evidence = prompt_gen.parse_evidence(raw_output)
            
            ress.append({
                'claim_id': idx,
                'iter': it,
                'label': label,
                'claim': claim,
                'prompt': prompt,
                'raw_output': raw_output,
                'pred_label': stance,
                # if it doesn't return any evidence, adjust to NEI regardless of what it says
                'pred_label_adjusted': 'N' if pred_evidence is None or len(pred_evidence.strip()) == 0 else stance,
                'pred_evidence': pred_evidence
            })
        
        if hparams['debug']:
            break

    completed_hadms.append(hadm_id)
    if not hparams['no_checkpoint']:
        pickle.dump({
            'ress': ress,
            'completed_hadms': completed_hadms
        }, (out_dir/'checkpoint.pkl').open('wb'))
    
res = pd.DataFrame(ress)
res.to_pickle(out_dir/'res.pkl')

with open(os.path.join(out_dir, 'done'), 'w') as f:
    f.write('done')
