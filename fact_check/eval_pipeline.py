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
from fact_check.constants import DATA_DIR
from fact_check.prompting import prompter
from fact_check.utils import Tee, path_serial
from fact_check.data import data_select, table_utils
from fact_check.prompting import claude_query
from fact_check.models.llms import LLM
from tqdm import tqdm
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--claim_df_path', default = DATA_DIR/'claims.csv')
parser.add_argument('--prompt', default = 'full', choices = ['full', 'no_umls', 'no_gkg', 'neither'], help = 'ablation to use; full = no ablations')
parser.add_argument('--llm', default = 'claude', choices = ['claude', 'codellama', 'claude-v1', 'medalpaca', 'clinicalcamel'],
                    help = 'llm for text2sql; `claude` is `claude-2` and `claude-v1` is `claude-instant-1`.')
parser.add_argument('--temperature', default = 0.0, type = float)
parser.add_argument('--subset_adms', default = None, help = "take the first k adms instead of using all of them", type = int)
parser.add_argument('--n_samples', default = 1, type = int, help = '# samples in generation')
parser.add_argument('--medcat_model_paths', default = [DATA_DIR/'medcat_mimic_small/',
                                                       DATA_DIR/'medcat_mimic_snomed/',
                                                       'umls_api'], 
                    nargs = '+', type = str,help='used to extract entities from claim')
parser.add_argument('--cui_mapping_path', default = Path(__file__).parent.absolute()/'data/mimic3_cui_mapping.csv')
parser.add_argument('--semmeddb_path', default = DATA_DIR/'semmeddb/semmeddb_processed_10.csv')
parser.add_argument('--generics_path', default = DATA_DIR/'semmeddb/semmedVER43_2023_R_GENERIC_CONCEPT.csv')
parser.add_argument('--subset_predicates', default = ['ISA', 'TREATS', 'PREVENTS'], nargs = '+', type = str)
parser.add_argument('--umls_api_n_concepts', type = int, default = 1, help = 'For the UMLS API tagger, # of CUIs to return from UMLS API per Claude tagged entity')
parser.add_argument('--no_checkpoint', action = 'store_true', help = 'Do not checkpoint. Otherwise, checkpoint after every adm')
parser.add_argument('--output_dir', type = str, required = True)
parser.add_argument('--no_examples', action = 'store_true', help = 'no zero shot examples')
parser.add_argument('--seed', type = int, default = 42)
parser.add_argument('--debug', action = 'store_true')
args = parser.parse_args()
hparams = vars(args)

hparams['include_all_evidence'] = False

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
    adms = adms.iloc[:5]

print("Total # of claims: " + str(len(claims[claims.HADM_ID.isin(adms.index)])))
prompt_gen = prompter.BasePrompt(ds.tag_claim, list(ds.cui_mapping.values()), add_examples = not args.no_examples, add_sem_types = True, 
                                    prompt = hparams['prompt'], target_llm = hparams['llm'])

if hparams['llm'].startswith('claude'):
    query_fn = lambda x: claude_query.query(x, temperature = hparams['temperature'], model = {
        'claude': 'claude-2',
        'claude-v1': 'claude-instant-1'
    }[args.llm])
else:    
    model = LLM('medalpaca_p2' if hparams['llm'] == 'medalpaca' else  hparams['llm'], max_new_tokens = 1024, temperature = hparams['temperature'])
    query_fn = lambda x: model(input = x)

if (out_dir/'checkpoint.pkl').is_file():
    checkpoint = pickle.load((out_dir/'checkpoint.pkl').open('rb'))
    ress = checkpoint['ress']
    completed_hadms = checkpoint['completed_hadms']
    print(f"Loaded checkpoint with {len(completed_hadms)} adms completed!")
else:
    ress = []
    completed_hadms = []

conn, db_path = table_utils.make_sqlite_db(in_memory = False)
conn.close()

for hadm_id in tqdm(adms.index):
    if hadm_id in completed_hadms:
        continue
    
    conn = table_utils.open_sqlite_conn(db_path)
    existing_tables = pd.read_sql_query('select name from sqlite_schema where type = "table";', conn)['name'].values
    tables = ds.make_tables(hadm_id) # all claims are made with all available data, so no need to restrict t
    tables = table_utils.subset_to_sql_schema(tables)    
    table_utils.add_tables_to_sqlite_db(tables, conn, preprocess = {
                'Global_KG': table_utils.add_identity_edges
            }, skip = ['Global_KG'] if 'Global_KG' in existing_tables else [])
    conn.close()

    claims_i = claims[claims.HADM_ID == hadm_id]
    for c, (idx, row) in enumerate(tqdm(claims_i.iterrows(), total = len(claims_i))):
        claim = row['claim']
        t_C = row['t_C']
        label = row['label']
        prompt = prompt_gen.get_prompt(claim, t_C, tables)

        for it in range(hparams['n_samples']):
            raw_output = query_fn(prompt)
            sql_output = prompt_gen.parse_answer(raw_output)
            if hparams['debug']:
                print(sql_output)
            if sql_output:
                try:
                    # out_table = table_utils.run_query(sql_output, conn)
                    out_table = table_utils.fetch_data_with_timeout(sql_output, db_path, timeout = 600)
                    stance = prompt_gen.parse_stance(raw_output, out_table)
                except Exception as e:
                    error = True
                    error_str = str(e)
                    stance = 'N'
                    out_table = None
                else:
                    error = False
                    error_str = None
            else:
                error = True
                error_str = 'No valid SQL output.'
                stance = 'N'
                out_table = None

            len_out_table = len(out_table) if out_table is not None else 0

            if out_table is not None and len(out_table) >= 1000:
                out_table = out_table.iloc[:1000] # when SQL query is catastrophically bad, avoid storing huge tables
            
            ress.append({
                'claim_id': idx,
                'iter': it,
                'label': label,
                'claim': claim,
                'prompt': prompt,
                'raw_output': raw_output,
                'sql_output': sql_output,
                'pred_label': stance,
                'pred_out_table': out_table,
                'len_pred_out_table': len_out_table,
                'pred_label_with_gold_bounds': prompt_gen._parse_stance(row['lower'], str(row['upper']), row['stance'], len_out_table),
                'error_str': error_str
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
