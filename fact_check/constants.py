from pathlib import Path
import pandas as pd
import pickle
import os
from collections import defaultdict

MIMIC3_DIR = Path('/data/healthy-ml/gobi1/data/mimiciii')
MIMIC4_DIR = Path('/data/healthy-ml/gobi1/data/mimiciv/2.2')
DATA_DIR = Path('/data/healthy-ml/scratch/haoran/clinical_fact_check/data')
UMLS_DIR = DATA_DIR/'2023AA/META/'

model_mapping = {
    'gatortron': 'UFNLP/gatortron-medium',
    'mpnet-v2': 'sentence-transformers/all-mpnet-base-v2'
}

max_seq_lengths = {
    'gatortron': 512,
    'mpnet-v2': 512
}

label_mapping = {
    0: 'F',
    1: 'T',
    2: 'N'
}

restrict_types = {
    'Vital': ['Clinical Attribute', 'Finding', 'Organism Attribute', 'Laboratory Procedure', 'Laboratory or Test Result', 'Diagnostic Procedure'],
    'Input': ['Pharmacologic Substance', 'Clinical Drug'],
    'Lab': ['Laboratory Procedure', 'Laboratory or Test Result'],
    'Admission': ['Disease or Syndrome', 'Injury or Poisoning', 'Pathologic Function', 'Anatomical Abnormality',
                 'Finding', 'Sign or Symptom', 'Mental or Behavioral Dysfunction', 'Congenital Abnormality',
                 'Acquired Abnormality'],
    'Other': [
        'Clinical Attribute', 'Finding', 'Organism Attribute', 'Laboratory Procedure', 'Laboratory or Test Result', 'Diagnostic Procedure',
        'Pharmacologic Substance', 'Clinical Drug', 'Therapeutic or Preventive Procedure'
    ]
}

restrict_types_set = set([j for i in restrict_types for j in restrict_types[i]])

# load from cache or recompute two mappings
cache_dir = Path(os.path.abspath(os.path.dirname(__file__)))/'cache'
cache_dir.mkdir(exist_ok=True)

if (cache_dir/'umls_cat_mapping.pkl').is_file():
    umls_cat_mapping = pd.read_pickle(cache_dir/'umls_cat_mapping.pkl')
else:
    umls_cat_mapping = pd.read_csv(UMLS_DIR/'MRSTY.RRF', sep = '|', header = None)[[0, 3]]
    umls_cat_mapping.columns = ['CUI', 'Category']
    umls_cat_mapping = umls_cat_mapping.set_index('CUI')['Category'].groupby(level = 0).agg(list).to_dict()
    pickle.dump(umls_cat_mapping, Path(cache_dir/'umls_cat_mapping.pkl').open('wb'))

umls_cat_mapping = defaultdict(list, umls_cat_mapping)

if (cache_dir/'snomed_umls_mapping.pkl').is_file():
    snomed_umls_mapping = pd.read_pickle(cache_dir/'snomed_umls_mapping.pkl')
else:
    snomed_umls_mapping = pd.read_csv(Path(UMLS_DIR)/'MRCONSO.RRF', sep = '|', header = None)
    snomed_umls_mapping = snomed_umls_mapping[snomed_umls_mapping[11] == "SNOMEDCT_US"][[0, 13]].drop_duplicates(subset = [13])
    snomed_umls_mapping = snomed_umls_mapping.set_index(13)[0].to_dict()
    pickle.dump(snomed_umls_mapping, Path(cache_dir/'snomed_umls_mapping.pkl').open('wb'))
