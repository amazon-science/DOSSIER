
from fact_check.constants import MIMIC3_DIR
import pandas as pd
import polars as pl
from pathlib import Path
from fact_check.data.mimic3_dataset import MIMIC3_Dataset
from collections import defaultdict
from medcat.vocab import Vocab
from medcat.cdb import CDB
from medcat.cat import CAT
from medcat.meta_cat import MetaCAT
from fact_check.constants import UMLS_DIR, snomed_umls_mapping
from fact_check.data.umls_api_tagger import UMLS_API_Tagger 
import os

def load_taggers(paths, umls_api_n_concepts = None, prompt = 'full'):
    cats = {}
    for i in paths:
        i = str(i)
        if 'snomed' in i.lower():
            unzip = i
            vocab = Vocab.load(os.path.join(unzip, 'vocab.dat'))
            cdb = CDB.load(os.path.join(unzip, 'cdb.dat'))
            cdb.config.linking.filters.cuis = set()
            cdb.config.general.spacy_model = os.path.join(unzip, 'spacy_model')
            mc_status = MetaCAT.load(os.path.join(unzip, 'meta_Status/'))

            cats[i] = {'type': 'snomed', 
                       'cat': CAT(cdb=cdb, config=cdb.config, vocab=vocab, meta_cats=[mc_status]), 
                       'mapping': snomed_umls_mapping
            }
        elif 'umls_api' in i.lower():
            cats[i] = {
                'type': 'umls_api',
                'cat': UMLS_API_Tagger(umls_api_n_concepts, tag_subtypes = prompt == 'no_gkg')
            }
        else:
            cats[i] = {
                'type': 'default',
                'cat': CAT.load_model_pack(i)
            }
    return cats

def data_select(hparams, claims_override = None):
    '''
        hparams (dict): dictionary of parameters
        claims_override (pd.DataFrame): override the claims for the 
    '''
    claims = pd.read_csv(Path(hparams['claim_df_path'])) if claims_override is None else claims_override
    all_adms = claims['HADM_ID'].unique()
    if 'cui_mapping_path' in hparams:
        if str(hparams['cui_mapping_path']).endswith('pkl'):
            cui_mapping = pd.read_pickle(hparams['cui_mapping_path'])
        elif str(hparams['cui_mapping_path']).endswith('csv'):
            cui_mapping = pd.read_csv(hparams['cui_mapping_path'])
        cui_mapping = defaultdict(lambda: None, cui_mapping.set_index('ID')['cui'].to_dict())
    else:
        cui_mapping = None

    dfs = {}
    adms = pd.read_csv(MIMIC3_DIR/'ADMISSIONS.csv')
    adms = adms[adms.HADM_ID.isin(all_adms)]
    adms['ADMITTIME'] = pd.to_datetime(adms['ADMITTIME'])
    adms['DIAGNOSIS'] = adms['DIAGNOSIS'].apply(lambda x: str(x).split(';'))
    if cui_mapping:
        adms['ADMIT_CUI'] = adms['DIAGNOSIS'].apply(lambda x: [cui_mapping[i] for i in x])
    dfs['adms'] = adms

    q = (pl.scan_csv(MIMIC3_DIR/'LABEVENTS.csv')
        .filter(pl.col("HADM_ID").cast(pl.Int64).is_in(all_adms))
        )
    labs = (q.collect(streaming=True).to_pandas()
            .merge(pd.read_csv(MIMIC3_DIR/'D_LABITEMS.csv'), on = 'ITEMID')
            .rename(columns = {
            'ROW_ID_x': 'ROW_ID'
        }))
    
    labs['CHARTTIME'] = pd.to_datetime(labs['CHARTTIME'])
    if cui_mapping:
        labs['CUI'] = labs['ITEMID'].astype(str).map(cui_mapping)
        labs = labs.dropna(subset = ['CUI'])
    
    q = (pl.scan_csv(MIMIC3_DIR/'CHARTEVENTS.csv', dtypes = {'VALUE': pl.Utf8})
        .filter(pl.col("HADM_ID").cast(pl.Int64).is_in(all_adms))
        .filter(~pl.col('CHARTTIME').is_null())
        )
    vits = (q.collect(streaming=True).to_pandas()
        .merge(pd.read_csv(MIMIC3_DIR/'D_ITEMS.csv'), on = 'ITEMID')
        .rename(columns = {
            'ROW_ID_x': 'ROW_ID'
        }))
    vits['CHARTTIME'] = pd.to_datetime(vits['CHARTTIME'])
    vits = vits[~vits.CATEGORY.isin(['Labs', 'ADT', 'Restraint/Support Systems', 'Alarms'])] 
    vits = vits[~vits.LABEL.str.lower().str.contains('alarm')]
    if cui_mapping:
        vits['CUI'] = vits['ITEMID'].astype(str).map(cui_mapping)
        vits = vits.dropna(subset = ['CUI'])
    
    q = (pl.scan_csv(MIMIC3_DIR/'INPUTEVENTS_MV.csv', dtypes = {'TOTALAMOUNT': pl.Utf8, 'AMOUNT': pl.Utf8})
        .filter(pl.col("HADM_ID").is_in(all_adms))
        )
    inputs = (q.collect(streaming=True).to_pandas()).merge(pd.read_csv(MIMIC3_DIR/'D_ITEMS.csv'), on = 'ITEMID')[
        ['ROW_ID_x', 'HADM_ID', 'SUBJECT_ID', 'ICUSTAY_ID', 'STARTTIME', 'ENDTIME', 'ITEMID', 'AMOUNT', 'RATE', 'CATEGORY',
            'ORIGINALAMOUNT', 'ORIGINALRATE', 'LABEL', 'UNITNAME', 'ABBREVIATION', 'AMOUNTUOM', 'RATEUOM']]
    
    q2 = (pl.scan_csv(MIMIC3_DIR/'INPUTEVENTS_CV.csv', dtypes = {'TOTALAMOUNT': pl.Utf8, 'AMOUNT': pl.Utf8})
        .filter(pl.col("HADM_ID").is_in(all_adms))
        )
    inputs2 = (q2.collect(streaming=True).to_pandas()).merge(pd.read_csv(MIMIC3_DIR/'D_ITEMS.csv'), on = 'ITEMID')[
        ['ROW_ID_x', 'HADM_ID',  'SUBJECT_ID', 'ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'AMOUNT', 'RATE', 'CATEGORY',
            'ORIGINALAMOUNT', 'ORIGINALRATE', 'LABEL', 'UNITNAME', 'ABBREVIATION', 'AMOUNTUOM', 'RATEUOM']].rename(columns = {'CHARTTIME': 'STARTTIME'})
    
    inputs = pd.concat((inputs, inputs2), ignore_index = True).rename(columns = {'ROW_ID_x': 'ROW_ID'})
    inputs['STARTTIME'] = pd.to_datetime(inputs['STARTTIME'])
    if cui_mapping:
        inputs['CUI'] = inputs['ITEMID'].astype(str).map(cui_mapping)
        inputs = inputs[~inputs.ITEMID.isin([225943, 30270])] # "Solution", "E 20 FS PO"
        inputs = inputs.dropna(subset = ['CUI'])

    dfs['labs'] = labs
    dfs['vits'] = vits
    dfs['inputs'] = inputs

    for i in dfs:
        if isinstance(dfs[i].iloc[0]['HADM_ID'], str):
            dfs[i]['HADM_ID'] = dfs[i]['HADM_ID'].astype(int)
        dfs[i] = dfs[i].set_index('HADM_ID')
    
    taggers = load_taggers(hparams['medcat_model_paths'], hparams['umls_api_n_concepts'], hparams['prompt'])
    global_kg = pd.read_csv(hparams['semmeddb_path'])
    global_kg = global_kg[global_kg.PREDICATE.isin(hparams['subset_predicates'])]

    ignore_cuis = set(pd.read_csv(Path(hparams['generics_path']), header = None)[1].values) # concepts to ignore during tagging

    return MIMIC3_Dataset(claims, dfs, global_kg, taggers, ignore_cuis, cui_mapping, hparams)