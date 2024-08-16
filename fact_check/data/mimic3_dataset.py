from torch.utils.data import Dataset
from fact_check.constants import restrict_types_set, umls_cat_mapping
import numpy as np
import pandas as pd
import warnings
import torch
import networkx as nx

def check_numeric(x):
    try:
        float(x)
        return True
    except:
        return False
    
    
class MIMIC3_Dataset(Dataset):
    def __init__(self, claims, dfs, global_kg, taggers, ignore_cuis, cui_mapping, hparams):
        self.hparams = hparams
        self.claims = claims
        self.dfs = dfs
        self.global_kg = global_kg
        self.ignore_cuis = ignore_cuis
        self.cui_mapping = cui_mapping
        self.cats = taggers

    def make_tables(self, hadm_id, t = pd.Timestamp('2262-04-11'), rel_t = None, subset_numeric = True,
                    remove_generic = True):
        '''
        Given a hadm_id, returns the local patient tables and the global KG.
        t: claim time as a TimeStamp -- all evidence in resulting table will be before this time
        rel_t: claim time as a float in hours relative to admission time -- takes priority over t
        remove_generic: remove generic nodes from semmeddb (i.e. those with novelty = 0)
        '''        
        adm_rows = self.dfs['adms'].loc[[hadm_id]].explode(['DIAGNOSIS', 'ADMIT_CUI'])
        admit_time = adm_rows['ADMITTIME'].iloc[0]
        adm_rows = adm_rows.dropna(subset = ['ADMIT_CUI', 'DIAGNOSIS'])

        if rel_t is not None:
            t = admit_time + pd.Timedelta(hours = rel_t)

        sub_dfs = {} # tables
        sub_dfs['adm'] = adm_rows.reset_index()[['ADMITTIME', 'DIAGNOSIS', 'ADMIT_CUI']].assign(rel_t = 0.0)

        if 'labs' in self.dfs and hadm_id in self.dfs['labs'].index:
            if subset_numeric:
                self.dfs['labs'] = self.dfs['labs'].dropna(subset = ['VALUENUM'])
                self.dfs['labs']['VALUE'] = self.dfs['labs']['VALUENUM']
                self.dfs['labs']['VALUE'] = self.dfs['labs']['VALUE'].astype('float')

            lab_evidence = self.dfs['labs'].loc[[hadm_id]].query('CHARTTIME < @t').rename(columns = {
                                   'CHARTTIME': 't',
                                   'CUI': 'CUI',
                                    'VALUE': 'value',
                                    'LABEL': 'str_label',
                                    'VALUEUOM': 'units'
                               }).reset_index(drop = True).dropna(subset = ['str_label', 'value'])
            
            lab_evidence['rel_t'] = (lab_evidence['t'] - admit_time)/pd.Timedelta(hours=1)
            lab_evidence = lab_evidence[lab_evidence['rel_t'] >= 0]
            lab_evidence = lab_evidence[lab_evidence.t <= t]
            sub_dfs['lab'] = lab_evidence
            
        if 'inputs' in self.dfs and hadm_id in self.dfs['inputs'].index: 
            inputs_evidence = self.dfs['inputs'].loc[[hadm_id]].query('STARTTIME < @t')
            inputs_evidence['AMOUNT_TO_USE'] = inputs_evidence.apply(lambda row: row['AMOUNT'] if row['ORIGINALAMOUNT'] is None else row['ORIGINALAMOUNT'],
                                                                     axis = 1)
            inputs_evidence = inputs_evidence.dropna(subset = ['AMOUNT_TO_USE', 'LABEL']).rename(columns = {
                                   'STARTTIME': 't',
                                   'CUI': 'CUI',
                                    'AMOUNT_TO_USE': 'value',
                                    'LABEL': 'str_label',
                                    'AMOUNTUOM': 'units'
                               }).reset_index(drop = True)
            
            if len(inputs_evidence) > 0:            
                if subset_numeric:
                    inputs_evidence = inputs_evidence[inputs_evidence['value'].apply(check_numeric)].dropna(subset = ['value'])
                    inputs_evidence['value'] = inputs_evidence['value'].astype('float')
                inputs_evidence['rel_t'] = (inputs_evidence['t'] - admit_time)/pd.Timedelta(hours=1)
                inputs_evidence = inputs_evidence[inputs_evidence['rel_t'] >= 0]
                inputs_evidence = inputs_evidence[inputs_evidence['t'] <= t]
                
            sub_dfs['input'] = inputs_evidence

        if 'vits' in self.dfs and hadm_id in self.dfs['vits'].index:
            if subset_numeric:
                self.dfs['vits'] = self.dfs['vits'].dropna(subset = ['VALUENUM'])
                self.dfs['vits']['VALUE'] = self.dfs['vits']['VALUENUM']
                self.dfs['vits']['VALUE'] = self.dfs['vits']['VALUE'].astype('float')

            vits_evidence = self.dfs['vits'].loc[[hadm_id]].query('CHARTTIME < @t').rename(columns = {
                                   'CHARTTIME': 't',
                                   'CUI': 'CUI',
                                    'VALUE': 'value',
                                    'LABEL': 'str_label',
                                    'VALUEUOM': 'units'
                               }).dropna(subset = ['str_label', 'value'])
            if not subset_numeric:
                vits_evidence = vits_evidence[vits_evidence['value'].str.lower() != 'none'].reset_index(drop = True)
           
            vits_evidence['rel_t'] = (vits_evidence['t'] - admit_time)/pd.Timedelta(hours=1)
            vits_evidence = vits_evidence[vits_evidence['rel_t'] >= 0]
            vits_evidence = vits_evidence[vits_evidence['t'] <= t]
            sub_dfs['vit'] = vits_evidence
        
        for i in sub_dfs:
            sub_dfs[i] = sub_dfs[i].assign(evidence_type = 'local')
        
        if remove_generic:
            # find all triplets in global KG that contain zero or one node from the set, and add those edges
            # for triplets in global KG that contain both nodes from the set, only add if they flow forward in time
            self.global_kg = self.global_kg[(self.global_kg['OBJECT_NOVELTY'] == 1) & (self.global_kg['SUBJECT_NOVELTY'] == 1)]

        sub_dfs['global_kg'] = self.global_kg
        return sub_dfs

    def tag_claim(self, claim_str):
        all_entities, cui_list = [], []
        for _, cat in list(self.cats.items()): 
            entities = cat['cat'].get_entities(claim_str)['entities']
            for i in entities:
                if cat['type'] == 'snomed':                
                    if entities[i]['cui'] in cat['mapping']:
                        entities[i]['cui'] = cat['mapping'][entities[i]['cui']]
                    else:
                        continue
                        
                if (entities[i]['cui'] not in cui_list and entities[i]['cui'] not in self.ignore_cuis): 
                    entities[i]['type'] = umls_cat_mapping[entities[i]['cui']]
                    if len(set(entities[i]['type']).intersection(restrict_types_set)) > 0:
                        all_entities.append(entities[i])
                        cui_list.append(entities[i]['cui'])
        
        return all_entities, cui_list

    def __len__(self):
        return len(self.claims)
    