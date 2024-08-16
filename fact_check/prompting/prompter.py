import anthropic
import os
import re
from textwrap import dedent
from fact_check.prompting.prompts import full_prompts, no_gkg_prompts, no_umls_prompts, direct_llm_prompts, neither_prompts
import copy
import numpy as np
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import faiss

def negate(s):
    return {
        'T': 'F',
        'F': 'T',
        'N': 'N'
    }[s]

class Prompt():
    def __init__(self, tag_fn, cuis_in_ehr):
        self.tag_fn = tag_fn
        self.cuis_in_ehr = cuis_in_ehr

class BasePrompt(Prompt):
    def __init__(self, tag_fn, cuis_in_ehr, add_examples = True, add_sem_types = True, prompt = 'full', target_llm = 'claude'):
        '''
        Generates a prompt for SQL generation to feed into a downstream LLM

        tag_fn (func): reference to mimic3_dataset.tag_claim
        cuis_in_ehr (list): list of CUIs that are used in MIMIC
        add_examples (bool): whether we should concatenate examples to the prompt
        add_sem_types (bool): whether we should pass in the semantic type for each tagged CUI (e.g. "Organic Chemical")
        prompt (str): one of "full", "no_umls", "no_gkg", corresponding to our full pipeline and ablations.
        target_llm (str): which LLM the prompt will be passed into
        '''
        self.add_examples = add_examples
        self.add_sem_types = add_sem_types
        self.prompt = prompt
        self.target_llm = target_llm
        super().__init__(tag_fn, cuis_in_ehr)

    def get_prompt(self, CLAIM, t_C, tables):
        if self.prompt == 'full':
            self.HEADER = full_prompts.HEADER
            self.SCHEMA = full_prompts.SCHEMA
            prompt = self.get_prompt_full(CLAIM, t_C, tables)
        elif self.prompt == 'no_gkg':
            self.HEADER = no_gkg_prompts.HEADER
            self.SCHEMA = no_gkg_prompts.SCHEMA
            prompt = self.get_prompt_no_gkg(CLAIM, t_C, tables)
        elif self.prompt == 'no_umls':
            self.HEADER = no_umls_prompts.HEADER
            self.SCHEMA = no_umls_prompts.SCHEMA
            prompt = self.get_prompt_no_umls(CLAIM, t_C, tables)
        elif self.prompt == 'neither':
            self.HEADER = neither_prompts.HEADER
            self.SCHEMA = neither_prompts.SCHEMA
            prompt = self.get_prompt_neither(CLAIM, t_C, tables)
        else:
            raise NotImplementedError(self.prompt)
        
        if self.target_llm == 'codellama':
            prompt = self.parse_codellama_prompt(prompt)
        
        return prompt

    def parse_codellama_examples(self, exs):
        exs = exs.replace("Here are some examples:\n", "", 1)
        exs = exs.replace("\nH: ", "\n")
        exs = exs.replace("\nA: ", "\n")
        exs = exs.replace("<example>", "Example:")
        exs = exs.replace("</example>", "")
        return exs

    def parse_codellama_prompt(self, st):\
        return st

    def get_prompt_full(self, CLAIM, t_C, tables):
        entities, all_cuis = self.tag_fn(CLAIM)

        if self.add_sem_types:
            CUIs = ', '.join([str((i['pretty_name'], i['cui'], i['type'])) for i in entities])
        else:
            CUIs = ', '.join([str((i['pretty_name'], i['cui'])) for i in entities])

        CUI_subj_counts = ', '.join([str((i, int((tables['Global_KG']['Subject_CUI'] == i).sum()))) for i in all_cuis])
        CUI_obj_counts = ', '.join([str((i, int((tables['Global_KG']['Object_CUI'] == i).sum()))) for i in all_cuis])
        CUI_subj = ', '.join([str(i) for i in all_cuis if int((tables['Global_KG']['Subject_CUI'] == i).sum()) > 0])
        CUI_obj = ', '.join([str(i) for i in all_cuis if int((tables['Global_KG']['Object_CUI'] == i).sum()) > 0])
        CUI_sub = ', '.join([i for i in all_cuis if i in self.cuis_in_ehr])
        unique_predicates = tables['Global_KG']['Predicate'].unique()

        PROBLEM_SPECS = full_prompts.PROBLEM_SPECS.format(unique_predicates.tolist())
        
        if self.add_sem_types:
            PRIOR_KNOWLEDGE = dedent(f'''\
            You are given the following prior knowledge:
            - Potentially relevant CUIs found in the claim, along with their semantic types: {CUIs}
            - Out of the potentially relevant CUIs, the following appear at least once in the Subject_CUI column of Global_KG: {CUI_subj}
            - Out of the potentially relevant CUIs, the following appear at least once in the Object_CUI column of Global_KG: {CUI_obj}''')
        else:
            PRIOR_KNOWLEDGE = dedent(f'''\
            You are given the following prior knowledge:
            - Potentially relevant CUIs found in the claim: {CUIs}
            - Out of the potentially relevant CUIs, the following appear at least once in the Subject_CUI column of Global_KG: {CUI_subj}
            - Out of the potentially relevant CUIs, the following appear at least once in the Object_CUI column of Global_KG: {CUI_obj}''')

        CLAIM_PRE = f'Claim made at t={t_C}: '

        if self.target_llm.startswith('claude'):
            HUMAN_PROMPT = anthropic.HUMAN_PROMPT
            AI_PROMPT = anthropic.AI_PROMPT
            EXAMPLES = full_prompts.EXAMPLES
        elif self.target_llm == 'codellama':
            HUMAN_PROMPT = '\n<s>[INST]'
            AI_PROMPT = '[/INST]\n'
            EXAMPLES = self.parse_codellama_examples(full_prompts.EXAMPLES)            
            PRIOR_KNOWLEDGE = '<s>[INST]\n'+ PRIOR_KNOWLEDGE
        else:
            HUMAN_PROMPT = ''
            AI_PROMPT = ''
            EXAMPLES = full_prompts.MINI_EXAMPLES

        if self.add_examples:
            PROMPT = f"{HUMAN_PROMPT} {self.HEADER}\n\n{self.SCHEMA}\n\n{PROBLEM_SPECS}\n\n{EXAMPLES}\n\n{PRIOR_KNOWLEDGE}\n\n{CLAIM_PRE}{CLAIM}\n{AI_PROMPT}"
        else:
            PROMPT = f"{HUMAN_PROMPT} {self.HEADER}\n\n{self.SCHEMA}\n\n{PROBLEM_SPECS}\n\n{PRIOR_KNOWLEDGE}\n\n{CLAIM_PRE}{CLAIM}\n{AI_PROMPT}"
       
        return PROMPT
    
    def get_prompt_no_gkg(self, CLAIM, t_C, tables):
        assert self.target_llm in ['claude', 'claude-v1', 'codellama']
        entities, all_cuis = self.tag_fn(CLAIM)

        if self.add_sem_types:
            CUIs = ', '.join([str((i['pretty_name'], i['cui'], i['type'])) for i in entities])
        else:
            CUIs = ', '.join([str((i['pretty_name'], i['cui'])) for i in entities])

        PROBLEM_SPECS = no_gkg_prompts.PROBLEM_SPECS
        EXAMPLES = no_gkg_prompts.EXAMPLES

        if self.add_sem_types:
            PRIOR_KNOWLEDGE = dedent(f'''\
            You are given the following prior knowledge:
            - Potentially relevant CUIs found in the claim, along with their semantic types: {CUIs}''')
        else:
            PRIOR_KNOWLEDGE = dedent(f'''\
            You are given the following prior knowledge:
            - Potentially relevant CUIs found in the claim: {CUIs}''')

        CLAIM_PRE = f'Claim made at t={t_C}: '

        if self.target_llm.startswith('claude'):
            HUMAN_PROMPT = anthropic.HUMAN_PROMPT
            AI_PROMPT = anthropic.AI_PROMPT
        elif self.target_llm == 'codellama':
            HUMAN_PROMPT = '\n<s>[INST]'
            AI_PROMPT = '[/INST]\n'
            EXAMPLES = self.parse_codellama_examples(EXAMPLES)            
            PRIOR_KNOWLEDGE = '<s>[INST]\n'+ PRIOR_KNOWLEDGE

        if self.add_examples:
            PROMPT = f"{HUMAN_PROMPT} {self.HEADER}\n\n{self.SCHEMA}\n\n{PROBLEM_SPECS}\n\n{EXAMPLES}\n\n{PRIOR_KNOWLEDGE}\n\n{CLAIM_PRE}{CLAIM}\n{AI_PROMPT}"
        else:
            PROMPT = f"{HUMAN_PROMPT} {self.HEADER}\n\n{self.SCHEMA}\n\n{PROBLEM_SPECS}\n\n{PRIOR_KNOWLEDGE}\n\n{CLAIM_PRE}{CLAIM}\n{AI_PROMPT}"

        return PROMPT
    
    def get_prompt_no_umls(self, CLAIM, t_C, tables):
        assert self.target_llm in ['claude', 'claude-v1', 'codellama']
        unique_predicates = tables['Global_KG']['Predicate'].unique()

        PROBLEM_SPECS = no_umls_prompts.PROBLEM_SPECS.format(unique_predicates.tolist())
        EXAMPLES = no_umls_prompts.EXAMPLES
        CLAIM_PRE = f'Claim made at t={t_C}: '

        if self.target_llm.startswith('claude'):
            HUMAN_PROMPT = anthropic.HUMAN_PROMPT
            AI_PROMPT = anthropic.AI_PROMPT
        elif self.target_llm == 'codellama':
            HUMAN_PROMPT = '\n<s>[INST]'
            AI_PROMPT = '[/INST]\n'
            EXAMPLES = self.parse_codellama_examples(EXAMPLES)

        if self.add_examples:
            PROMPT = f"{HUMAN_PROMPT} {self.HEADER}\n\n{self.SCHEMA}\n\n{PROBLEM_SPECS}\n\n{EXAMPLES}\n\n{CLAIM_PRE}{CLAIM}\n{AI_PROMPT}"
        else:
            PROMPT = f"{HUMAN_PROMPT} {self.HEADER}\n\n{self.SCHEMA}\n\n{PROBLEM_SPECS}\n\n{CLAIM_PRE}{CLAIM}\n{AI_PROMPT}"

        return PROMPT

    def get_prompt_neither(self, CLAIM, t_C, tables):
        assert self.target_llm in ['claude', 'claude-v1', 'codellama']

        PROBLEM_SPECS = neither_prompts.PROBLEM_SPECS
        EXAMPLES = neither_prompts.EXAMPLES
        CLAIM_PRE = f'Claim made at t={t_C}: '

        if self.target_llm.startswith('claude'):
            HUMAN_PROMPT = anthropic.HUMAN_PROMPT
            AI_PROMPT = anthropic.AI_PROMPT
        elif self.target_llm == 'codellama':
            HUMAN_PROMPT = '\n<s>[INST]'
            AI_PROMPT = '[/INST]\n'
            EXAMPLES = self.parse_codellama_examples(EXAMPLES)

        if self.add_examples:
            PROMPT = f"{HUMAN_PROMPT} {self.HEADER}\n\n{self.SCHEMA}\n\n{PROBLEM_SPECS}\n\n{EXAMPLES}\n\n{CLAIM_PRE}{CLAIM}\n{AI_PROMPT}"
        else:
            PROMPT = f"{HUMAN_PROMPT} {self.HEADER}\n\n{self.SCHEMA}\n\n{PROBLEM_SPECS}\n\n{CLAIM_PRE}{CLAIM}\n{AI_PROMPT}"

        return PROMPT

    
    def parse_answer(self, completion):
        ans = re.findall(r"<sql>([\s\S]*)<\/sql>", str(completion))
        if len(ans):
            return ans[0].strip()
        if self.target_llm == 'codellama':
            ans2 = re.findall(r"```([\s\S]*)```", str(completion))
            if len(ans2):
                return ans2[0].strip()
        return None

    def parse_stance(self, completion, table):
        ans = re.findall(r"<stance>([a-zA-Z]*)<\/stance>", str(completion))
        lower = re.findall(r"<lower>([0-9]*)<\/lower>", str(completion))
        upper = re.findall(r"<upper>([0-9]*)<\/upper>", str(completion))

        if len(ans) > 1: # if they use the tags multiple times, select the first one that is non-empty
            ans = [i for i in ans if len(i) > 0]
        
        if len(lower) > 1:
            lower = [i for i in lower if len(i) > 0]

        if len(upper) > 1: # if any of the upper tags are non-infinite, pick it
            temp = [i for i in upper if len(i) > 0] 
            upper = temp if len(temp) > 0 else upper  

        if (not len(ans)) or (ans[0].strip() not in ['T', 'F']) or (table is None) or (not len(upper)) or (not len(lower)) or (not lower[0].isnumeric()) or (len(table) == 0):
            return 'N'
        stance = ans[0].strip()
        return self._parse_stance(int(lower[0]), upper[0], stance, len(table))
            
    @staticmethod       
    def _parse_stance(lower, upper, stance, len_table):
        if len_table < lower:
            return 'N'
        
        if len(upper) == 0 or upper.lower() == 'inf': # unbounded, return stance len_table >= lower
            return stance
        else: # upper bound is a scalar
            if not upper.isnumeric():
                return 'N'
            upper = int(upper)
            if len_table <= upper:
                return 'N' # return NEI since there could be more unobserved evidence that push it over the upper bound
            else:
                return negate(stance) # we have a lot of evidence

class DirectLLMPrompter():
    def __init__(self, include_examples = True, include_vitals = True, select_rows = 'rag', rag_type = 'bm25', rag_with_kg = False,
                                        kg_path = None, kg_emb_path = None, kg_subset_predicates = None):
        '''
        select_rows: 'rag' or 'random'. 
            If 'random', will randomly subset tables to meet token limit
            If 'rag', will select rows that best match the claim by BM25
        '''
        self.include_examples = include_examples
        self.include_vitals = include_vitals
        self.select_rows = select_rows
        self.rag_type = rag_type
        self.rag_with_kg = rag_with_kg

        if rag_type == 'knn':
            self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to('cuda')

        if rag_with_kg:
            self.kg_df = pd.read_csv(kg_path)
            idxs = self.kg_df.reset_index(drop = True).PREDICATE.isin(kg_subset_predicates).index 
            self.kg_df = self.kg_df.iloc[idxs]

            def clean_predicate(st):
                    st = st.replace('_', ' ')
                    if st == "ISA":
                        return 'IS A'
                    return st

            self.kg_df['predicate_clean'] = self.kg_df['PREDICATE'].apply(clean_predicate)
            self.kg_df['OBJECT_NAME'] = self.kg_df['OBJECT_NAME'].fillna(0.)
            self.kg_df['OBJECT_NAME'] = self.kg_df['OBJECT_NAME'].astype(str)
            self.kg_df['SUBJECT_NAME'] = self.kg_df['SUBJECT_NAME'].astype(str)
            self.kg_df['sent_str'] = self.kg_df.apply(lambda x: x['SUBJECT_NAME'] + ' ' + x['predicate_clean'] + ' ' + x['OBJECT_NAME'], axis = 1)
            
            if rag_type == 'bm25':                
                self.index = BM25Okapi([word_tokenize(i) for i in self.kg_df['sent_str'].str.lower().values])
            elif rag_type == 'knn':
                kg_embs = np.load(kg_emb_path)
                kg_embs = kg_embs[idxs, :]              

                self.index = faiss.IndexFlatIP(kg_embs.shape[1])
                self.index.add(kg_embs)
            else:
                raise NotImplementedError
            
        else:
            self.kg_df = None
            self.kg_emb = None

    def mean_pooling(self,model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def process_batch(self, sentences_batch): # for retrieval
        encoded_input = self.tokenizer(sentences_batch, padding=True, truncation=True, return_tensors='pt')
        for i in encoded_input:
            encoded_input[i] = encoded_input[i].to('cuda')    
        
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)    
        return sentence_embeddings.detach().cpu().numpy()

    def tuples_to_str(self, tuples):
        return '\n'.join(['(' + str(i[0]) + ', ' + str(i[1]) + ', ' + str(i[2]) + ')' for i in tuples])
    
    def tables_to_str(self, tables):
        inp_tuples = list(tables['Input'][['t', 'str_label', 'Amount']].assign(t = tables['Input']['t'].round(2)).sort_values('t').itertuples(index=False, name=None))
        inp_st = self.tuples_to_str(inp_tuples)
        
        lab_tuples = list(tables['Lab'][['t', 'str_label', 'Value']].assign(t = tables['Lab']['t'].round(2)).sort_values('t').itertuples(index=False, name=None))
        lab_st = self.tuples_to_str(lab_tuples)

        if self.include_vitals:
            vit_tuples = list(tables['Vital'][['t', 'str_label', 'Value']].assign(t = tables['Vital']['t'].round(2)).sort_values('t').itertuples(index=False, name=None))
            vit_st = self.tuples_to_str(vit_tuples)
            st = 'Input:\n' + inp_st + '\n\nLab:\n' + lab_st + '\n\nVital:\n'+vit_st + '\n\n'
        else:
            st = 'Input:\n' + inp_st + '\n\nLab:\n' + lab_st

        return st
    
    def to_string_rep(self, tables, tokenizer, token_limit = 1500, claim = None, indices = {}):
        '''
        tables: dictionary of pd.DataFrames
        tokenizer: makes sure that everything fits within token_limit
        claim: if 'rag', must be non-empty, and will be used for BM25 matching
        indices: can pass in a dictionary mapping table name to BM25Okapi index if previously computed. Otherwise will compute from scratch.
        '''
        num_tokens = None
        adm_diag = '- The patient had the following admission diagnoses: ' + ', '.join(tables['Admission']['str_label'].values)
        tables = copy.deepcopy(tables)
        
        if self.select_rows == 'random':
            while num_tokens is None or num_tokens > token_limit:
                if num_tokens is not None:
                    for i in ['Input', 'Lab', 'Vital']:
                        tables[i] = tables[i].sample(frac = max(token_limit/num_tokens - 0.1, 0.05), replace = False, random_state = 42)
                
                st = self.tables_to_str(tables)
                num_tokens = len(tokenizer(st)['input_ids'])
        
            return adm_diag, st, tables, None
        elif self.select_rows == 'rag':
            assert claim is not None
            if self.rag_type == 'bm25':
                for i in ['Input', 'Lab', 'Vital']: # construct index if not provided
                    if i not in indices:
                        indices[i] = BM25Okapi([word_tokenize(i) for i in tables[i]['str_label'].str.lower().values])                
                tok_claim = word_tokenize(claim)
                for i in ['Input', 'Lab', 'Vital']:
                    scores = indices[i].get_scores(tok_claim)
                    tables[i]['score'] = scores
                    # this will surely exceed the token limit, but we will subset later
                    tables[i] = tables[i].sort_values(by = 'score', ascending = False).iloc[:int(token_limit/15)] 
            elif self.rag_type == 'knn':
                for i in ['Input', 'Lab', 'Vital']: # construct index if not provided
                    if i not in indices:
                        batch_size = 1024
                        embs = []
                        sentences = list(tables[i]['str_label'].values)
                        for j in range(0, len(sentences), batch_size):
                            batch = sentences[j:j+batch_size]
                            embs.append(self.process_batch(batch))     
                        embs = np.concatenate(embs, axis=0)
                        indices[i] = faiss.IndexFlatIP(embs.shape[1])
                        indices[i].add(embs)
                claim_emb = self.process_batch([claim])
                for i in ['Input', 'Lab', 'Vital']:
                    D, I = indices[i].search(claim_emb, len(tables[i]))
                    D = D[0]
                    I = I[0]
                    tables[i]['score'] = D # similarity
                    tables[i] = tables[i].sort_values(by = 'score', ascending = False).iloc[:int(token_limit/15)] 

            if self.rag_with_kg: # take top 10 relevant sentences from KG
                if self.rag_type == 'bm25':
                    kg_scores = self.index.get_scores(tok_claim)
                    kg_str = '\n'.join(list(self.kg_df.iloc[(-np.array(kg_scores)).argsort()[:10]]['sent_str'].values))
                elif self.rag_type == 'knn':
                    D, I = self.index.search(claim_emb, 10)
                    kg_str = '\n'.join(list(self.kg_df.iloc[I[0]]['sent_str'].values))
            else:
                kg_str = ''
                
            while num_tokens is None or num_tokens > token_limit:
                if num_tokens is not None:
                    for i in ['Input', 'Lab', 'Vital']:
                        tables[i] = tables[i].iloc[:int(len(tables[i]) * max(token_limit/num_tokens - 0.1, 0.05))]

                st = self.tables_to_str(tables)
                num_tokens = len(tokenizer(st)['input_ids'])

            return adm_diag, st, tables, indices, kg_str
                    
    def get_prompt(self, CLAIM, t_C, tables, tokenizer, token_limit = 1500, indices = {}):
        adm_diag, table_st, subset_tables, indices, kg_str = self.to_string_rep(tables, tokenizer, token_limit=token_limit, claim = CLAIM, indices = indices)
        HEADER = direct_llm_prompts.HEADER
        CLAIM_PRE = f'Claim made at t={t_C}: '
        
        if self.include_examples and len(subset_tables['Input']): # generate an example, using what's left after subsampling
            ex_row = subset_tables['Input'].sample(1, random_state = 42).iloc[0]
            claim = f'pt was given {ex_row["str_label"]} after t={np.round(ex_row["t"], 0) - 1}'
            t_C = np.round(subset_tables['Input']['t'].max(), 2)
            CLAIM_PRE = f'Claim made at t={t_C}: '  
            examples = "Here is an example:\n" + f"<example>\nH: {CLAIM_PRE}{claim}\n\n" + f'A: <stance>T</stance>\n<evidence>({np.round(ex_row["t"], 2)}, {ex_row["str_label"]}, {ex_row["Amount"]})</evidence>\n</example>' 
        else:
            examples = ''

        if not self.rag_with_kg:
            if self.include_examples:
                PROMPT = f"{HEADER}\n{adm_diag}\n\n{table_st}\n\n{examples}\n\n{CLAIM_PRE}{CLAIM}\n"
            else:
                PROMPT = f"{HEADER}\n{adm_diag}\n\n{table_st}\n\n{CLAIM_PRE}{CLAIM}\n"
        else:
            kg_full = 'Potentially relevant medical knowledge: \n' + kg_str 
            if self.include_examples:
                PROMPT = f"{HEADER}\n{adm_diag}\n\n{kg_full}\n\n{table_st}\n\n{examples}\n\n{CLAIM_PRE}{CLAIM}\n"
            else:
                PROMPT = f"{HEADER}\n{adm_diag}\n\n{kg_full}\n\n{table_st}\n\n{CLAIM_PRE}{CLAIM}\n"
        return PROMPT, indices
    
    def parse_stance(self, completion):
        completion = completion.strip()
        if len(completion) == 1 and completion in ['T', 'F', 'N']: # didn't follow <stance></stance> syntax
            return completion
        ans = re.findall(r"<stance>([^<]*)<\/stance>", str(completion))
        if (not len(ans)) or (ans[-1].strip() not in ['T', 'F', 'N']):
            if completion and completion[-1] in ['T', 'F', 'N']:
                return completion[-1]
            return 'N'
        return ans[-1]
    
    def parse_evidence(self, completion):
        ans = re.findall(r"<evidence>([\s\S]*)<\/evidence>", str(completion))
        if (not len(ans)):
            return None
        return ans[0]