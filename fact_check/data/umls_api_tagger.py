from fact_check.constants import umls_cat_mapping, restrict_types_set
import os
import requests
from fact_check.prompting.claude_query import query
from anthropic import HUMAN_PROMPT, AI_PROMPT
import ast
import time
from textwrap import dedent

class UMLS_API_Tagger():
    def __init__(self, n_concepts, restrict_tagger = True, tag_subtypes = False):
        self.n_concepts = n_concepts # number of CUIs to return from UMLS API per Claude tagged entity
        self.restrict_tagger = restrict_tagger # only return concepts which match those in restrict_types_set
        self.tag_subtypes = tag_subtypes # whether to have claude list subtypes of categories (e.g. epinephrine, norepinephrine as subtypes of vasopressor)

    def get_entities(self, st, debug = False):
        if not self.tag_subtypes:
            NER_PROMPT = dedent('''Extract all the biomedical entities in this sentence as a Python list of dictionaries. Only include disorders, drugs, and measurements. 
            Expand acronyms if possible, but only if you are certain. If you are not certain, leave the acronyms in their original format.
            Here are some examples:
            <example>
            H: "pt was administered vasopressors after being given aspirin"
            A: [{'entity': 'vasopressors', 'type': 'drug'}, {'entity': 'aspirin', 'type': 'drug'}]
            </example>

            <example>
            H: "pt had a systolic BP measurement above 140"
            A: [{'entity': 'systolic blood pressure', 'type': 'measurement'}]
            </example>

            <example>
            H: "patient was given warfarin to treat his acute phlebothromboses"
            A: [{'entity': 'warfarin', 'type': 'drug'}, {'entity': 'acute phlebothromboses', 'type': 'disorder'}]
            </example>

            H: ''')
        else:
            NER_PROMPT = dedent('''Extract all the biomedical entities in this sentence as a Python list of dictionaries. Only include disorders, drugs, and measurements. 
            If an entity is a category (e.g. vasopressors), list up to ten subtypes of the category (e.g. epinephrine, norepinephrine).
            Expand acronyms if possible, but only if you are certain. If you are not certain, leave the acronyms in their original format.
            Here are some examples:
            <example>
            H: "pt was administered vasopressors after being given aspirin"
            A: [{'entity': 'vasopressors', 'type': 'drug'}, {'entity': 'epinephrine', 'type': 'drug'}, {'entity': 'norepinephrine', 'type': 'drug'}, {'entity': 'aspirin', 'type': 'drug'}]
            </example>

            <example>
            H: "pt had a systolic BP measurement above 140"
            A: [{'entity': 'systolic blood pressure', 'type': 'measurement'}]
            </example>

            <example>
            H: "patient was given warfarin to treat his acute phlebothromboses"
            A: [{'entity': 'warfarin', 'type': 'drug'}, {'entity': 'coumadin', 'type': 'drug'}, {'entity': 'acute phlebothromboses', 'type': 'disorder'}]
            </example>

            H: ''')
        PROMPT = f'{HUMAN_PROMPT} {NER_PROMPT}"{st}"\n{AI_PROMPT}'
        claude_entities = query(PROMPT)
        try:
            claude_entities = claude_entities[claude_entities.index('['): claude_entities.index(']')+1] # rough way to extract list when Claude decides to add text up front
            claude_entities = ast.literal_eval(claude_entities)
            assert all(['entity' in i and 'type' in i for i in claude_entities])
        except:
            if debug:
                return {'entities': {}}, claude_entities
            else:
                return {'entities': {}}
        
        all_entities = []
        for i in claude_entities:
            all_entities.extend(self.extract(i['entity']))
            if i['type'].lower() == 'measurement': # if an entity is a measurement, we call the API again with 'measurement' appended
                all_entities.extend(self.extract(str(i['entity']) + ' measurement'))

        ret = {'entities': {c: i for c, i in enumerate(all_entities)}}
        if debug:
            return ret, claude_entities
        return ret

    def extract(self, st, retry_timeout = 5):
        api_key = os.environ['UMLS_API_KEY']
        version = '2023AA'
        uri = "https://uts-ws.nlm.nih.gov"
        content_endpoint = "/rest/search/"+version
        full_url = uri+content_endpoint

        query = {'string':st, 'apiKey':api_key, 'partialSearch': 'true',
                'sabs': 'SNOMEDCT_US', 'searchType': 'words'}
        
        for attempt in range(retry_timeout):
            try:
                r = requests.get(full_url,params=query)
                r.raise_for_status()
            except Exception as e:
                print(f'Exception in UMLS API, retrying: {e}')
                time.sleep(1)
                continue
            else:
                break
        else:
            print(f"UMLS API timed out on '{st}'")
        outputs = r.json()
        
        if self.restrict_tagger: # only return concepts which match those in constants.restrict_types_set
            sub_results = []
            for i in outputs['result']['results']:
                if len(set(umls_cat_mapping[i['ui']]).intersection(restrict_types_set)) > 0:
                    i['types'] = umls_cat_mapping[i['ui']]
                    sub_results.append(i)
        else:
            sub_results = outputs['result']['results']

        sub_results = sub_results[:self.n_concepts]
        return [
            {'cui': i['ui'], 'pretty_name': i['name']}
            for c, i in enumerate(sub_results)
        ]