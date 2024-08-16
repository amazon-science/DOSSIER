from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, pipeline, LlamaForCausalLM, LlamaTokenizer
import torch
import os
from fact_check.constants import DATA_DIR
from fact_check.utils import to_device


class LLM():
    def __init__(self, llm_type, max_new_tokens = 512, weights = 7, temperature = 0., multigpu = True):
        self.llm_type = llm_type
        self.weights = weights
        self.max_new_tokens = max_new_tokens
        self.generation_config = {
            'temperature': temperature
        }
        
        if llm_type == 'medalpaca':
            from medAlpaca.medalpaca.inferer import Inferer
            self.model = Inferer(f"medalpaca/medalpaca-{weights}b", 
                                 os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../medAlpaca/medalpaca/prompt_templates/medalpaca.json"),
                                torch_dtype=torch.float16)
            self.tokenizer = self.model.data_handler.tokenizer
            self.req_fields = ['instruction', 'input']
            self.separate_input = True
            self.func = lambda x: self.model(**x, **self.generation_config).removesuffix(r'<s>').removesuffix(r'</s>')
        elif llm_type == 'llama':
            self.model = pipeline("text-generation", model=f"huggyllama/llama-{weights}b", tokenizer=f'huggyllama/llama-{weights}b', 
                                  device = 'cpu')
            self.req_fields = ['input']
            self.separate_input = False
            self.func = lambda x: self.model(x['input'], max_new_tokens = self.max_new_tokens, return_full_text = False)[0]
        elif llm_type in ['vicuna', 'clinicalcamel']:
            if llm_type == 'vicuna':
                self.tokenizer = LlamaTokenizer.from_pretrained(DATA_DIR/"vicuna-13b")
                self.model = LlamaForCausalLM.from_pretrained(DATA_DIR/"vicuna-13b", device_map = 'auto')
            elif llm_type == 'clinicalcamel':
                self.tokenizer = LlamaTokenizer.from_pretrained(DATA_DIR/"clinicalcamel", )
                self.model = LlamaForCausalLM.from_pretrained(DATA_DIR/"clinicalcamel", device_map = 'auto', torch_dtype=torch.float16)
            self.req_fields = ['input']
            self.separate_input = False
            self.func = lambda x: self.tokenizer.decode(self.model.generate(
                 **to_device(self.tokenizer('USER: ' + x['input'] + "\nASSISTANT:", return_tensors='pt'), 'cuda'),
                max_new_tokens = self.max_new_tokens,
                **self.generation_config
            )[0, :], skip_special_tokens=True)[len(x['input'] + 'USER: ' + "\nASSISTANT:"):]
        elif llm_type == 'codellama':
            from transformers import CodeLlamaTokenizer
            self.tokenizer = CodeLlamaTokenizer.from_pretrained("codellama/CodeLlama-13b-Instruct-hf")
            self.model = LlamaForCausalLM.from_pretrained("codellama/CodeLlama-13b-Instruct-hf", device_map = 'auto', torch_dtype=torch.float16)
            self.req_fields = ['input']
            self.separate_input = False
            def decode(x):
                st = x['input']
                a = to_device(self.tokenizer(st, return_tensors='pt', add_special_tokens=False), 'cuda')
                return self.tokenizer.decode(self.model.generate(**a, max_new_tokens=self.max_new_tokens,
                    )[0, a['input_ids'].shape[1]:])
            self.func = decode
        elif llm_type in ['llama2', 'asclepius', 'clinicalcamel_new']:
            if llm_type == 'llama2':
                self.tokenizer = AutoTokenizer.from_pretrained("togethercomputer/LLaMA-2-7B-32K")
                self.model = AutoModelForCausalLM.from_pretrained("togethercomputer/LLaMA-2-7B-32K", device_map = 'auto', torch_dtype=torch.float16)
            elif llm_type == 'clinicalcamel_new':
                self.tokenizer = AutoTokenizer.from_pretrained("wanglab/ClinicalCamel-70B")
                self.model = AutoModelForCausalLM.from_pretrained("wanglab/ClinicalCamel-70B", device_map = 'auto', load_in_8bit=True)
            elif llm_type == 'asclepius':
                self.tokenizer = AutoTokenizer.from_pretrained("starmpcc/Asclepius-Llama2-13B")
                self.model = AutoModelForCausalLM.from_pretrained("starmpcc/Asclepius-Llama2-13B", device_map = 'auto', torch_dtype=torch.float16)
            self.req_fields = ['input']
            self.separate_input = False
            def decode(x):
                st = x['input']
                a = to_device(self.tokenizer(st, return_tensors='pt', add_special_tokens=True), 'cuda')
                return self.tokenizer.decode(self.model.generate(**a, max_new_tokens=self.max_new_tokens,
                    )[0, a['input_ids'].shape[1]:])
            self.func = decode
        else:
            raise NotImplementedError
    
    def __call__(self, **x):
        for i in self.req_fields:
            assert i in x
        with torch.no_grad():
            return self.func(x)