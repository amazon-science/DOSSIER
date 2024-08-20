# DOSSIER: Fact Checking in Electronic Health Records while Preserving Patient Privacy

## Paper

If you use this code in your research, please cite our [paper](https://www.amazon.science/publications/dossier-fact-checking-in-electronic-health-records-while-preserving-patient-privacy):

```
@inproceedings{zhang2024dossier,
 author={Haoran Zhang and Supriya Nagesh and Milind Shyani and Nina Mishra},
 title={DOSSIER: Fact checking in electronic health records while preserving patient privacy},
 year={2024},
 booktitle={Machine Learning for Healthcare Conference},
 organization={PMLR}
}
```

## Dataset

We release a dataset of 4,250 natural language claims regarding the patient-specific EHR records of 100 admissions from the MIMIC-III dataset, which require information from one or more of the `admissions`, `chartevents`, `labevents`, and `inputevents` tables, and 35% of which require external medical knowledge to verify.

As this dataset is derived from MIMIC-III, we are unable to release it publicly. We are currently working on releasing this through PhysioNet under the PhysioNet Credentialed Health Data License. In the meantime, if access to the data is required, please email `haoranz [at] mit.edu` with a copy of your [CITI certificate](https://physionet.org/about/citi-course/) and we will reply with the .csv file.


## To replicate the experiments in the paper:
### Step 0. Python Environment
Run the following commands to create the Conda environment and install required packages:

```
conda env create -f environment.yml
conda activate fact_check
pip install -U git+https://github.com/huggingface/transformers@9ed538f2e67ee10323d96c97284cf83d44f0c507
pip install -U medcat==1.9.0 --no-deps
pip install -U huggingface_hub
```

### Step 1. Download Data

#### MIMIC-III
Obtain access to PhysioNet, and download [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) (v1.4). If the files you downloaded are in `.csv.gz` format, extract them to `.csv`.

#### UMLS
Obtain access to the UMLS, and download and extract the [UMLS Metathesaurus Full Subset](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html) (we use v2023AA). 

#### SemMedDB
Download the `PREDICATION`, `PREDICATION_AUX`, and `GENERIC_CONCEPTS` csv files from [SemMedDB](https://lhncbc.nlm.nih.gov/ii/tools/SemRep_SemMedDB_SKR/SemMedDB_download.html). Place them in a folder named `semmeddb`. Note that we use the 2023 version. 

#### MedCat
Download the `UMLS Small` and `SNOMED International` [MedCat models](https://github.com/CogStack/MedCAT). You will need an UMLS account. 

#### Updating Paths
Place all of the previous folders into a single `DATA_DIR`.  Update the paths found in `fact_check/constants.py`. For example, your `DATA_DIR` should contain the folders `{DATA_DIR}/2023AA`, `{DATA_DIR}/semmeddb`, and `{DATA_DIR}/medcat_mimic_snomed`.

### Step 2. API Keys

Obtain access to Anthropic and UMLS API keys. Add these lines to your `~/.bashrc`, or run them before running any script.

```
export ANTHROPIC_API_KEY="YOUR KEY"
export UMLS_API_KEY="YOUR KEY"
```

### Step 3. SemMedDB Preprocessing

Run `notebooks/process_semmeddb.ipynb` to process SemMedDB (filtering edges by citations, and merging with the SNOMED hierarchy). This will output a `semmeddb_processed_10.csv` file in your `semmeddb` folder.


### Step 4. Running the Pipeline

#### DOSSIER


To evaluate the DOSSIER pipeline, run `eval_pipeline`, e.g. 
```
python -m fact_check.eval_pipeline \
--prompt full \
--llm claude \
--output_dir /output/dir
```
This outputs `res.pkl` in the `output_dir`, which contains the predicted stance for each claim in the cohort.


#### LLM Baselines

To evaluate the baseline LLMs which directly take in EHR records, run `eval_baseline_llm`, e.g.

```
python -m fact_check.eval_baseline_llm \
--llm asclepius \
--rag_type knn \
--rag_with_kg \
--output_dir /output/dir
```

To run MedAlpaca, you will need to clone the [MedAlpaca repo](https://github.com/kbressem/medAlpaca) into the root directory of this repo (i.e. there should be a folder `DOSSIER/medAlpaca`). 

If using `rag_type = knn`, you will have to first run `notebooks/cache_sbert_embs.ipynb`.

The first run of the codebase may be unresponsive for the first 10 minutes as the program caches two additional files in the `fact_check/cache` folder. Subsequent runs should be much faster.


### Step 5. Evaluating Results

To analyze the output of `eval_pipeline` and `eval_baseline_llm`, use `notebooks/agg_results.ipynb`.
