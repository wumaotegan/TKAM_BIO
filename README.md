 # TKAM: Toxicity Knowledge-Augmented Modeling

This repository provides the workflow code, data organization, model-training pipeline, and prediction interfaces associated with the manuscript by Yang et al., **_Automated Knowledge-guided Retrieval of In Vitro Bioactivity Enables In Vivo Toxicity Extrapolation via Curriculum Multi-endpoint Learning_**.

Toxicity knowledge-augmented modeling (TKAM) addresses the scarcity of in vivo toxicity labels for per- and polyfluoroalkyl substances (PFAS) by integrating PubChem BioAssay data, adverse outcome pathway (AOP) knowledge, semantic and knowledge-graph retrieval, ensemble large language model (LLM) reasoning, and curriculum multi-task learning. The current implementation supports hepatotoxicity (HepTox) and reproductive toxicity (ReproTox) classification.

## Overview

TKAM automatically identifies in vitro bioassays that are mechanistically relevant to a target in vivo endpoint and uses these assays as upstream learning tasks. Bioactivity knowledge is then progressively transferred to in vivo toxicity classification through a staged curriculum.

The workflow contains three main components:

1. **Automated, knowledge-guided bioassay retrieval**
   - Encode assay descriptions with the Qwen3 `text-embedding-v4` model.
   - Extract biological entities and relationships with Qwen3-Max and construct a Neo4j knowledge graph (KG).
   - Use AOP-Wiki abstracts for hepatotoxicity and reproductive toxicity as mechanistic queries.
   - Combine semantic similarity with KG-based connectivity to retrieve directly and indirectly related assays.
   - Apply model-based re-ranking and ensemble LLM reasoning to retain high-confidence, endpoint-relevant bioassays.

2. **Bioassay curation and PFAS-specific molecular representation learning**
   - Clean and standardize PubChem BioAssay activity records.
   - Standardize SMILES, identify C–F-containing chemicals, and annotate PFAS.
   - Remove redundant tasks using activity concordance and assay-title similarity.
   - Continue masked language model pre-training of ChemBERTa-100M-MLM on 6,892,694 PFAS SMILES to obtain a PFAS-adapted molecular encoder, referred to as PFAS-ChemBERTa.

3. **Curriculum multi-task learning for in vivo toxicity classification**
   - **Stage 0:** uncertainty-weighted multi-task pre-training on endpoint-relevant in vitro bioassays.
   - Apply dynamic task pruning to reduce interference from underperforming bioassay tasks.
   - **Stage 1:** adapt the shared encoder to in vivo toxicity data for C–F-containing chemicals.
   - **Stage 2:** fine-tune a PFAS-specific in vivo task head using PFAS toxicity labels.

## Repository Structure

```text
.
├── code/
│   ├── bioassay_retrieval/
│   │   ├── generate_knowledge_graph_qwen3_base.py
│   │   ├── get_info_pub.py
│   │   ├── llm.py
│   │   ├── workflow_hepatotoxicity_bioassay_retrieval.ipynb
│   │   ├── workflow_reproductive_toxicity_bioassay_retrieval.ipynb
│   │   ├── bioassay_process_heptox.ipynb
│   │   └── bioassay_process_reprotox.ipynb
│   ├── chembert_pfas/
│   │   ├── PFAS_SMILES_standardization.ipynb
│   │   └── ChemBERTa-100M_full_finetune.ipynb
│   ├── multi_tasks_training/
│   │   ├── multitask-pretraining-with-pruning-FT_heptox.ipynb
│   │   ├── invivo_train.py
│   │   └── invivo_train_curriculum_save_model.py
│   └── predict/
│       ├── predict_heptox.ipynb
│       ├── predict_reprotox.ipynb
│       └── predict_models
│           ├── TKAM_heptox_models/
│           └── TKAM_reprotox_models/
├── data/
│   ├── neo4j.dump
│   ├── aid_assay_name_map.json
│   ├── pfas_6892694.txt
│   ├── pfas_set.pickle
│   ├── bioassays_description/
│   └── invivodata/
│  
├── requirements_bioassay_retrieval.txt
└── requirements_model.txt

```
The `data/` and `predict_models/` directories are hosted on Zenodo:https://doi.org/10.5281/zenodo.20815599. 

Please download and extract the Zenodo archive, and then place the extracted directories as follows:

TKAM_BIO/
├── data/
└── code/
    └── predict/
        └── predict_models/
            ├── TKAM_heptox_models/
            └── TKAM_reprotox_models/

Specifically:

Place the extracted data/ directory in the repository root.
Place the extracted predict_models/ directory under code/predict/.

## Environments

Two separate environments are recommended.

### Bioassay Retrieval and Processing Environment

```bash
conda create -n bioassay_retrieval python=3.9.19
conda activate bioassay_retrieval
pip install -r requirements_bioassay_retrieval.txt
```

This environment is used for PubChem BioAssay retrieval, semantic embedding, Neo4j KG construction, AOP querying, re-ranking, ensemble LLM reasoning, and bioassay processing.

### Model Training and Prediction Environment

```bash
conda create -n model python=3.10.19
conda activate model
pip install -r requirements_model.txt
```

GPU acceleration is recommended for model training. The computational environment used for the manuscript was based on CUDA 12.1 and PyTorch 2.5.1.

## 1. Build or Load the PubChem BioAssay Knowledgebase

The semantic embeddings of PubChem BioAssay descriptions and the corresponding LLM-derived KG are stored in:

```text
data/neo4j.dump
```

The dump contains:

- semantic embeddings of PubChem BioAssay description chunks;
- a Neo4j KG assembled from biological entities and relationships extracted from assay descriptions by Qwen3-Max.

### Load the Existing Neo4j Dump

To use the pre-built knowledgebase:

1. Install Neo4j.
2. Import `data/neo4j.dump` into a Neo4j database.
3. Start the Neo4j service.
4. Use Neo4j Browser or a Bolt client to inspect and query the bioassay knowledgebase.

### Rebuild the Embeddings and Knowledge Graph

Uncomment the if \_\_name__ == "\_\_main__": code block in the script and run it.

```bash
cd code/bioassay_retrieval
python generate_knowledge_graph_qwen3_base.py
```

Configure the LLM and embedding model before execution, for example:

```python
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
model = "qwen3-max"
EMBEDDING_MODEL = "text-embedding-v4"
```

`generate_knowledge_graph_qwen3_base.py` was adapted from the [neo4j-labs/llm-graph-builder](https://github.com/neo4j-labs/llm-graph-builder) project to construct a KG and embedding database from PubChem BioAssay descriptions. The script supports both API-hosted embedding models and local Hugging Face embedding models. Other compatible LLMs may also be configured.

## 2. Retrieve Endpoint-Relevant Bioassays

Retrieval notebooks:

```text
code/bioassay_retrieval/workflow_hepatotoxicity_bioassay_retrieval.ipynb
code/bioassay_retrieval/workflow_reproductive_toxicity_bioassay_retrieval.ipynb
```

The retrieval workflow performs the following steps:

1. Search AOP-Wiki for AOPs associated with hepatotoxicity and reproductive toxicity.
2. Download and organize the corresponding AOP titles and abstracts.
3. Convert AOP abstracts into semantic embeddings and query semantically similar PubChem BioAssays in Neo4j.
4. Expand the candidate set using KG-based connectivity through shared biological entities.
5. Apply a two-step re-ranking procedure to prioritize AOP–bioassay pairs with stronger mechanistic relevance.
6. Ask multiple LLMs whether active substances in each bioassay could plausibly contribute to the associated in vivo toxicity endpoint.
7. Retain high-confidence bioassays by ensemble consensus.
8. Download the activity records for the final assay set.

Input files:

```text
data/aid_assay_name_map.json
data/bioassays_description/aid_test_100/
```

Configure Neo4j credentials and model-service API keys for the local environment:

```python
uri = "bolt://localhost:7687"
userName = "neo4j"
password = "your_neo4j_password"

openrouter_base_url = "https://openrouter.ai/api/v1"
openrouter_api_key = "your_openrouter_api_key"
dashscope.api_key = "your_qwen_api_key"
EMBEDDING_MODEL = "text-embedding-v4"
```

Main output directories:

```text
code/bioassay_retrieval/hepatotoxicity_data/
code/bioassay_retrieval/reproductive toxicity_data/
data/bioassay_raw/hepatotoxicity/
data/bioassay_raw/reproductive toxicity/
```

Representative output files:

```text
*_aop_similar_assay_qwen3.csv
*_aop_similar_assay_rerank_qwen3.csv
*_aop_similar_assay_LLMs_reasoning.csv
aop_similar_assay_rerank_llm_infer_T.csv
```

## 3. Curate and Deduplicate Bioassay Data

Processing notebooks:

```text
code/bioassay_retrieval/bioassay_process_heptox.ipynb
code/bioassay_retrieval/bioassay_process_reprotox.ipynb
```

These notebooks:

1. Read raw activity records for the endpoint-relevant PubChem BioAssays.
2. Standardize compound SMILES by removing salts, retaining the largest molecular fragment, neutralizing formal charges, and generating canonical isomeric SMILES.
3. Identify chemicals containing C–F bonds.
4. Annotate PFAS using `data/pfas_set.pickle`.
5. Summarize positive and negative records for C–F-containing chemicals and PFAS in each assay.
6. Retain bioassays containing at least 25 positive and 25 negative C–F-containing chemicals.
7. Identify potentially redundant assay tasks using Cohen's kappa activity concordance and assay-title similarity.

Input files:

```text
data/pfas_set.pickle
data/bioassay_raw/hepatotoxicity/
data/bioassay_raw/reproductive toxicity/
```

Per-assay processed files are written to:

```text
data/processed/bioassay/hepatotoxicity/assay_processed/*_processed.csv
data/processed/bioassay/reproductive toxicity/assay_processed/*_processed.csv
```

Summary files are written to:

```text
code/bioassay_retrieval/hepatotoxicity_assay_desc.csv
code/bioassay_retrieval/hepatotoxicity_assay_25.csv
code/bioassay_retrieval/hepatotoxicity_assay_similarity_results.csv
code/bioassay_retrieval/heptox_assay_similarity_analysis.csv
code/bioassay_retrieval/hepatotoxicity_assay_desc_deduplicated_with_kappa_and_text.csv
code/bioassay_retrieval/hepatotoxicity_assay_desc_deduplicated_with_kappa_and_text_25.csv

code/bioassay_retrieval/reproductive toxicity_assay_desc.csv
code/bioassay_retrieval/reproductive toxicity_assay_25.csv
code/bioassay_retrieval/reproductive toxicity_assay_similarity_results.csv
code/bioassay_retrieval/reproductive toxicity_assay_similarity_analysis.csv
code/bioassay_retrieval/reproductive toxicity_assay_desc_deduplicated_with_kappa_and_text.csv
code/bioassay_retrieval/reproductive toxicity_assay_desc_deduplicated_with_kappa_and_text_25.csv
```

The deduplication procedure uses two complementary criteria:

- **Activity concordance:** Cohen's kappa is calculated for assay pairs sharing a sufficient number of standardized chemicals. Pairs with highly concordant binary activity profiles are flagged as potentially redundant.
- **Title similarity:** cleaned assay titles are represented with term frequency–inverse document frequency (TF–IDF) vectors and compared using cosine similarity.

For each redundant pair, assays with `summary` in the title are preferentially retained because they generally represent more comprehensive aggregated datasets. Otherwise, the assay with the greater redundancy degree is removed. When redundancy degrees are equal, the lower AID is removed, following the rule implemented in the processing notebooks.

## 4. Standardize PFAS SMILES

Notebook:

```text
code/chembert_pfas/PFAS_SMILES_standardization.ipynb
```

Input:

```text
data/pfas_6892694.txt
```

The notebook performs:

- salt removal;
- retention of the largest molecular fragment for multi-fragment SMILES;
- neutralization of formal charges;
- conversion to canonical isomeric SMILES.

Outputs:

```text
pfas_smiles_processed.csv
pfas_smiles_unique.txt
```

## 5. Adapt ChemBERTa to the PFAS Chemical Space

Notebook:

```text
code/chembert_pfas/ChemBERTa-100M_full_finetune.ipynb
```

This workflow performs continued full-model pre-training of `DeepChem/ChemBERTa-100M-MLM` on standardized PFAS SMILES using masked language modeling. The resulting PFAS-ChemBERTa encoder is used to derive PFAS-specific molecular representations for downstream multi-task learning.

Input:

```text
pfas_smiles_unique.txt
```

Example training configuration:

```python
training_args = TrainingArguments(
    output_dir="./ChemBERTa-Full_FT-PFAS",
    per_device_train_batch_size=128,
    per_device_eval_batch_size=256,
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    evaluation_strategy="steps",
    eval_steps=1000,
    save_steps=1000,
    logging_steps=1000,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    bf16=True
)
```

For GPUs without bfloat16 support, set:

```python
bf16 = False
```

The resulting checkpoint is used as the shared molecular encoder for bioassay multi-task pre-training and subsequent in vivo toxicity transfer.

## 6. Perform Uncertainty-Weighted Multi-Task Pre-training on Bioassays

Notebook:

```text
code/multi_tasks_training/multitask-pretraining-with-pruning-FT_heptox.ipynb
```

The workflow constructs a multi-task model with PFAS-ChemBERTa as the shared encoder and an independent binary classification head for each retained bioassay task.

Main steps:

1. Load the filtered and deduplicated bioassay list.
2. Load standardized activity data for each bioassay.
3. Retain C–F-containing chemicals.
4. Stratify each task by class label and create an 8:2 train/test split.
5. Apply resampling only to the training set and leave the test set unchanged for unbiased evaluation.
6. Balance positive and negative samples within each task.
7. Apply temperature-based resampling to balance training sizes across bioassay tasks.
8. Optimize an uncertainty-weighted binary cross-entropy objective with learnable task-uncertainty parameters.
9. Dynamically prune tasks with validation AUC below the configured threshold after the warm-up period.

Example inputs:

```text
hepatotoxicity_assay_desc_deduplicated_with_kappa_and_text_25.csv
data/processed/bioassay/hepatotoxicity/assay_processed/
best_full_finetuned_model/
```

For each bioassay task \(k\), the effective sampling size $|S_k| $is defined as:

$$
|S_k| =
\begin{cases}
M_k, & \text{if } M_k < \lambda \\
\left\lfloor \lambda^{1-\tau} M_k^{\tau} \right\rfloor, & \text{if } M_k \ge \lambda
\end{cases},
\quad k \in \mathcal{T}_{\mathrm{Bioassay}}
$$

The default hyperparameters are \(\lambda = 256\) and \(\tau = 0.5\).

Default model and pruning parameters:

```python
pruning_start_epoch = 3
pruning_threshold = 0.55
p_batch_size = 32
p_dropout = 0.4
p_hidden_dim = 512
p_lr_stage1 = 0.0004
p_lr_stage2 = 1e-5
```

Outputs:

```text
hepatotoxicity_assay_desc_deduplicated_with_kappa_and_text_25_MTL-with-Pruning-result.xlsx
chemberta-multitask-pfas-with-pruning
```

## 7. Fine-tune the Model through Curriculum Multi-Task Learning

Script:

```text
code/multi_tasks_training/invivo_train.py
```

The script extends the bioassay-pretrained multi-task model with two in vivo task heads:

- `Invivo_CF`: in vivo toxicity classification for C–F-containing chemicals outside the PFAS target subset;
- `Invivo_PFAS`: PFAS-specific in vivo toxicity classification.

The implemented training strategies include:

- **No Replay:** fine-tune using only in vivo toxicity data.
- **Data Replay:** include samples from retained upstream bioassay tasks during in vivo adaptation.
- **Curriculum Learning:** train `Invivo_CF` before `Invivo_PFAS`.
- **Curriculum Learning + Bioassay Data Replay:** perform staged transfer and replay bioassay samples during PFAS-specific adaptation.

In the curriculum configuration, the workflow corresponds to:

- **Stage 0 — Bioassay warm-up:** pre-train the shared encoder and task-specific heads on endpoint-relevant in vitro bioassays.
- **Stage 1 — C–F in vivo adaptation:** freeze the pretrained bioassay heads, add `Invivo_CF`, and adapt the shared encoder to organism-level toxicity data for C–F-containing chemicals.
- **Stage 2 — PFAS-specific adaptation:** add `Invivo_PFAS`, optimize on PFAS in vivo labels, and replay bioassay samples to preserve upstream bioactivity knowledge.

The Matthews correlation coefficient (MCC) is calculated using an optimized threshold selected to maximize the MCC. The same evaluation procedure is applied consistently across all scripts.

Example hepatotoxicity inputs:

```text
data/invivodata/hep_all_unique_standardized.csv
code/multi_tasks_training/best_model_with_pruning_heptox/
data/processed/bioassay/hepatotoxicity/train/
data/processed/bioassay/hepatotoxicity/test/
```

Default training parameters:

```python
train_batch_size = 32
encoder_lr = 1e-5
head_lr = 1e-5
accumulation_steps = 4
epochs = 20
replay_ratios_to_test = [0.2, 0.5, 0.8]

```

Run from the repository root:

```bash
python code/multi_tasks_training/invivo_train.py
```

To run and save only the best curriculum-learning configuration, use:

```text
code/multi_tasks_training/invivo_train_curriculum_save_model.py
```

This script implements curriculum learning with bioassay data replay.

## 8. Predict PFAS In Vivo Toxicity with TKAM

Prediction notebooks and model directories:

```text
code/predict/predict_heptox.ipynb
code/predict/TKAM_heptox_models/

code/predict/predict_reprotox.ipynb
code/predict/TKAM_reprotox_models/
```



Usage:

1. Open `predict_heptox.ipynb` or `predict_reprotox.ipynb`.
2. Set `SMILES` in the final code cell.
3. Keep `TASK_NAME = "Invivo_PFAS"` for PFAS in vivo toxicity inference.
4. Run all notebook cells in order.

Example:

```python
SMILES = "your_smiles"
TASK_NAME = "Invivo_PFAS"
SMILES_STANDARDIZED = standardize_smiles(SMILES)
prediction = predict_five_model_ensemble(SMILES_STANDARDIZED, TASK_NAME)

print(f"{TASK_NAME}_pred: {prediction['invivo_mean']:.6f}")
display(prediction["bioassay_predictions"])
```

Outputs include:

- `Invivo_PFAS_pred`: the mean PFAS in vivo toxicity probability across five model checkpoints;
- `bioassay_predictions`: mean predicted probabilities for the retained bioassay tasks across the five checkpoints, sorted by `mean_probability` in descending order.

## Recommended Execution Order

1. Create the bioassay retrieval and processing environment.
2. Import `data/neo4j.dump` into Neo4j, or rebuild the semantic embeddings and KG.
3. Run the hepatotoxicity and reproductive-toxicity bioassay retrieval notebooks.
4. Download, curate, and deduplicate the PubChem BioAssay activity data.
5. Standardize the PFAS SMILES corpus.
6. Continue masked language model pre-training of ChemBERTa on PFAS SMILES.
7. Perform uncertainty-weighted multi-task pre-training on endpoint-relevant bioassays.
8. Run curriculum multi-task learning for in vivo toxicity adaptation.
9. Use the prediction notebooks for HepTox or ReproTox inference.

## Notes

- API keys are not included in this repository. Configure DashScope, OpenRouter, or another compatible model service before running LLM, embedding, or re-ranking workflows.
- Ensure that the Neo4j database is running and accessible before bioassay retrieval.
## Citation



