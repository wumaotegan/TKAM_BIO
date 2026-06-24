import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer, AutoConfig
from sklearn.metrics import roc_auc_score,matthews_corrcoef,roc_curve
import numpy as np
import pandas as pd
import logging
import random
import optuna
import functools
from collections import defaultdict
from typing import Dict, List, Optional
from sklearn.model_selection import StratifiedShuffleSplit
from collections import defaultdict
from sklearn.utils import resample
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import cross_val_score,StratifiedKFold
import json
import copy
from transformers import get_linear_schedule_with_warmup
import warnings
import gc
import os
from transformers import logging  as transformers_logging
from pathlib import Path

transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

optuna.logging.set_verbosity(optuna.logging.INFO)
def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
toxicity="hepatotoxicity"


PROJECT_ROOT =Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT/"data"


tokenizer = AutoTokenizer.from_pretrained(PROJECT_ROOT/"code/chembert_pfas/DeepChem/ChemBERTa-100M-MLM" )

def split_invivo_datasets(df, Mark,test_size=0.2,random_seed=42,class_col='class'):
    df = df.dropna(subset=["standardized_smiles", class_col, "pfas_target"])
    total_count = len(df)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    train_idx, test_idx = next(splitter.split(df, df[class_col]))
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    logger.info(f"Task [{Mark}]: Total={total_count} -> Train={len(train_df)}, Test={len(test_df)}") 
    return train_df, test_df

def load_mixed_finetuning_dataset(
    target_configs,       
    replay_configs,       
    replay_ratio=0.2,    
    balance_target=True,    
    random_seed=42
):

    all_smiles = []
    all_labels_list = []
    
    task_names = set()
    rng = np.random.default_rng(random_seed)
    target_count = 0
    
    for config in target_configs:
        path = config['path']
        task = config['task_name']
        s_col = config['smiles_col']
        l_col = config['label_col']
        task_names.add(task)
        if type(path) is  str:
            df = pd.read_csv(path)
            df = df.dropna(subset=[s_col, l_col])
        else:
            df = path
        
        if balance_target:
            df_pos = df[df[l_col] == 1]
            df_neg = df[df[l_col] == 0]
            
            n_pos = len(df_pos)
            n_neg = len(df_neg)
            
            if n_pos > 0 and n_neg > 0:
                target_n = max(n_pos, n_neg)
                df_pos_bal = resample(df_pos, replace=True, n_samples=target_n, random_state=random_seed)
                df_neg_bal = resample(df_neg, replace=True, n_samples=target_n, random_state=random_seed)
                df = pd.concat([df_pos_bal, df_neg_bal])
                logger.info(f"  - Task [{task}]: Balanced total={len(df)} (Pos={target_n}, Neg={target_n})")
        
        for _, row in df.iterrows():
            smi = row[s_col]
            label = int(row[l_col])
            
            all_smiles.append(smi)
            all_labels_list.append({task: label}) 
            
        target_count += len(df) 

    n_replay_needed = int(target_count * (replay_ratio / (1.0 - replay_ratio)))
    n_replay_needed = max(n_replay_needed, 64)
    
    if replay_ratio == 0.0:
        task_names_list = sorted(list(task_names))
        logger.info(f"Fine-tuning dataset ready: {len(all_smiles)} samples ({len(task_names_list)} tasks)")
        return all_smiles, all_labels_list, task_names_list
    
    logger.info(f">>> Step 2: Replay budget (Target: {target_count}, Ratio: {replay_ratio}) -> Need: {n_replay_needed}")
    logger.info(">>> Step 3: Sampling balanced replay tasks...")
    
    valid_replay_configs = [c for c in replay_configs if os.path.exists(c['path'])]
    if not valid_replay_configs:
        logger.warning("No valid replay data paths were found.")
        return all_smiles, all_labels_list, sorted(list(task_names))

    per_task_quota = n_replay_needed // len(valid_replay_configs)
    selected_replay = []

    for config in valid_replay_configs:
        path, task, s_col, l_col = config['path'], config['task_name'], config['smiles_col'], config['label_col']
        task_names.add(task)
        
        df_task = pd.read_csv(path)
        if "contains_CF" in df_task.columns:
            df_task = df_task[df_task["contains_CF"] == 1]
        df_task = df_task.dropna(subset=[s_col, l_col])
        
        if len(df_task) == 0: continue

        df_pos = df_task[df_task[l_col] == 1]
        df_neg = df_task[df_task[l_col] == 0]
        
        if len(df_pos) > 0 and len(df_neg) > 0:
            half_q = per_task_quota // 2
            df_pos_bal = resample(df_pos, replace=True, n_samples=half_q, random_state=random_seed)
            df_neg_bal = resample(df_neg, replace=True, n_samples=half_q, random_state=random_seed)
            df_bal = pd.concat([df_pos_bal, df_neg_bal])
        else:
            df_bal = resample(df_task, replace=True, n_samples=per_task_quota, random_state=random_seed)

        for _, row in df_bal.iterrows():
            selected_replay.append((row[s_col], task, int(row[l_col])))
            
    rng.shuffle(selected_replay)
    for smi, task, label in selected_replay:
        all_smiles.append(smi)
        all_labels_list.append({task: label})

    # -----------------------------------------------------------
    task_names_list = sorted(list(task_names))
    logger.info(f"Dataset ready: {len(all_smiles)} samples (Target: {target_count}, Replay: {len(selected_replay)})")
    
    return all_smiles, all_labels_list, task_names_list


class MoleculeDataset(torch.utils.data.Dataset):
    def __init__(self, smiles_list, labels_list, tokenizer, max_len=128):
        self.labels_list = labels_list
        self.encodings = tokenizer(smiles_list, truncation=True, padding=True, max_length=max_len)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels_dict'] = self.labels_list[idx]
        return item

    def __len__(self):
        return len(self.labels_list)

class MultitaskCollator:
    def __init__(self, task_names, ignore_index=-100): 
        self.task_names = task_names
        self.task_to_idx = {name: i for i, name in enumerate(task_names)}
        self.ignore_index = ignore_index

    def __call__(self, batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        batch_size = len(batch)
        num_tasks = len(self.task_names)
        labels_tensor = torch.full((batch_size, num_tasks), self.ignore_index, dtype=torch.float)
        for i, item in enumerate(batch):
            sample_labels = item['labels_dict']
            for task_name, label_val in sample_labels.items():
                if task_name in self.task_to_idx:
                    col_idx = self.task_to_idx[task_name]
                    labels_tensor[i, col_idx] = label_val
        
        return input_ids, attention_mask, labels_tensor


def get_train_dataloader(df_invivo_pfas_train,df_invivo_cf_train,replay_ratio,train_batch_size,tokenizer,seed,task_names,balance_target=True):
    
    replay_train_dir=  DATA_DIR / "processed" / "bioassay" / toxicity / "train"
    replay_files = [f for f in os.listdir(replay_train_dir) if f.endswith(".csv")]
    replay_configs = []

    for f in replay_files:
        aid = f.split('_')[1].replace('.csv', '')
        replay_configs.append({
            "path": os.path.join(replay_train_dir, f),
            "task_name": f"AID:{aid}",
            "smiles_col": "standardized_smiles",
            "label_col": "class"
        })
    target_train_configs = [
        {
            "path": df_invivo_pfas_train , 
            "task_name": "Invivo_PFAS", 
            "smiles_col": "standardized_smiles", 
            "label_col": "hepatotoxicity"
        },
        {
            "path": df_invivo_cf_train, 
            "task_name": "Invivo_CF", 
            "smiles_col": "standardized_smiles", 
            "label_col": "hepatotoxicity"
        }
    ]
    train_smiles, train_labels, _ = load_mixed_finetuning_dataset(
        target_configs=target_train_configs,
        replay_configs=replay_configs,
        replay_ratio=replay_ratio,      
        balance_target=balance_target,  
        random_seed=seed
    )
    
    
    train_dataset = MoleculeDataset(train_smiles, train_labels, tokenizer, max_len=128)
    train_collator = MultitaskCollator(task_names=task_names)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,     
        shuffle=True,     
        collate_fn=train_collator, 
        num_workers=8,    
        pin_memory=True     
    )
    return train_loader,replay_configs


def get_test_loader(df_invivo_pfas_test,df_invivo_cf_test,replay_configs,all_task_names,batch_size,tokenizer):
    replay_configs_test = copy.deepcopy(replay_configs)
    for config in replay_configs_test:
        config['path'] = config['path'].replace('train', 'test')
    pfas_test_config = [{
        "path": df_invivo_pfas_test,
        "task_name": "Invivo_PFAS", 
        "smiles_col": "standardized_smiles",
        "label_col": "hepatotoxicity"     
    }]

    nfig = copy.deepcopy(pfas_test_config)
    nfig[0]['path'] = df_invivo_cf_test
    nfig[0]['task_name'] = "Invivo_CF"
    cf_test_config = nfig

    current_test_configs = replay_configs_test + cf_test_config + pfas_test_config
    smiles_list = []
    labels_list = []
    print(f"Building the test DataLoader from {len(current_test_configs)} sources...")
    
    for cfg in current_test_configs:
        path = cfg['path']
        task_name = cfg['task_name']
        s_col = cfg['smiles_col']
        l_col = cfg['label_col']  
        if type(path) is  str:
            df = pd.read_csv(path)
        else:  
            df = path
        count = 0
        for _, row in df.iterrows():
            smi = row[s_col]
            label = float(row[l_col]) 
            label_dict = {task_name: label}
            
            smiles_list.append(smi)
            labels_list.append(label_dict)
            count += 1


    dataset = MoleculeDataset(smiles_list, labels_list, tokenizer)
    collator = MultitaskCollator(task_names=all_task_names)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator, num_workers=8)

    return test_loader
    
class MultiTaskChemBERTa(nn.Module):
    def __init__(self, model_name, task_names, head_dropout=0.5, head_hidden_dim=512):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.config = self.encoder.config
        self.task_names = task_names
        self.heads = nn.ModuleDict()
        

        self.task_log_vars = nn.ParameterDict()
        
        for name in task_names:
            self.heads[name] = nn.Sequential(
                nn.Dropout(head_dropout),
                nn.Linear(self.config.hidden_size, head_hidden_dim),
                nn.Tanh(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden_dim, 1)
            )

            self.task_log_vars[name] = nn.Parameter(torch.zeros(()))

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :] 
        
        logits_dict = {}
        losses_dict = {} 
        
        for i, task_name in enumerate(self.task_names):
            logits = self.heads[task_name](pooled_output).squeeze(-1)
            logits_dict[task_name] = logits
            
            if labels is not None:
                task_labels = labels[:, i]
                mask = task_labels != -100
                
                if mask.sum() > 0:
                    valid_logits = logits[mask]
                    valid_labels = task_labels[mask]
                    
                    loss_fct = nn.BCEWithLogitsLoss() 
                    raw_loss = loss_fct(valid_logits, valid_labels)
                    
                    if task_name in self.task_log_vars:
                        log_var = self.task_log_vars[task_name]
                        precision = torch.exp(-log_var)
                        # Kendall Uncertainty Loss
                        final_loss = 0.5 * precision * raw_loss + 0.5 * log_var
                    else:
                        final_loss = raw_loss
                    
                    losses_dict[task_name] = final_loss
                    
        return {"logits": logits_dict, "losses_dict": losses_dict}

    
def save_huggingface_model(model, tokenizer, output_dir, pruned_tasks=None, pfas_mcc_best_thresh=None):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tokenizer.save_pretrained(output_dir)

    config_dict = model.config.to_dict()

    config_dict.update({
        "architectures": ["MultiTaskChemBERTa"], 
        "task_names": model.task_names,          
        "head_hidden_dim": model.heads[model.task_names[0]][1].out_features, 
        "head_dropout": model.heads[model.task_names[0]][0].p,
        "pruned_tasks": pruned_tasks if pruned_tasks is not None else [],
        "mcc_best_thresh": float(pfas_mcc_best_thresh) if pfas_mcc_best_thresh is not None else None
        })
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    
    print("model saved successfully!")
    if pruned_tasks:
        print(f"Note: recorded {len(pruned_tasks)} pruned tasks in the model config.")
    


def load_huggingface_model(model_dir, device="cpu"):

    config = AutoConfig.from_pretrained(model_dir)
    
    task_names = getattr(config, "task_names", [])
    head_hidden_dim = getattr(config, "head_hidden_dim", 512)
    head_dropout = getattr(config, "head_dropout", 0.5)
    pruned_tasks = getattr(config, "pruned_tasks", [])
    
    if len(pruned_tasks) > 0:
        print(f"Detected {len(pruned_tasks)} pruned tasks: {pruned_tasks}")
    
    model = MultiTaskChemBERTa(model_dir, task_names, head_dropout, head_hidden_dim)
    
    state_dict = torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location=device, weights_only=True)
    
    model.load_state_dict(state_dict)
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.to(device)
    model.pruned_tasks = pruned_tasks
    return model, tokenizer


def get_model(model_name, new_task_names, device):
    model, tokenizer = load_huggingface_model(model_name, device=device)

    old_task_names = [t for t in model.task_names if "AID" in t]
    pruned_tasks = getattr(model, 'pruned_tasks', [])
    
    print(f"Original task count: {len(model.task_names)}")
    
    sample_head = model.heads[old_task_names[0]]
    hidden_dim = sample_head[1].out_features 
    dropout_rate = sample_head[0].p
    
    print(f"Extending the model with new tasks: {new_task_names}")
    for new_task in new_task_names:
        if new_task in model.heads: continue 
        
        new_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(model.config.hidden_size, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1)
        )
        
        for module in new_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None: nn.init.zeros_(module.bias)
        
        model.heads[new_task] = new_head.to(device)
    

    model.task_names = sorted(old_task_names + new_task_names)
    print(f"Model extension complete: {len(model.task_names)} total tasks, {len(pruned_tasks)} pruned tasks")
    
    for name, param in model.heads.named_parameters():
        task_name = name.split('.')[0]
        if task_name in new_task_names:
            param.requires_grad = True   
        else:
            param.requires_grad = False 
            
    for param in model.task_log_vars.values():
        param.requires_grad = False
        
    return model, tokenizer

class WeightedFinetuneTrainer:
    def __init__(self, model, train_loader, val_loader, device, loss_weights, scheduler=None, accumulation_steps=1):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.loss_weights = loss_weights
        self.scheduler = scheduler
        self.accumulation_steps = accumulation_steps
        self.optimizer = None

    def train_epoch(self):
        self.model.train()
        total_loss_val, steps = 0, 0
        self.optimizer.zero_grad()
        
        for i, batch in enumerate(self.train_loader):
            input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
            output = self.model(input_ids, attention_mask, labels=labels)
            
            batch_loss = torch.tensor(0.0, device=self.device)
            valid_batch = False
            for task_name, task_loss in output['losses_dict'].items():
                w = self.loss_weights.get(task_name, 0.0)
                if w > 0:
                    batch_loss += w * task_loss
                    valid_batch = True
            
            if not valid_batch: continue

            loss = batch_loss / self.accumulation_steps
            loss.backward()
            
            if (i + 1) % self.accumulation_steps == 0 or (i + 1) == len(self.train_loader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss_val += batch_loss.item()
            steps += 1
            
        return total_loss_val / max(steps, 1)

    def find_optimal_threshold_and_mcc(self, y_true, y_probs):
        if len(np.unique(y_true)) < 2: return 0.5, 0.0
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)
        best_mcc = -1
        best_thresh = 0.5
        for thresh in thresholds:
            if thresh > 1: continue
            y_pred_binary = (y_probs >= thresh).astype(int)
            mcc = matthews_corrcoef(y_true, y_pred_binary)
            if mcc > best_mcc:
                best_mcc = mcc
                best_thresh = thresh
        return best_thresh, best_mcc

    @torch.no_grad()
    def evaluate(self):

        self.model.eval()
        
        eval_tasks = [t for t, w in self.loss_weights.items() if t not in self.model.pruned_tasks]
        
        all_preds = {t: [] for t in eval_tasks}
        all_labels = {t: [] for t in eval_tasks}

        for batch in self.val_loader:
            input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
            output = self.model(input_ids, attention_mask)
            
            for task in eval_tasks:
                if task not in self.model.task_names: continue
                idx = self.model.task_names.index(task)
                
                preds = torch.sigmoid(output['logits'][task]).cpu().numpy()
                lbls = labels[:, idx].cpu().numpy()
                
                mask = lbls != -100
                if mask.any():
                    all_preds[task].extend(preds[mask])
                    all_labels[task].extend(lbls[mask])
        
        scores_task = {}
        for task in eval_tasks:
            y_true = np.array(all_labels[task])
            y_pred = np.array(all_preds[task])

            if len(y_true) == 0: 
                scores_task[task] = 0.0
                continue
                
            try:
                auc = roc_auc_score(y_true, y_pred)
            except:
                auc = 0.5
            
            scores_task[task] = auc
            if "Invivo"  in task:
                best_thresh, best_mcc = self.find_optimal_threshold_and_mcc(y_true, y_pred)
                scores_task[f"{task}_mcc"] = best_mcc
                scores_task[f"{task}_mcc_best_thresh"] = best_thresh

        return scores_task

seeds = [42, 123, 999, 2026, 888]
def save_result_dynamically(result_dict, filepath):
    columns = [
        "Seed", "Strategy", "Replay_Ratio", 
        "pfas_auc", "pfas_mcc","pfas_mcc_best_thresh", "bio_auc", 
        "cf_auc", "cf_mcc",  
        "epoch","loss"
    ]

    for col in columns:
        if col not in result_dict:
            result_dict[col] = "error"

    df = pd.DataFrame([result_dict])

    file_exists = os.path.isfile(filepath)

    try:
        df.to_csv(filepath, mode='a', header=not file_exists, index=False, columns=columns)
        print(f"  [Auto-Save] Result appended to {filepath}")
    except Exception as e:
        print(f"  [Save Error] Failed to write csv: {e}")
        

def run_single_experiment(
    model_name_or_path,
    seed,
    replay_ratio,
    train_batch_size,
    pfas_weight,
    cf_weight,
    bio_weight,
    epochs,
    encoder_lr,
    head_lr,
    device,
    accumulation_steps=4,
    input_model=None,      
    input_tokenizer=None,  
    save_model=False,      
    save_path_suffix=""    
):
    set_seed(seed)
    
    new_task_names = ["Invivo_PFAS", "Invivo_CF"] 
    
    if input_model is not None and input_tokenizer is not None:
        model = input_model
        tokenizer = input_tokenizer
    else:
        model, tokenizer = get_model(model_name=model_name_or_path, new_task_names=new_task_names, device=device)
    
    pruned_tasks = getattr(model, 'pruned_tasks', [])
    invivo_pfas_train, invivo_pfas_test = split_invivo_datasets(invivo_data_PFAS, "invivo_PFAS", test_size=0.2, random_seed=seed, class_col='hepatotoxicity')
    invivo_cf_train, invivo_cf_test = split_invivo_datasets(invivo_data_cf, "invivo_CF", test_size=0.2, random_seed=seed, class_col='hepatotoxicity')

    train_dataloader, replay_configs = get_train_dataloader(
        invivo_pfas_train, invivo_cf_train, 
        replay_ratio=replay_ratio, 
        train_batch_size=train_batch_size, 
        tokenizer=tokenizer, 
        seed=seed, 
        task_names=model.task_names, 
        balance_target=True
    )
    
    test_dataloader = get_test_loader(invivo_pfas_test, invivo_cf_test, replay_configs, model.task_names, batch_size=512, tokenizer=tokenizer)
    LOSS_WEIGHTS = {}
    for task in model.task_names:
        if task == "Invivo_PFAS":
            LOSS_WEIGHTS[task] = pfas_weight
        elif task == "Invivo_CF":
            LOSS_WEIGHTS[task] = cf_weight
        elif task in pruned_tasks:
            LOSS_WEIGHTS[task] = 0.0
        else:
            LOSS_WEIGHTS[task] = bio_weight
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.encoder.named_parameters() if p.requires_grad], "lr": encoder_lr},
        {"params": [p for n, p in model.heads.named_parameters() if p.requires_grad], "lr": head_lr}
    ]
    optimizer = AdamW(optimizer_grouped_parameters) 
    
    total_steps = len(train_dataloader) * epochs
    warmup_steps = int(total_steps * 0.1)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    trainer = WeightedFinetuneTrainer(model, train_dataloader, test_dataloader, device, LOSS_WEIGHTS, scheduler=scheduler, accumulation_steps=accumulation_steps)
    trainer.optimizer = optimizer

    best_pfas_auc = 0.0
    best_metrics = {
        "epoch": 0,
        "pfas_auc": 0.0,
        "pfas_mcc": 0.0,
        "cf_auc": 0.0,
        "cf_mcc": 0.0,
        "bio_auc": 0.0,
        "pfas_mcc_best_thresh":0.0,
        "loss": 0.0
    }
    history = []
    logger.info(f"Start Training: Seed={seed}, Replay={replay_ratio}, PFAS_W={pfas_weight}, CF_W={cf_weight}")

    for epoch in range(epochs):
        avg_loss = trainer.train_epoch()
        scores = trainer.evaluate()
        
        pfas_auc = scores.get("Invivo_PFAS", 0)
        pfas_mcc = scores.get("Invivo_PFAS_mcc", 0)
        cf_auc = scores.get("Invivo_CF", 0)
        cf_mcc = scores.get("Invivo_CF_mcc", 0)
        pfas_mcc_best_thresh = scores.get("Invivo_PFAS_mcc_best_thresh", None)
        
        bioassay_aucs = []
        for task, score in scores.items():
            if "aid:" in task.lower() and "mcc" not in task.lower():
                bioassay_aucs.append(score)
        
        bio_auc_mean = np.nanmean(bioassay_aucs) if bioassay_aucs else 0.0

        metric_to_track = pfas_auc if pfas_weight > 0 else cf_auc
        
        current_epoch_metrics = {
            "epoch": epoch + 1,
            "loss": avg_loss,     
            "pfas_auc": pfas_auc,
            "pfas_mcc": pfas_mcc,
            "cf_auc": cf_auc,
            "cf_mcc": cf_mcc,
            "bio_auc": bio_auc_mean,
            "pfas_mcc_best_thresh":pfas_mcc_best_thresh
        }
        
        history.append(current_epoch_metrics)
        
        if metric_to_track > best_pfas_auc:
            best_pfas_auc = metric_to_track
            best_metrics = {
                "epoch": epoch + 1,
                "loss": avg_loss,  
                "pfas_auc": pfas_auc,
                "pfas_mcc": pfas_mcc,
                "cf_auc": cf_auc,
                "cf_mcc": cf_mcc,
                "bio_auc": bio_auc_mean, 
                "pfas_mcc_best_thresh":pfas_mcc_best_thresh
            }
            print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | PFAS(AUC): {pfas_auc:.4f}, PFAS(MCC): {pfas_mcc:.4f} | CF(AUC): {cf_auc:.4f}, CF(MCC): {cf_mcc:.4f} | Bio(avg): {bio_auc_mean:.4f}  <-- New Best!")
            if save_model:
                save_dir = f"./invivo_curriculum_save_model/{save_path_suffix}_seed{seed}"
                save_huggingface_model(model, tokenizer, save_dir, pruned_tasks,pfas_mcc_best_thresh)

    return history,best_metrics, model, tokenizer

def main_experiment_with_dynamic_save():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_model_path = (
        PROJECT_ROOT/ "code"/ "multi_tasks_training"/ "best_model_with_pruning_heptox"
    )
    
    
    seeds = [42, 123, 999, 2026, 888]
    replay_ratios_to_test = [0.8] 
    
    output_csv_file = "experiment_results_realtime_save_model_s4_cl_0_8.csv"
    common_params = {
        "train_batch_size": 32,
        "encoder_lr": 1e-5,
        "head_lr": 1e-5,
        "accumulation_steps": 4,
        "device": device,
        "epochs": 20
    }
    
    print(f"Starting Experiment... Results will be auto-saved to: {output_csv_file}")
    output_history_all_csv="experiment_results_realtime_save_model_s4_cl_0_8_all_epochs.csv"
    for seed in seeds:
        print(f"\n>>> [S4] Curriculum + Bioassay (Preparing Phase 1...)")
        
        print("  ...Phase 1 (CF Common Base)...")
        history_phase1,_, model_s4_base, tokenizer_s4_base = run_single_experiment(
            model_name_or_path=base_model_path,
            seed=seed,
            replay_ratio=0.0,
            pfas_weight=0.0, cf_weight=1.0, bio_weight=0.0,
            epochs=10,
            save_model=True,
            save_path_suffix=f"s4_curriculum_stage1_seed_{seed}",
            **{k:v for k,v in common_params.items() if k != 'epochs'}
        )
        for epoch_data in history_phase1:
            row = {"Seed": seed, "Strategy": "4_Curriculum_Phase1", "Replay_Ratio": 0.0, **epoch_data}
            save_result_dynamically(row, output_history_all_csv)
            
        for ratio in replay_ratios_to_test:
            print(f"  ...Phase 2 (PFAS + Bio Replay={ratio})...")
            model_phase2_input = copy.deepcopy(model_s4_base)
            
            history_phase2,res4, model_s2, _ = run_single_experiment(
                model_name_or_path=None,
                input_model=model_phase2_input, 
                input_tokenizer=tokenizer_s4_base,
                seed=seed,
                replay_ratio=ratio,
                pfas_weight=5.0, cf_weight=1.0, bio_weight=1.0,
                save_model=True,
                save_path_suffix=f"s4_curriculum_stage2_seed_{seed}_replay_{ratio}",
                **common_params
            )
            row_s4 = {
                "Seed": seed, "Strategy": "4_Curriculum_Bio", "Replay_Ratio": ratio,
                **res4
            }
            for epoch_data in history_phase2:
                row = {"Seed": seed, "Strategy": "4_Curriculum_phase2", "Replay_Ratio": ratio, **epoch_data}
                save_result_dynamically(row, output_history_all_csv)
            save_result_dynamically(row_s4, output_csv_file) 
            del model_phase2_input
            torch.cuda.empty_cache()
        
        del model_s4_base, tokenizer_s4_base
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\nAll experiments finished. Final data saved to {output_csv_file}")

if __name__ == "__main__":

    invivo_data = pd.read_csv(
        DATA_DIR / "invivodata" / "hep_all_unique_standardized.csv"
    )
    invivo_data_PFAS = invivo_data[invivo_data["pfas_target"]==1]
    invivo_data_cf = invivo_data[(invivo_data["contains_CF"]==1) & (invivo_data["pfas_target"]==0)]
    print(f"Data Loaded: PFAS={len(invivo_data_PFAS)}, CF={len(invivo_data_cf)}")

    main_experiment_with_dynamic_save()

