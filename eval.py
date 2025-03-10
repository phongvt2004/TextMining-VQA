from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import torch
from transformers import ViltProcessor, ViltConfig
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np
import torch.nn as nn
from transformers import ViltForQuestionAnswering
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import List, Optional, Tuple, Union
import sys
import os
from huggingface_hub import login
import wandb
import pandas as pd
from tqdm import tqdm
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
import logging
import tensorboardX
import warnings
warnings.filterwarnings("ignore")
import argparse
from dotenv import load_dotenv

load_dotenv()

# Import data processing components
from modules.dataset_processing import preprocess_lazy, VQADataset, collate_fn, transform, ROOT_FOLDER

# Import model and metrics
from modules.model import CustomViltForVQA
from modules.metrics import compute_metrics
test_df = pd.read_csv("/workspace/vqa-info-data/test.csv")


if __name__ == "__main__": # Add main block

    # Argument Parser
    parser = argparse.ArgumentParser(description="ViLT VQA Eval Script")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of eval samples per batch")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of data workers")
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Load ViLT model
    model = CustomViltForVQA.from_pretrained("phonghoccode/vilt-vqa-finetune").to(device)
    processor = ViltProcessor.from_pretrained("phonghoccode/vilt-vqa-finetune")
    
    # Initialize datasets and dataloaders - Pass processor, answer2id, transform
    test_dataset = VQADataset(test_df, processor, model.config.label2id, transform, ROOT_FOLDER)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                collate_fn=lambda batch: collate_fn(batch, processor), # Pass processor to collate_fn
                                num_workers=args.num_workers)


    model.eval()
    with torch.no_grad():
        eval_loss = 0.0
        eval_logits = []
        eval_labels = []
        eval_progress_bar = tqdm(test_dataloader, desc=f"Evaluating at step {global_step}", leave=False)
        eval_step_count += 1 # Increment eval step counter for logging purposes
        for eval_batch in eval_progress_bar:
            eval_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in eval_batch.items()}
            with torch.no_grad():
                eval_outputs = model(**eval_batch)
                batch_eval_loss = eval_outputs.loss
                eval_loss += batch_eval_loss.item()
                eval_logits.extend(eval_outputs.logits.cpu().numpy())
                eval_labels.extend(eval_batch['labels'].cpu().numpy())

        avg_eval_loss_epoch = eval_loss / len(test_dataloader) if len(test_dataloader) > 0 else eval_loss # Average eval loss for epoch
        metrics = compute_metrics(np.array(eval_logits), np.array(eval_labels))
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}:{metric_value}")
        print(f"Eval Loss: {avg_eval_loss_epoch}")
