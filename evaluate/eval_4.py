from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import torch
from transformers import ViltProcessor
import numpy as np
import os
from huggingface_hub import login
import pandas as pd
from tqdm import tqdm
import argparse
from dotenv import load_dotenv

# Import data processing components
from modules.dataset_processing import VQADataset, collate_fn, transform, ROOT_FOLDER
from modules.model import CustomViltForVQA
from modules.metrics import compute_metrics

load_dotenv()

dataset_path = r"C:\\Users\\tcgnh\\OneDrive\\Desktop\\workspace\\vqa-info-data"
test_csv_path = os.path.join(dataset_path, "test.csv")
test_df = pd.read_csv(test_csv_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViLT VQA Eval Script")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of eval samples per batch")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of data workers")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Load model and processor
    model = CustomViltForVQA.from_pretrained("phonghoccode/vilt-vqa-finetune-pytorch").to(device)
    processor = ViltProcessor.from_pretrained("phonghoccode/vilt-vqa-finetune-pytorch")
    
    # Initialize dataset and dataloader
    test_dataset = VQADataset(test_df, processor, model.config.label2id, transform, ROOT_FOLDER)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=lambda batch: collate_fn(batch, processor),
                                 num_workers=0)
    
    model.eval()
    eval_logits, eval_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for eval_batch in tqdm(test_dataloader, desc="Evaluating", leave=False):
            eval_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in eval_batch.items()}
            
            # Forward pass
            outputs = model(**eval_batch)
            loss = outputs.loss.item()
            total_loss += loss
            logits = outputs.logits.cpu().numpy()
            labels = eval_batch['labels'].cpu().numpy()
            
            eval_logits.extend(logits)
            eval_labels.extend(labels)
    
    # Convert logits to probabilities before taking argmax
    eval_logits = np.array(eval_logits)
    eval_labels = np.array(eval_labels)
    
    avg_loss = total_loss / len(test_dataloader) if len(test_dataloader) > 0 else total_loss
    metrics = compute_metrics(eval_logits, eval_labels)
    
    print(f"Evaluation Metrics: {metrics}")
    print(f"Evaluation Loss: {avg_loss}")
