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
    all_questions = []  # Store questions separately

    with torch.no_grad():
        for eval_batch in tqdm(test_dataloader, desc="Evaluating", leave=False):
            eval_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in eval_batch.items()}
            
            # Store questions and answers for debugging
            batch_questions = eval_batch.pop("questions", ["UNKNOWN"] * len(eval_batch["labels"]))  
            all_questions.extend(batch_questions)  # Store for later printing
            

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

    # Debugging output: print predictions vs ground truth
    eval_preds = np.argmax(eval_logits, axis=-1)
    # Get label mappings (class index → answer text)
    id2label = {v: k for k, v in model.config.label2id.items()}  # Reverse mapping
    # Print debugging output: question, predicted answer (mapped), and true answer
    print("\n--- Debugging Output (First 10 Samples) ---")
    for i in range(min(10, len(eval_preds))):
        pred_answer = id2label.get(eval_preds[i], "UNKNOWN")  # Convert index to text
        actual_answer = id2label.get(eval_labels[i], "UNKNOWN")  # Convert index to text

        print(f"Q: {all_questions[i]}")
        print(f"Predicted: {pred_answer} | Actual: {actual_answer}\n") 

    # Save results to CSV
    debug_data = {
        "Question": all_questions[:len(eval_preds)],  # Ensure length matches
        "Predicted Answer": [id2label.get(p, "UNKNOWN") for p in eval_preds],
        "Actual Answer": [id2label.get(a, "UNKNOWN") for a in eval_labels],
    }
    debug_df = pd.DataFrame(debug_data)
    debug_df.to_csv("vqa_debug_output.csv", index=False, encoding="utf-8")

    print("\nSaved debugging results to 'vqa_debug_output.csv'")
                