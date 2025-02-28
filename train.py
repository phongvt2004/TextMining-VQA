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
from dotenv import load_dotenv

load_dotenv()

# Import data processing components
from modules.dataset_processing import preprocess_lazy, VQADataset, collate_fn, transform, ROOT_FOLDER

# Import model and metrics
from modules.model import CustomViltForVQA
from modules.metrics import compute_metrics

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


df = pd.read_csv("/workspace/vqa-info-data/data.csv")
train_df = pd.read_csv("/workspace/vqa-info-data/train.csv")
val_df = pd.read_csv("/workspace/vqa-info-data/val.csv")


# Create an answer vocabulary - Keep vocab creation in train file
unique_answers = set(df["answer"].tolist())  # Get all unique answers
answer2id = {ans: i for i, ans in enumerate(unique_answers)}  # Map to index
id2answer = {i: ans for i, ans in enumerate(unique_answers)}  # Map to index
num_labels = len(answer2id)  # Total unique answers

# Load the ViLT processor - Keep processor loading in train file
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# Set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Initialize datasets and dataloaders - Pass processor, answer2id, transform
train_dataset = VQADataset(train_df, processor, answer2id, transform, ROOT_FOLDER)
val_dataset = VQADataset(val_df, processor, answer2id, transform, ROOT_FOLDER)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True,
                              collate_fn=lambda batch: collate_fn(batch, processor), # Pass processor to collate_fn
                              num_workers=8)
val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False,
                            collate_fn=lambda batch: collate_fn(batch, processor), # Pass processor to collate_fn
                            num_workers=8)


# Load ViLT model
model = CustomViltForVQA.from_pretrained("dandelin/vilt-b32-finetuned-vqa", num_labels=num_labels, id2label=id2answer, label2id=answer2id, ignore_mismatched_sizes=True).to(device)


# Optimizer and scheduler
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
total_steps = len(train_dataloader) * 3 # num_train_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Login programmatically
login(token=os.getenv("HUGGINGFACE_TOKEN"))
wandb.login(key=os.getenv("WANDB_API_KEY"))

# Initialize WandB and TensorBoard
wandb.init(project="vilt-vqa-finetune", name="vilt-vqa-run-1")
writer = tensorboardX.SummaryWriter("./runs") # Optional TensorBoard logging


output_dir = "./results_pytorch" # Directory to save checkpoints
hub_model_id = "phonghoccode/vilt-vqa-finetune-pytorch" # Change to your Hugging Face model repo
save_steps = 500
logging_steps = 100
push_to_hub_steps = save_steps
eval_steps = 250 # Define separate eval steps


# Training loop
num_epochs = 6
global_step = 0
eval_step_count = 0 # Counter for eval steps
best_eval_loss = float('inf') # Track best eval loss for saving best model

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
    for batch in progress_bar:
        global_step += 1

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        train_loss += loss.item()

        if global_step % logging_steps == 0:
            avg_train_loss = train_loss / logging_steps
            wandb.log({"train_loss": avg_train_loss, "lr": scheduler.get_last_lr()[0]}, step=global_step)
            writer.add_scalar('Loss/train', avg_train_loss, global_step) # Tensorboard logging
            progress_bar.set_postfix({"loss": avg_train_loss})
            train_loss = 0.0

        if global_step % eval_steps == 0: # Evaluate at eval_steps interval
            model.eval()
            eval_loss = 0.0
            eval_logits = []
            eval_labels = []
            eval_progress_bar = tqdm(val_dataloader, desc=f"Evaluating at step {global_step}", leave=False)
            eval_step_count += 1 # Increment eval step counter for logging purposes
            for eval_batch in eval_progress_bar:
                eval_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in eval_batch.items()}
                with torch.no_grad():
                    eval_outputs = model(**eval_batch)
                    batch_eval_loss = eval_outputs.loss
                    eval_loss += batch_eval_loss.item()
                    eval_logits.extend(eval_outputs.logits.cpu().numpy())
                    eval_labels.extend(eval_batch['labels'].cpu().numpy())

            avg_eval_loss_epoch = eval_loss / len(val_dataloader) if len(val_dataloader) > 0 else eval_loss # Average eval loss for epoch
            metrics = compute_metrics(np.array(eval_logits), np.array(eval_labels))
            wandb.log({"eval_loss": avg_eval_loss_epoch, **metrics, "eval_step": eval_step_count, "global_step": global_step}, step=global_step) # Log epoch eval loss and metrics, include eval step count
            writer.add_scalar('Loss/eval_epoch', avg_eval_loss_epoch, global_step) # Tensorboard logging for epoch eval loss
            for metric_name, metric_value in metrics.items():
                writer.add_scalar(f'Metrics/{metric_name}', metric_value, global_step) # Tensorboard metrics
            logger.info(f"Evaluation at step {global_step} - Eval Loss: {avg_eval_loss_epoch}")


            if avg_eval_loss_epoch < best_eval_loss: # Save best model based on eval loss
                best_eval_loss = avg_eval_loss_epoch
                checkpoint_path_best = os.path.join(output_dir, f"checkpoint-{global_step}-best")
                model.save_pretrained(checkpoint_path_best)
                processor.save_pretrained(checkpoint_path_best)
                logger.info(f"Best model saved at step {global_step} with eval loss: {best_eval_loss}")
            model.train() # Set back to train mode after evaluation


        if global_step % save_steps == 0: # Save checkpoint at save_steps interval
            checkpoint_path = os.path.join(output_dir, f"checkpoint-{global_step}") # Save every save_steps
            model.save_pretrained(checkpoint_path)
            processor.save_pretrained(checkpoint_path)
            logger.info(f"Checkpoint saved at step {global_step}")

            if hub_model_id and global_step % push_to_hub_steps == 0:
                model.push_to_hub(hub_model_id, checkpoint_path)
                processor.push_to_hub(hub_model_id, checkpoint_path)
                logger.info(f"Pushed checkpoint to hub at step {global_step}")


# Save final model and processor
final_output_dir = os.path.join(output_dir, "final_model")
model.save_pretrained(final_output_dir)
processor.save_pretrained(final_output_dir)
if hub_model_id:
    model.push_to_hub(hub_model_id, final_output_dir)
    processor.push_to_hub(hub_model_id, final_output_dir)
    logger.info("Final model pushed to hub")

wandb.finish()
writer.close() # Close TensorBoard writer
logger.info("Training finished!")
