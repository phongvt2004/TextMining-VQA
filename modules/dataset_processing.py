from torchvision import transforms
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import ViltProcessor
import os
import pandas as pd

ROOT_FOLDER = r"C:\Users\tcgnh\OneDrive\Desktop\vqa_dataset\images"

def preprocess_lazy(sample):
    # Don't load images here, just store the paths
    return {"image_path": os.path.join(ROOT_FOLDER, sample["image_path"]), "question": sample["question"], "answer": sample["answer"]}

# Chuẩn hóa ảnh để phù hợp với ViLT
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


class VQADataset(Dataset):
    def __init__(self, dataframe, processor, answer2id, transform, root_folder):
        self.dataframe = dataframe
        self.processor = processor
        self.answer2id = answer2id
        self.transform = transform
        self.root_folder = root_folder

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        sample = self.dataframe.iloc[idx]
        image_path = os.path.join(self.root_folder, sample["image_path"])
        question = str(sample["question"]).strip()
        answer = sample["answer"]

        image = Image.open(image_path).convert("RGB")  # Đảm bảo ảnh luôn là RGB
        image = self.transform(image)  # Áp dụng transform

        labels = torch.tensor(self.answer2id[answer], dtype=torch.long)

        return {"image": image, "question": question, "labels": labels}

def collate_fn(batch, processor):
    images = [sample["image"] for sample in batch]
    questions = [sample["question"] for sample in batch]
    labels = torch.stack([sample["labels"] for sample in batch])

    encoding = processor(images=images, text=questions, return_tensors="pt", padding=True)
    encoding["labels"] = labels
    return encoding
