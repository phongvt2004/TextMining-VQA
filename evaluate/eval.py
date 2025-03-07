import torch
from transformers import AutoProcessor, ViltForQuestionAnswering
import pandas as pd
from PIL import Image
from tqdm import tqdm
import os
import numpy as np
from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import f1_score

#Set dataset paths (Adjust as per your folder structure)
dataset_path = r"C:\Users\tcgnh\OneDrive\Desktop\workspace\vqa-info-data"
test_csv_path = os.path.join(dataset_path, "test.csv")
image_base_path = dataset_path  # Assuming images are in the dataset folder

#Load dataset
df = pd.read_csv(test_csv_path)

#Load the model and processor
model_name = "phonghoccode/vilt-vqa-finetune-pytorch"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_name)
model = ViltForQuestionAnswering.from_pretrained(model_name).to(device)
model.eval()

#Initialize evaluation metrics
exact_matches = 0
total = 0
all_predictions = []
all_ground_truths = []
bleu_scores = []
category_accuracy = defaultdict(list)

#Function to evaluate model
def evaluate_vqa_model(df, model, processor):
    global exact_matches, total, all_predictions, all_ground_truths, bleu_scores

    for _, row in tqdm(df.iterrows(), total=len(df)):
        image_path = os.path.join(image_base_path, row["image_path"])
        question = row["question"]
        true_answer = row["answer"]

        try:
            # Load and process image
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, text=question, return_tensors="pt").to(device)

            # Get model predictions
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_answer_idx = logits.argmax(-1).item()

            # Convert index to actual answer
            predicted_answer = processor.tokenizer.decode([predicted_answer_idx], skip_special_tokens=True)

            all_predictions.append(predicted_answer)
            all_ground_truths.append(true_answer)

            # Exact Match (EM)
            if predicted_answer.lower().strip() == true_answer.lower().strip():
                exact_matches += 1

            # Compute BLEU Score
            bleu = sentence_bleu([true_answer.split()], predicted_answer.split())
            bleu_scores.append(bleu)

            # Categorize questions for per-category analysis
            if "how many" in question.lower():
                category_accuracy["counting"].append(predicted_answer == true_answer)
            elif "what" in question.lower():
                category_accuracy["what"].append(predicted_answer == true_answer)
            elif "is there" in question.lower() or "are there" in question.lower():
                category_accuracy["yes/no"].append(predicted_answer == true_answer)
            else:
                category_accuracy["other"].append(predicted_answer == true_answer)

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

        total += 1

#Run evaluation
evaluate_vqa_model(df, model, processor)

#Compute final metrics
accuracy = exact_matches / total if total > 0 else 0
f1 = f1_score(
    [a.lower().strip() for a in all_ground_truths],
    [p.lower().strip() for p in all_predictions],
    average="weighted"
)
avg_bleu = np.mean(bleu_scores)

#Per-category accuracy
category_results = {cat: np.mean(acc) for cat, acc in category_accuracy.items()}

#Print results
print(f"🔹 Model Evaluation Results:")
print(f"✅ Exact Match (EM) Score: {accuracy:.4f}")
print(f"✅ F1 Score: {f1:.4f}")
print(f"✅ BLEU Score: {avg_bleu:.4f}")
print(f"✅ Per-Category Accuracy: {category_results}")

#Save predictions
pred_df = pd.DataFrame({
    "image_id": df["image_id"],
    "question": df["question"],
    "predicted_answer": all_predictions,
    "true_answer": all_ground_truths
})
pred_df.to_csv("./vqa_predictions_detailed.csv", index=False)

print("\n Evaluation Complete! Predictions saved to vqa_predictions_detailed.csv")
