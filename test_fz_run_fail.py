import os
import pandas as pd
from gemini_Calls import GeminiRotator
from tqdm import tqdm
import time
import random

API_KEYS = ["AIzaSyAcC3UTdxkOENHLjplP9Rkd_66XeX61L3o", "AIzaSyCbWTETatIkc2w9N7dLVYrYfYtjf6BL0M8", "AIzaSyD1HKKrmNFGbLBcDdGRALo6JAJyMdgTyTU", "AIzaSyDzDXa9rA83lzPNhDAJEP5udLTGJcOrvnU", "AIzaSyCutxRksGyG7GGoTdKVY8eKwVZYPEUvnwc", "AIzaSyDgje6Of5JaxBZVllNKearH2aP5tDI9ycE" ,"AIzaSyDoRmOJK7NK6K6eB6YHgV_SGmhPdKo4pvQ", "AIzaSyD22-9DA9oTtVVxQ1iOmrmx7Xre_kaLqdU", "AIzaSyBJtVot2TPrrX4B7ccFeprreE42gGrCcaU", "AIzaSyCkks9uQoWgG3Z1qWYCa_KQH7KTVtDfjzY"]

rotator = GeminiRotator(API_KEYS)
df = pd.read_csv("test_results_with_predictions.csv")
base_dir = r"D:\Project\VQA\vqa_dataset\images"

# Create backup
df.to_csv("test_results_with_predictions_backup.csv", index=False)

def safe_call(rotator, method, image_file, question, retries=5, base_delay=5):
    for attempt in range(retries):
        try:
            return rotator.call(method, image_file, question)
        except Exception as e:
            print(f"Unexpected error on attempt {attempt+1}/{retries}: {e}")
            delay = base_delay * (2 ** attempt) + random.uniform(0, 3)
            print(f"Retrying after {delay:.2f} seconds...")
            time.sleep(delay)
    return "No answer possible"

rule = {
    "0": "No",
    "1": "sports ball",
    "2": "above",
    "3": "Orange",
    "4": "cat",
    "5": "Behind",
    "6": "wine glass",
    "7": "Above",
    "8": "stop sign",
    "9": "traffic light",
    "10": "cake",
    "11": "apple",
    "12": "8",
    "13": "sandwich",
    "14": "carrot",
    "15": "cow",
    "16": "scissors",
    "17": "cell phone",
    "18": "6",
    "19": "airplane",
    "20": "skateboard",
    "21": "bottle",
    "22": "remote",
    "23": "Below",
    "24": "sheep",
    "25": "person",
    "26": "fire hydrant",
    "27": "motorcycle",
    "28": "potted plant",
    "29": "spoon",
    "30": "Left",
    "31": "clock",
    "32": "book",
    "33": "handbag",
    "34": "4",
    "35": "fork",
    "36": "elephant",
    "37": "5",
    "38": "tie",
    "39": "giraffe",
    "40": "toothbrush",
    "41": "teddy bear",
    "42": "1",
    "43": "Front",
    "44": "orange",
    "45": "right",
    "46": "Green",
    "47": "Pink",
    "48": "snowboard",
    "49": "hot dog",
    "50": "donut",
    "51": "9",
    "52": "vase",
    "53": "surfboard",
    "54": "baseball bat",
    "55": "tennis racket",
    "56": "train",
    "57": "White",
    "58": "parking meter",
    "59": "cup",
    "60": "left",
    "61": "Red",
    "62": "tv",
    "63": "bowl",
    "64": "horse",
    "65": "Black",
    "66": "banana",
    "67": "Right",
    "68": "keyboard",
    "69": "dining table",
    "70": "kite",
    "71": "bench",
    "72": "baseball glove",
    "73": "sink",
    "74": "mouse",
    "75": "pizza",
    "76": "7",
    "77": "laptop",
    "78": "bed",
    "79": "bear",
    "80": "umbrella",
    "81": "car",
    "82": "couch",
    "83": "truck",
    "84": "Yellow",
    "85": "boat",
    "86": "Blue",
    "87": "bird",
    "88": "dog",
    "89": "zebra",
    "90": "0",
    "91": "bus",
    "92": "refrigerator",
    "93": "bicycle",
    "94": "knife",
    "95": "3",
    "96": "frisbee",
    "97": "Yes",
    "98": "skis",
    "99": "suitcase",
    "100": "2",
    "101": "chair",
    "102": "oven",
    "103": "Brown",
    "104": "Grey",
    "105": "broccoli",
    "106": "toilet",
    "107": "Purple"
}

def check_pred(pred, valid_answers):
    """Check if prediction is valid"""
    if pd.isna(pred) or pred is None:
        return False
    pred = str(pred).strip().strip('"')
    # Case-insensitive comparison
    return any(pred.lower() == val.lower() for val in valid_answers)

# Clean up prediction columns and handle missing values
df["zeroshot_pred"] = df["zeroshot_pred"].fillna("").astype(str).str.strip()
df["fewshot_pred"] = df["fewshot_pred"].fillna("").astype(str).str.strip()

# Define conditions for retry (empty, "No answer possible", "nan", or actual NaN)
def needs_retry(value):
    if pd.isna(value):
        return True
    value_str = str(value).strip().lower()
    return value_str in ["", "no answer possible", "nan", "none"]

# Create mask for rows that need retry
mask = (
    df["zeroshot_pred"].apply(needs_retry) |
    df["fewshot_pred"].apply(needs_retry)
)

df_retry = df[mask].copy()
print(f"Found {len(df_retry)} rows that need retry")

# Create the question rule
q_rule = f"""
You are only allowed to answer using one or more of the following exact string values:

{set(rule.values())}

- Do NOT return any keys or labels like "42" or "95" — only use the exact **values** listed above.
- Note: Some valid answers may look like numbers (e.g., "8", "1", "0", etc.) — these are allowed **only if they appear in the list above**.
- If no valid answer is possible from the list, return "No answer possible".
- Do NOT explain your answer or include anything else — return the value only.

Now, answer the following question accordingly:
"""

valid_answers = set(rule.values())

# Process each row that needs retry
for idx, row in tqdm(df_retry.iterrows(), total=len(df_retry), desc="Processing rows"):
    image_file = os.path.join(base_dir, row["image_path"])
    question = q_rule + row["question"]
    
    # Check if image file exists
    if not os.path.exists(image_file):
        print(f"Warning: Image file not found: {image_file}")
        continue

    # Handle zeroshot prediction
    if needs_retry(row["zeroshot_pred"]):
        print(f"Retrying zeroshot for row {idx}")
        z_pred = safe_call(rotator, "ask_zeroshot", image_file, question)
        if not check_pred(z_pred, valid_answers):
            z_pred = "No answer possible"
        df.loc[idx, "zeroshot_pred"] = z_pred
        print(f"Zeroshot result: {z_pred}")

    # Handle fewshot prediction
    if needs_retry(row["fewshot_pred"]):
        print(f"Retrying fewshot for row {idx}")
        f_pred = safe_call(rotator, "ask_fewshot", image_file, question)
        if not check_pred(f_pred, valid_answers):
            f_pred = "No answer possible"
        df.loc[idx, "fewshot_pred"] = f_pred
        print(f"Fewshot result: {f_pred}")
    
    # Optional: Save periodically to avoid losing progress
    if idx % 50 == 0:  # Save every 50 rows
        df.to_csv("test_results_with_predictions_temp.csv", index=False)
        print(f"Progress saved at row {idx}")

# Save final results
df.to_csv("test_results_with_predictions.csv", index=False)
print("Processing complete!")

# Optional: Print summary statistics
zero_retry_count = df["zeroshot_pred"].apply(needs_retry).sum()
few_retry_count = df["fewshot_pred"].apply(needs_retry).sum()
print(f"Remaining empty zeroshot predictions: {zero_retry_count}")
print(f"Remaining empty fewshot predictions: {few_retry_count}")