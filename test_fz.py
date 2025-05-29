import os
from gemini_Calls import GeminiRotator
from tqdm import tqdm
import pandas as pd

def check_pred(pred, valid_answers):
    pred = pred.strip().lower()
    for val in valid_answers:
        if pred.lower() == val.lower():
            return True
    return False

API_KEYS = ["AIzaSyAcC3UTdxkOENHLjplP9Rkd_66XeX61L3o", "AIzaSyCbWTETatIkc2w9N7dLVYrYfYtjf6BL0M8", "AIzaSyD1HKKrmNFGbLBcDdGRALo6JAJyMdgTyTU", "AIzaSyDzDXa9rA83lzPNhDAJEP5udLTGJcOrvnU", "AIzaSyCutxRksGyG7GGoTdKVY8eKwVZYPEUvnwc", "AIzaSyDgje6Of5JaxBZVllNKearH2aP5tDI9ycE" ,"AIzaSyDoRmOJK7NK6K6eB6YHgV_SGmhPdKo4pvQ", "AIzaSyD22-9DA9oTtVVxQ1iOmrmx7Xre_kaLqdU", "AIzaSyBJtVot2TPrrX4B7ccFeprreE42gGrCcaU", "AIzaSyCkks9uQoWgG3Z1qWYCa_KQH7KTVtDfjzY"]

rotator = GeminiRotator(API_KEYS)
df = pd.read_csv("test_checked_v8.csv")
zeroshot_preds = []
fewshot_preds = []
base_dir = r"D:\Project\VQA\vqa_dataset\images"
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
q_rule = f"""
You are only allowed to answer using one or more of the following exact string values:

{set(rule.values())}

- Do NOT return any keys or labels like "42" or "95" — only use the exact **values** listed above.
- Note: Some valid answers may look like numbers (e.g., "8", "1", "0", etc.) — these are allowed **only if they appear in the list above**.
- If no valid answer is possible from the list, return "No answer possible".
- Do NOT explain your answer or include anything else — return the value only.

Now, answer the following question accordingly:
"""

retries = 5
for _, row in tqdm(df.iterrows(), total=len(df)):
    image_file = os.path.join(base_dir, row["image_path"])
    question = row["question"]
    question = q_rule + question

    count = 0
    while True:
        z_pred = rotator.call("ask_zeroshot", image_file, question)
        check = check_pred(z_pred, rule.values())
        count += 1
        if check or count > retries:
            break
    count = 0

    while True:
        f_pred = rotator.call("ask_fewshot", image_file, question)
        check = check_pred(f_pred, rule.values())
        count += 1
        if check or count > retries:
            break

    if not check_pred(z_pred, rule.values()):
        z_pred = "No answer possible"
    if not check_pred(f_pred, rule.values()):
        f_pred = "No answer possible"

    zeroshot_preds.append(z_pred)
    fewshot_preds.append(f_pred)

df["zeroshot_pred"] = zeroshot_preds
df["fewshot_pred"] = fewshot_preds
df.to_csv("test_results_with_predictions_fallback.csv", index=False)

