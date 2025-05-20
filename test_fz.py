import os
from gemini_Calls import GeminiRotator
from tqdm import tqdm
import pandas as pd

API_KEYS = ["AIzaSyAcC3UTdxkOENHLjplP9Rkd_66XeX61L3o", "AIzaSyCbWTETatIkc2w9N7dLVYrYfYtjf6BL0M8", "AIzaSyD1HKKrmNFGbLBcDdGRALo6JAJyMdgTyTU", "AIzaSyDzDXa9rA83lzPNhDAJEP5udLTGJcOrvnU", "AIzaSyCutxRksGyG7GGoTdKVY8eKwVZYPEUvnwc", "AIzaSyDgje6Of5JaxBZVllNKearH2aP5tDI9ycE" ,"AIzaSyDoRmOJK7NK6K6eB6YHgV_SGmhPdKo4pvQ", "AIzaSyD22-9DA9oTtVVxQ1iOmrmx7Xre_kaLqdU", "AIzaSyBJtVot2TPrrX4B7ccFeprreE42gGrCcaU", "AIzaSyCkks9uQoWgG3Z1qWYCa_KQH7KTVtDfjzY"]

rotator = GeminiRotator(API_KEYS)
df = pd.read_csv("test_checked_v8.csv")
zeroshot_preds = []
fewshot_preds = []
base_dir = r"D:\Project\VQA\vqa_dataset\images"

for _, row in tqdm(df.iterrows(), total=len(df)):
    image_file = os.path.join(base_dir, row["image_path"])
    question = row["question"]
    z_pred = rotator.call("ask_zeroshot", image_file, question)
    f_pred = rotator.call("ask_fewshot", image_file, question)

    zeroshot_preds.append(z_pred)
    fewshot_preds.append(f_pred)

df["zeroshot_pred"] = zeroshot_preds
df["fewshot_pred"] = fewshot_preds
df.to_csv("test_results_with_predictions.csv", index=False)

