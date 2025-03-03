from PIL import Image
from transformers import ViltProcessor, ViltConfig
from modules.model import CustomViltForVQA
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViLT VQA Finetuning Script")
    parser.add_argument("--model", type=str, default=None, help="Path or model name")
    parser.add_argument("--image_path", type=str, default=None, help="Path to image")
    parser.add_argument("--question", type=str, default=None, help="Question")
    
    args = parser.parse_args()
    model = CustomViltForVQA.from_pretrained(args.model)
    processor = ViltProcessor.from_pretrained(args.model)
    image = Image.open(args.image_path)
    question = args.question
    inputs = processor(image, question, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = np.argmax(logits.detach().numpy(), axis=-1)
    answer = model.config.id2label[predictions[0]]
    print(answer)