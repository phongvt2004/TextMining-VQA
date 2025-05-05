from google import genai

class GeminiVQA:
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def ask_zeroshot(self, image_file, question: str) -> str:
        uploaded_file = self.client.files.upload(file=image_file)
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[
                uploaded_file,
                {"text": question}
            ]
        )
        return response.text

    def ask_fewshot(self, image_file, question: str) -> str:
        prompt = []
        few_shot_examples = [
            {"image_file": "sample/bottle.jpg", "question": "How many bottles are there?", "answer": "1."},
            {"image_file": "sample/bowl.jpeg", "question": "How many bottles are there?", "answer": "1."},
            {"image_file": "sample/cat.jpg", "question": "What is the color of the cat?", "answer": "Orange."},
            {"image_file": "sample/clock.jpeg", "question": "Location of the clock in the image?", "answer": "Center."},
            {"image_file": "sample/dog.jpeg", "question": "Location of the dog in the image?", "answer": "Center."},
            {"image_file": "sample/motorcycle.jpeg", "question": "What is the color of the motorcycle?", "answer": "Blue."},
            {"image_file": "sample/plane.png", "question": "What is the color of the plane?", "answer": "Blue."},
            {"image_file": "sample/scissors.jpeg", "question": "Location of the scissors in the image?", "answer": "Left."},
            {"image_file": "sample/wineglass.jpeg", "question": "How many wine glasses are there?", "answer": "1."},
            {"image_file": "sample/zebra.jpg", "question": "How many zebras are there?", "answer": "1."}
        ]
        if few_shot_examples:
            for example in few_shot_examples:
                ex_file = self.client.files.upload(file=example["image_file"])
                prompt.extend([
                    ex_file,
                    {"text": f"Q: {example['question']}\nA: {example['answer']}"}
                ])

        uploaded_file = self.client.files.upload(file=image_file)
        prompt.extend([
            uploaded_file,
            {"text": f"Q: {question}\nA:"}
        ])

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return response.text.strip()