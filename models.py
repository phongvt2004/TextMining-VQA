import streamlit as st
from transformers import ViltForQuestionAnswering, BlipForQuestionAnswering, AutoProcessor
from PIL import Image

models = {
    "BLIP": (AutoProcessor, BlipForQuestionAnswering, "Salesforce/blip-vqa-base"),
    "ViLT": (AutoProcessor, ViltForQuestionAnswering, "dandelin/vilt-b32-finetuned-vqa"),
    "My Model": (AutoProcessor, ViltForQuestionAnswering, "phonghoccode/vilt-vqa-finetune-pytorch")
}

def get_format_response(image,question,selected_model):

    processor, model_class, model_name = models[selected_model]
    processor = processor.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name)
    
    encoding = processor(image, question, return_tensors="pt")
    
    if selected_model in ['ViLT', 'My Model']:
        outputs = model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        answer = model.config.id2label[idx]
        return answer
    else:
        outputs = model.generate(**encoding)
        answer = processor.decode(outputs[0], skip_special_tokens=True)
        return answer

def run():
    st.title("Visual Question Answering (VQA)")
    st.subheader("A demo app showcasing VQA models.")

    selected_model = st.sidebar.selectbox("Select Model", list(models.keys()))

    uploaded_image = st.file_uploader("Upload Image")

    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image")

    question = st.text_input("Ask a Question about the Image")

    if uploaded_image and question:
        answer = get_format_response(image, question, selected_model)
        st.write(f"🤔 {selected_model} Answer: {answer}")

run()
