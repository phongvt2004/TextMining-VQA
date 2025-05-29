import streamlit as st
from PIL import Image
from transformers import ViltForQuestionAnswering, AutoProcessor, BeitImageProcessor, BeitForImageClassification
from timm.models import create_model
from collections import OrderedDict
import torch
import json
import tempfile
import os
import io
import traceback
import time

# --- Các thành phần giả lập hoặc import tùy chọn ---
startup_warnings = []

try:
    from modules.model import CustomViltForVQA
except ImportError:
    startup_warnings.append("MODULES: CustomViltForVQA không tìm thấy. ViLT (Custom VQA) sẽ dùng ViltForQuestionAnswering mặc định.")
    CustomViltForVQA = ViltForQuestionAnswering # Fallback

try:
    from modules.beit_3 import Beit3Processing
except ImportError:
    startup_warnings.append("MODULES: Beit3Processing không tìm thấy. BEiT3 (Local VQA) có thể không hoạt động đúng.")
    class Beit3Processing: # Hàm giả
        def __init__(self, *args, **kwargs): print("WARNING: Beit3Processing (dummy) initialized.")
        def __call__(self, image, text, **kwargs):
            print("WARNING: Beit3Processing (dummy) called.")
            return {"image": torch.randn(1, 3, 480, 480), "language_tokens": torch.randint(0, 100, (1, 40)), "padding_mask": torch.zeros(1, 40, dtype=torch.long)}

try:
    from gemini_Calls import GeminiVQA
    gemini_available = True
except ImportError:
    startup_warnings.append("MODULES: GeminiVQA không tìm thấy. Các model Gemini sẽ không khả dụng.")
    gemini_available = False
    class GeminiVQA: # Hàm giả
        def __init__(self, api_key): self.api_key = api_key
        def ask_zeroshot(self, image_file, question): raise NotImplementedError("GeminiVQA (zeroshot) không khả dụng.")
        def ask_fewshot(self, image_file, question): raise NotImplementedError("GeminiVQA (fewshot) không khả dụng.")

# --- STT Local: Whisper ---
try:
    import whisper
    whisper_available = True
except ImportError:
    startup_warnings.append("STT LOCAL (WHISPER): Thư viện 'openai-whisper' chưa được cài đặt hoặc thiếu 'ffmpeg'. STT Local sẽ không hoạt động.")
    whisper_available = False

# --- TTS (API): gTTS ---
# Cần: pip install gTTS
try:
    from gtts import gTTS
    gtts_available = True
except ImportError:
    startup_warnings.append("TTS API (gTTS): Thư viện 'gTTS' chưa được cài đặt. Chức năng nói (API) sẽ không hoạt động.")
    gtts_available = False

# --- Dịch thuật (API): translate ---
try:
    from translate import Translator
    translator_available = True
except ImportError:
    startup_warnings.append("TRANSLATE API: Thư viện 'translate' chưa được cài đặt. Chức năng dịch sẽ không hoạt động.")
    translator_available = False

# --- Ghi âm: st_audiorec ---
try:
    from st_audiorec import st_audiorec
    audiorec_available = True
except ImportError:
    startup_warnings.append("AUDIO REC: Thư viện 'st_audiorec' chưa được cài đặt. Chức năng ghi âm sẽ không hoạt động.")
    audiorec_available = False
    def st_audiorec():
        st.error("Thành phần ghi âm (st_audiorec) không khả dụng. Vui lòng cài đặt: pip install st_audiorec")
        return None

# --- Cấu hình chung ---
st.set_page_config(layout="centered", page_title="VQA (Local STT, API TTS/Dịch)")

GEMINI_API_KEY = "AIzaSyD22-9DA9oTtVVxQ1iOmrmx7Xre_kaLqdU"
WHISPER_MODEL_NAME = "medium" # "base", "small", "medium", "large"

# Ngôn ngữ được hỗ trợ: "Tên hiển thị": ("mã_whisper", "mã_translate_và_gTTS")
SUPPORTED_LANGUAGES = {
    "Tiếng Việt": ("vi", "vi"),
    "English": ("en", "en"),
}

# --- Định nghĩa Model VQA ---
models_vqa = OrderedDict()
models_vqa["BeiT3-base (Image Class.)"] = (BeitImageProcessor, BeitForImageClassification, "microsoft/beit-base-patch16-224")
models_vqa["ViLT-base (VQA)"] = (AutoProcessor, ViltForQuestionAnswering, "dandelin/vilt-b32-finetuned-vqa")
models_vqa["ViLT (Custom VQA)"] = (AutoProcessor, CustomViltForVQA, "phonghoccode/vilt-vqa-finetune-pytorch")
models_vqa["BEiT3 (Local VQA)"] = "beit3_local_vqa_placeholder"
if gemini_available:
    models_vqa["Gemini_zeroshot (API)"] = ("Gemini", GeminiVQA(api_key=GEMINI_API_KEY), "Gemini")
    models_vqa["Gemini_fewshot (API)"] = ("Gemini", GeminiVQA(api_key=GEMINI_API_KEY), "Gemini")


# --- Các hàm tải và cache model ---
@st.cache_resource
def load_stt_whisper_model_cached(model_name=WHISPER_MODEL_NAME):
    if not whisper_available: return None
    print(f"INFO: Đang tải model STT Whisper '{model_name}'...")
    try:
        model = whisper.load_model(model_name)
        print(f"INFO: Model STT Whisper '{model_name}' đã tải xong.")
        return model
    except Exception as e:
        print(f"ERROR: Lỗi tải model Whisper '{model_name}': {e}.")
        return None

# Các hàm cache model VQA (load_hf_model_and_processor_cached, load_beit3_local_model_cached)
# Giữ nguyên như phiên bản trước (đã có trong code bạn cung cấp gần đây nhất)
# ... (Copy các hàm load_hf_model_and_processor_cached và load_beit3_local_model_cached vào đây) ...
@st.cache_resource
def load_hf_model_and_processor_cached(model_key_name):
    if model_key_name not in models_vqa or models_vqa[model_key_name][0] == "Gemini" or models_vqa[model_key_name] == "beit3_local_vqa_placeholder":
        return None, None
    print(f"INFO: Loading HF model and processor for: {model_key_name}...")
    processor_class, model_class, model_hf_name = models_vqa[model_key_name]
    try:
        processor = processor_class.from_pretrained(model_hf_name)
        model = model_class.from_pretrained(model_hf_name)
        print(f"INFO: Successfully loaded: {model_key_name}")
        return processor, model
    except Exception as e:
        print(f"ERROR loading {model_key_name} from Hugging Face: {e}\n{traceback.format_exc()}")
        return None, None

@st.cache_resource
def load_beit3_local_model_cached(): # BEiT3 VQA
    print("INFO: Loading BEiT3 (Local VQA) model...")
    label_file = "label2answer.json"; spm_file = "beit3.spm"; checkpoint_file = "best.pth"
    if not os.path.exists(label_file) or not os.path.exists(checkpoint_file):
        missing_files_msg = []
        if not os.path.exists(label_file): missing_files_msg.append(label_file)
        if not os.path.exists(checkpoint_file): missing_files_msg.append(checkpoint_file)
        print(f"ERROR: Thiếu tệp cho BEiT3 (Local VQA): {', '.join(missing_files_msg)}.")
        return None, None, None
    try:
        with open(label_file, "r", encoding="utf-8") as f: label2answer = json.load(f)
        processor = Beit3Processing(sentencepiece_model=spm_file)
        num_classes = len(label2answer)
        model = create_model("beit3_base_patch16_480_vqav2", pretrained=False, drop_path_rate=0.1, vocab_size=64010, num_classes=num_classes)
    except Exception as e:
        print(f"ERROR: Lỗi khởi tạo cấu trúc model BEiT3 (Local VQA): {e}")
        return None, None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(checkpoint_file, map_location=device)
        state_dict_to_load = checkpoint.get("model", checkpoint.get("module", checkpoint))
        if state_dict_to_load is None: state_dict_to_load = checkpoint
        new_state_dict = OrderedDict((k[7:] if k.startswith('module.') else k, v) for k, v in state_dict_to_load.items())
        m_keys, u_keys = model.load_state_dict(new_state_dict, strict=False)
        if m_keys: print(f"WARNING (BEiT3 Local): Thiếu keys khi load checkpoint: {m_keys}")
        if u_keys: print(f"WARNING (BEiT3 Local): Keys không mong muốn trong checkpoint: {u_keys}")
        print(f"INFO: Đã tải checkpoint {checkpoint_file} cho BEiT3 (Local VQA).")
    except Exception as e:
        print(f"ERROR: Lỗi tải checkpoint {checkpoint_file} cho BEiT3 (Local VQA): {e}")
        print("WARNING: BEiT3 (Local VQA) sẽ sử dụng trọng số ngẫu nhiên.")
    model.to(device); model.eval()
    return processor, model, label2answer

# --- Hàm tiện ích STT (Local), TTS (API), Dịch (API) ---

def recognize_speech_local_whisper(audio_bytes, whisper_model_instance, lang_code_whisper="auto"):
    # (Giữ nguyên hàm này từ phiên bản trước)
    if not whisper_available or not whisper_model_instance:
        return None, "Model STT Whisper không khả dụng."
    if not audio_bytes:
        return None, "Không có dữ liệu âm thanh."
    tmp_audio_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio_file:
            tmp_audio_file.write(audio_bytes)
            tmp_audio_path = tmp_audio_file.name
        audio_np = whisper.load_audio(tmp_audio_path)
        options = {"fp16": torch.cuda.is_available()}
        if lang_code_whisper and lang_code_whisper != "auto":
            options["language"] = lang_code_whisper
        result = whisper_model_instance.transcribe(audio_np, **options)
        text = result["text"]
        return text, None
    except Exception as e:
        return None, f"Lỗi STT (Whisper): {e}"
    finally:
        if tmp_audio_path and os.path.exists(tmp_audio_path):
            os.unlink(tmp_audio_path)

def text_to_speech_api_gtts(text, lang_code_gtts): # Quay lại dùng gTTS
    if not gtts_available: return None, "Thư viện gTTS không khả dụng."
    if not text: return None, "Không có văn bản cho TTS."
    try:
        tts = gTTS(text=text, lang=lang_code_gtts, slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.read(), None
    except Exception as e:
        return None, f"Lỗi TTS (gTTS API): {e}"

def translate_text_api(text, target_lang_code, source_lang_code="auto"):
    # (Giữ nguyên hàm này từ phiên bản trước)
    if not translator_available: return None, "Thư viện dịch không khả dụng."
    if not text: return None, "Không có văn bản để dịch."
    try:
        translator = Translator(to_lang=target_lang_code, from_lang=source_lang_code)
        translated_text = translator.translate(text)
        return translated_text, None
    except Exception as e:
        return None, f"Lỗi dịch (API): {e}"

def speak_message_api_tts(message_text, lang_code_gtts, force_speak=False): # Sử dụng TTS API
    if 'last_spoken_ui_message_api' not in st.session_state: st.session_state.last_spoken_ui_message_api = ""
    if not gtts_available: return False

    if message_text and (message_text != st.session_state.last_spoken_ui_message_api or force_speak):
        audio_bytes, error = text_to_speech_api_gtts(message_text, lang_code_gtts)
        if error:
            st.warning(f"Không thể phát thông báo (TTS API): {error}")
            st.session_state.last_spoken_ui_message_api = ""
            return False
        if audio_bytes:
            st.session_state.audio_to_play_next = audio_bytes
            st.session_state.last_spoken_ui_message_api = message_text
            return True
    return False

# --- Hàm VQA (get_vqa_response - giữ nguyên như phiên bản trước) ---
# (Copy hàm get_vqa_response vào đây)
def get_vqa_response(image_pil, question_text_en, selected_model_key):
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    answer = None; error_msg = None
    try:
        if selected_model_key.startswith('Gemini'):
            if not gemini_available: raise NotImplementedError("GeminiVQA class not available.")
            _, gemini_instance, _ = models_vqa[selected_model_key]
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                image_pil.save(tmp, format="PNG"); tmp_path = tmp.name
            try:
                if "zeroshot" in selected_model_key:
                    answer = gemini_instance.ask_zeroshot(image_file=tmp_path, question=question_text_en)
                else:
                    answer = gemini_instance.ask_fewshot(image_file=tmp_path, question=question_text_en)
            finally: os.unlink(tmp_path)
        elif selected_model_key == 'BEiT3 (Local VQA)':
            proc, model, lbl2ans = load_beit3_local_model_cached()
            if not all([proc, model, lbl2ans]): raise ValueError("Lỗi tải model BEiT3 local.")
            data = proc(image_pil, question_text_en) # Beit3Processing được gọi ở đây
            img_t = data["image"].to(device)
            lang_t = data["language_tokens"].to(device)
            pad_m = data["padding_mask"].to(device)
            with torch.no_grad(): logits = model(image=img_t, question=lang_t, padding_mask=pad_m)
            idx = logits.argmax(-1).item(); answer = lbl2ans.get(str(idx), f"BEiT3: ID không tìm thấy: {idx}")
        else: # HF Models
            proc, model = load_hf_model_and_processor_cached(selected_model_key)
            if not proc or not model: raise ValueError(f"Lỗi tải model HF {selected_model_key}.")
            model.to(device); model.eval()
            if "Image Class." in selected_model_key:
                inputs = proc(images=image_pil, return_tensors="pt").to(device)
            else:
                inputs = proc(image_pil, question_text_en, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad(): outputs = model(**inputs)
            logits = outputs.logits; idx = logits.argmax(-1).item()
            if hasattr(model.config, 'id2label') and model.config.id2label:
                answer = model.config.id2label.get(idx, f"HF: ID không rõ: {idx}")
            else: answer = f"HF: Class index: {idx}"
    except NotImplementedError as e: error_msg = f"Lỗi triển khai VQA: {e}"
    except ValueError as e: error_msg = str(e)
    except Exception as e: error_msg = f"Lỗi chung VQA ({selected_model_key}): {e}\n{traceback.format_exc()}"
    time.sleep(1)
    end = time.time()

    print(f"Total runtime of the program is {end - start} seconds") 
    return answer, error_msg

# --- Hàm Run chính của Streamlit ---
def run():
    for warning_msg in startup_warnings:
        st.warning(warning_msg)

    st.title("VQA (Local STT, API TTS/Dịch)")

    # --- Tải model STT ---
    if 'stt_model' not in st.session_state:
        st.session_state.stt_model = None
        if whisper_available:
            with st.spinner(f"Đang tải model STT Whisper ({WHISPER_MODEL_NAME})..."):
                st.session_state.stt_model = load_stt_whisper_model_cached(WHISPER_MODEL_NAME)
            if st.session_state.stt_model: st.success(f"Model STT Whisper '{WHISPER_MODEL_NAME}' đã sẵn sàng.")
            else: st.error("Không thể tải model STT Whisper.")
    
    # TTS không cần tải model nếu dùng gTTS

    # --- Khởi tạo Session State ---
    default_lang_name = list(SUPPORTED_LANGUAGES.keys())[0]
    if 'user_language_name' not in st.session_state: st.session_state.user_language_name = default_lang_name
    # ... (các state khác như current_pil_image, processed_file_identifier, etc.)
    if 'current_pil_image' not in st.session_state: st.session_state.current_pil_image = None
    if 'processed_file_identifier' not in st.session_state: st.session_state.processed_file_identifier = None
    if 'recognized_question_original' not in st.session_state: st.session_state.recognized_question_original = ""
    if 'question_for_model_en' not in st.session_state: st.session_state.question_for_model_en = ""
    if 'vqa_answer_en' not in st.session_state: st.session_state.vqa_answer_en = ""
    if 'vqa_answer_original_lang' not in st.session_state: st.session_state.vqa_answer_original_lang = ""
    if 'status_message_ui' not in st.session_state: st.session_state.status_message_ui = ""
    if 'error_message_ui' not in st.session_state: st.session_state.error_message_ui = ""
    if 'audio_to_play_next' not in st.session_state: st.session_state.audio_to_play_next = None
    if 'last_stt_trigger_time' not in st.session_state: st.session_state.last_stt_trigger_time = 0


    # --- Sidebar ---
    st.sidebar.header("Cài đặt")
    prev_lang_name = st.session_state.user_language_name
    st.session_state.user_language_name = st.sidebar.selectbox(
        "Ngôn ngữ của bạn:",
        options=list(SUPPORTED_LANGUAGES.keys()),
        index=list(SUPPORTED_LANGUAGES.keys()).index(st.session_state.user_language_name),
        key="user_lang_sb_main"
    )
    if st.session_state.user_language_name != prev_lang_name:
        st.session_state.recognized_question_original = "" # Reset
        st.session_state.question_for_model_en = ""
        st.session_state.vqa_answer_en = ""
        st.session_state.vqa_answer_original_lang = ""
        st.session_state.status_message_ui = f"Ngôn ngữ đã đổi thành {st.session_state.user_language_name}."
        # speak_message_api_tts(st.session_state.status_message_ui, SUPPORTED_LANGUAGES[st.session_state.user_language_name][1], force_speak=True)
        st.rerun()

    current_lang_codes = SUPPORTED_LANGUAGES[st.session_state.user_language_name]
    whisper_code = current_lang_codes[0]
    translate_gtts_code = current_lang_codes[1] # Dùng chung cho dịch và gTTS

    selected_vqa_model_key = st.sidebar.selectbox("Chọn Model VQA:", list(models_vqa.keys()), key="vqa_model_sb_main")

    if st.sidebar.button("🎤 Hướng dẫn (Nghe)", key="help_btn_main"):
        help_text = "Chào mừng! Chọn ngôn ngữ, tải ảnh, rồi nhấn nút micro để hỏi."
        st.session_state.status_message_ui = help_text
        speak_message_api_tts(help_text, translate_gtts_code, force_speak=True)

    # --- Giao diện chính ---
    # ... (Phần tải ảnh giống như code trước, dùng processed_file_identifier)
    st.header("1. Tải ảnh lên")
    uploaded_file = st.file_uploader("Chọn một ảnh", type=["jpg", "jpeg", "png"], key="img_upload_key_main")
    new_image_processed = False
    if uploaded_file is not None:
        file_id = (uploaded_file.name, uploaded_file.size)
        if st.session_state.processed_file_identifier != file_id:
            try:
                st.session_state.current_pil_image = Image.open(uploaded_file).convert("RGB")
                st.session_state.processed_file_identifier = file_id
                new_image_processed = True
                st.session_state.status_message_ui = "Ảnh mới đã được tải lên."
            except Exception as e:
                st.session_state.current_pil_image = None; st.session_state.processed_file_identifier = None
                st.session_state.error_message_ui = f"Lỗi mở ảnh: {e}"
    if new_image_processed:
        st.session_state.recognized_question_original = "" # Reset
        st.session_state.question_for_model_en = ""
        st.session_state.vqa_answer_en = ""
        st.session_state.vqa_answer_original_lang = ""
        st.session_state.error_message_ui = ""
    if st.session_state.current_pil_image:
        st.image(st.session_state.current_pil_image, caption="Ảnh đã tải", use_column_width=True)


    st.header("2. Đặt câu hỏi bằng giọng nói")
    if not st.session_state.current_pil_image:
        st.info("Vui lòng tải ảnh lên ở Bước 1.")
    elif not audiorec_available:
        st.error("Chức năng ghi âm không khả dụng (st_audiorec chưa được cài đặt).")
    elif not st.session_state.stt_model:
        st.error("Model STT (Whisper) chưa được tải hoặc không khả dụng.")
    else:
        st.write(f"Nhấn micro và nói câu hỏi bằng **{st.session_state.user_language_name}**.")
        audio_data_bytes = st_audiorec()

        if audio_data_bytes:
            current_time = time.time()
            if current_time - st.session_state.last_stt_trigger_time > 1.5:
                st.session_state.last_stt_trigger_time = current_time
                st.session_state.status_message_ui = "Đang nhận dạng giọng nói (Whisper)..."
                st.session_state.error_message_ui = ""
                # speak_message_api_tts(st.session_state.status_message_ui, translate_gtts_code)
                
                with st.spinner(st.session_state.status_message_ui):
                    q_original, err_stt = recognize_speech_local_whisper(audio_data_bytes, st.session_state.stt_model, whisper_code)
                
                if err_stt:
                    st.session_state.error_message_ui = f"Lỗi STT: {err_stt}"
                    st.session_state.recognized_question_original = ""
                elif not q_original:
                    st.session_state.error_message_ui = "Không nhận dạng được nội dung."
                    st.session_state.recognized_question_original = ""
                else:
                    st.session_state.recognized_question_original = q_original
                    st.session_state.status_message_ui = f"Đã nhận dạng: \"{q_original}\""
                    # speak_message_api_tts(st.session_state.status_message_ui, translate_gtts_code)

                    # Dịch sang tiếng Anh (API)
                    q_for_model = q_original
                    if translate_gtts_code != "en":
                        if not translator_available:
                            st.session_state.error_message_ui = "Thư viện dịch chưa cài đặt."
                            q_for_model = ""
                        else:
                            st.session_state.status_message_ui = "Đang dịch câu hỏi sang Tiếng Anh (API)..."
                            with st.spinner(st.session_state.status_message_ui):
                                q_en_translated, err_trans_q = translate_text_api(q_original, "en", translate_gtts_code)
                            if err_trans_q:
                                st.session_state.error_message_ui = f"Lỗi dịch câu hỏi: {err_trans_q}"
                                q_for_model = ""
                            else: q_for_model = q_en_translated
                    st.session_state.question_for_model_en = q_for_model
                    
                    if st.session_state.question_for_model_en:
                        st.session_state.status_message_ui = f"Model VQA ({selected_vqa_model_key}) đang xử lý..."
                        with st.spinner(st.session_state.status_message_ui):
                            ans_en, err_vqa = get_vqa_response(st.session_state.current_pil_image, st.session_state.question_for_model_en, selected_vqa_model_key)
                        
                        if err_vqa:
                            st.session_state.error_message_ui = f"Lỗi VQA: {err_vqa}"
                            st.session_state.vqa_answer_en = ""; st.session_state.vqa_answer_original_lang = ""
                        else:
                            st.session_state.vqa_answer_en = ans_en
                            ans_final = ans_en
                            if translate_gtts_code != "en":
                                if not translator_available:
                                    st.session_state.error_message_ui = "Thư viện dịch chưa cài đặt."
                                    ans_final = f"{ans_en} (Không thể dịch)"
                                else:
                                    st.session_state.status_message_ui = f"Đang dịch câu trả lời về {st.session_state.user_language_name} (API)..."
                                    with st.spinner(st.session_state.status_message_ui):
                                        ans_translated_orig, err_trans_a = translate_text_api(ans_en, translate_gtts_code, "en")
                                        pre_ans, err_trans_pre = translate_text_api("Answer", translate_gtts_code, "en")
                                    if err_trans_a or err_trans_pre:
                                        st.session_state.error_message_ui = f"Lỗi dịch câu trả lời: {err_trans_a}"
                                        ans_final = f"{ans_en} (Lỗi dịch)"
                                    else: ans_final = ans_translated_orig
                            st.session_state.vqa_answer_original_lang = ans_final
                            if translate_gtts_code == "en":
                                st.session_state.status_message_ui = f"Answer: \"{ans_final}\""
                            else:
                                st.session_state.status_message_ui = f"\"{pre_ans}\": \"{ans_final}\""
                            if gtts_available: # Chỉ nói nếu gTTS có sẵn
                                speak_message_api_tts(st.session_state.status_message_ui, translate_gtts_code, force_speak=True)
                    elif not st.session_state.error_message_ui:
                        st.session_state.status_message_ui = "Không có câu hỏi hợp lệ để xử lý VQA."

    # --- Hiển thị kết quả ---
    st.header("3. Kết quả")
    if st.session_state.error_message_ui: st.error(st.session_state.error_message_ui)
    elif st.session_state.status_message_ui:
        if any(keyword in st.session_state.status_message_ui for keyword in ["Câu trả lời:", "Đã nhận dạng:", "Ảnh mới", "Ngôn ngữ đã đổi"]):
            st.info(st.session_state.status_message_ui)

    if st.session_state.recognized_question_original:
        st.write(f"**Câu hỏi của bạn ({st.session_state.user_language_name}):** {st.session_state.recognized_question_original}")
        if st.session_state.question_for_model_en and translate_gtts_code != "en":
             st.caption(f"(Đã dịch sang Tiếng Anh cho model: {st.session_state.question_for_model_en})")
    
    if st.session_state.vqa_answer_original_lang:
        st.success(f"**Câu trả lời ({st.session_state.user_language_name}):** {st.session_state.vqa_answer_original_lang}")
        if st.session_state.vqa_answer_en and translate_gtts_code != "en" and "(Lỗi dịch" not in st.session_state.vqa_answer_original_lang:
            st.caption(f"(Câu trả lời gốc Tiếng Anh từ model: {st.session_state.vqa_answer_en})")
        
        if gtts_available and st.button(f"🔊 Nghe lại câu trả lời", key="replay_btn_main"):
            speak_message_api_tts(st.session_state.vqa_answer_original_lang, translate_gtts_code, force_speak=True)

    if st.session_state.get('audio_to_play_next') and gtts_available:
        st.audio(st.session_state.audio_to_play_next, format="audio/mp3", autoplay=True) # gTTS ra mp3
        st.session_state.audio_to_play_next = None

    st.sidebar.markdown("---")
    st.sidebar.caption("VQA: Local STT, API TTS/Dịch")

# --- Điểm vào chính ---
if __name__ == "__main__":
    # Tạo file giả lập cho BEiT3 nếu chưa có
    # ... (Giữ nguyên phần tạo file giả label2answer.json, beit3.spm)
    if not os.path.exists("label2answer.json"):
        with open("label2answer.json", "w", encoding="utf-8") as f: json.dump({str(i): f"mock_ans_{i}" for i in range(100)}, f)
    if not os.path.exists("beit3.spm"):
        with open("beit3.spm", "w") as f: f.write("mock spm")
    
    if gemini_available and (GEMINI_API_KEY == "YOUR_API_KEY" or not GEMINI_API_KEY):
        startup_warnings.append("GEMINI API KEY: Vui lòng đặt GEMINI_API_KEY nếu muốn sử dụng model Gemini.")

    run()