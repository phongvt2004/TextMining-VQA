# app.py

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
import numpy as np
import threading

# --- Các thành phần WebRTC và VAD ---
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase, RTCConfiguration
    import librosa # Cho resampling
    import webrtcvad # Cho Voice Activity Detection
    webrtc_components_available = True
except ImportError:
    st.error("Lỗi: Cần cài đặt streamlit-webrtc, aiortc, librosa, webrtcvad. Chạy: pip install streamlit-webrtc aiortc librosa webrtcvad")
    webrtc_components_available = False

# --- Các thành phần khác (STT, TTS, Dịch, Ghi âm thủ công, VQA modules) ---
startup_warnings = []
# (Copy các khối try-except import cho CustomViltForVQA, Beit3Processing, GeminiVQA,
#  whisper, gTTS, Translator, st_audiorec từ code trước vào đây)
# Ví dụ:
try:
    from modules.model import CustomViltForVQA
except ImportError:
    startup_warnings.append("MODULES: CustomViltForVQA không tìm thấy...")
    CustomViltForVQA = ViltForQuestionAnswering
# ... (các import khác tương tự) ...
try:
    import whisper
    whisper_available = True
except ImportError:
    startup_warnings.append("STT WHISPER: Lỗi import 'whisper'...")
    whisper_available = False
try:
    from gtts import gTTS
    gtts_available = True
except ImportError:
    startup_warnings.append("TTS GTTS: Lỗi import 'gTTS'...")
    gtts_available = False
try:
    from translate import Translator
    translator_available = True
except ImportError:
    startup_warnings.append("TRANSLATE: Lỗi import 'Translator'...")
    translator_available = False
try:
    from st_audiorec import st_audiorec
    audiorec_available = True
except ImportError:
    startup_warnings.append("AUDIOREC: Lỗi import 'st_audiorec'...")
    audiorec_available = False
    def st_audiorec(): st.error("st_audiorec chưa cài đặt."); return None
try:
    from gemini_Calls import GeminiVQA
    gemini_available = True
except ImportError:
    startup_warnings.append("MODULES: GeminiVQA không tìm thấy...")
    gemini_available = False
    class GeminiVQA: # Hàm giả
        def __init__(self, api_key): self.api_key = api_key
        def ask_zeroshot(self, image_file, question): raise NotImplementedError("GeminiVQA (zeroshot) không khả dụng.")
        def ask_fewshot(self, image_file, question): raise NotImplementedError("GeminiVQA (fewshot) không khả dụng.")

try:
    from modules.beit_3 import Beit3Processing
except ImportError:
    startup_warnings.append("MODULES: Beit3Processing không tìm thấy...")
    class Beit3Processing: # Hàm giả
        def __init__(self, *args, **kwargs): print("WARNING: Beit3Processing (dummy) initialized.")
        def __call__(self, image, text, **kwargs):
            print("WARNING: Beit3Processing (dummy) called.")
            return {"image": torch.randn(1, 3, 480, 480), "language_tokens": torch.randint(0, 100, (1, 40)), "padding_mask": torch.zeros(1, 40, dtype=torch.long)}


# --- Cấu hình chung ---
st.set_page_config(layout="centered", page_title="VQA Tự Động Nghe")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "") # Nên dùng biến môi trường
WHISPER_MODEL_NAME = "base"
TARGET_SAMPLE_RATE_STT = 16000 # Cho Whisper và VAD

SUPPORTED_LANGUAGES = {
    "Tiếng Việt": ("vi", "vi"), "English": ("en", "en"), "Español (Spanish)": ("es", "es"),
    "Français (French)": ("fr", "fr"), "日本語 (Japanese)": ("ja", "ja"), "한국어 (Korean)": ("ko", "ko"),
}
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# --- Định nghĩa Model VQA ---
models_vqa = OrderedDict()
# (Copy định nghĩa models_vqa từ code trước)
models_vqa["BeiT3-base (Image Class.)"] = (BeitImageProcessor, BeitForImageClassification, "microsoft/beit-base-patch16-224")
models_vqa["ViLT-base (VQA)"] = (AutoProcessor, ViltForQuestionAnswering, "dandelin/vilt-b32-finetuned-vqa")
models_vqa["ViLT (Custom VQA)"] = (AutoProcessor, CustomViltForVQA, "phonghoccode/vilt-vqa-finetune-pytorch")
models_vqa["BEiT3 (Local VQA)"] = "beit3_local_vqa_placeholder"
if gemini_available and GEMINI_API_KEY:
    models_vqa["Gemini_zeroshot (API)"] = ("Gemini", GeminiVQA(api_key=GEMINI_API_KEY), "Gemini")
    models_vqa["Gemini_fewshot (API)"] = ("Gemini", GeminiVQA(api_key=GEMINI_API_KEY), "Gemini")


# --- Lớp VADAudioProcessor ---
if webrtc_components_available:
    class VADAudioProcessor(AudioProcessorBase):
        def __init__(self, target_sample_rate=TARGET_SAMPLE_RATE_STT, vad_frame_ms=30, vad_aggressiveness=2,
                     silence_to_end_speech_ms=700, speech_to_start_ms=150):
            self.target_sample_rate = target_sample_rate
            self.vad_frame_ms = vad_frame_ms
            self.vad = webrtcvad.Vad(vad_aggressiveness)
            self.vad_bytes_per_frame = int(target_sample_rate * (vad_frame_ms / 1000.0) * 2) # 16-bit mono
            self.silence_threshold_frames = silence_to_end_speech_ms // vad_frame_ms
            self.speaking_threshold_frames = speech_to_start_ms // vad_frame_ms

            self.audio_buffer = bytearray()
            self.is_speaking = False
            self.silence_frames_count = 0
            self.speaking_frames_count = 0
            self.frames_lock = threading.Lock()
            print(f"VAD Processor Initialized: SR={target_sample_rate}, Aggressiveness={vad_aggressiveness}")

        def _preprocess_frame(self, frame_data_np, input_sample_rate):
            if frame_data_np.ndim > 1: frame_data_np = np.mean(frame_data_np, axis=1) # Stereo to Mono
            if input_sample_rate != self.target_sample_rate:
                frame_data_np = librosa.resample(frame_data_np.astype(np.float32),
                                                 orig_sr=input_sample_rate, target_sr=self.target_sample_rate)
            if frame_data_np.dtype != np.int16:
                if np.issubdtype(frame_data_np.dtype, np.floating):
                    frame_data_np = np.clip(frame_data_np, -1.0, 1.0) * 32767
                frame_data_np = frame_data_np.astype(np.int16)
            return frame_data_np.tobytes()

        def recv(self, frame):
            try:
                processed_bytes = self._preprocess_frame(frame.to_ndarray(), frame.sample_rate)
                num_vad_frames = len(processed_bytes) // self.vad_bytes_per_frame

                with self.frames_lock:
                    for i in range(num_vad_frames):
                        vad_frame_chunk = processed_bytes[i*self.vad_bytes_per_frame : (i+1)*self.vad_bytes_per_frame]
                        if len(vad_frame_chunk) != self.vad_bytes_per_frame: continue
                        
                        try: is_speech = self.vad.is_speech(vad_frame_chunk, self.target_sample_rate)
                        except Exception: is_speech = False # Lỗi VAD, coi như im lặng

                        if not self.is_speaking:
                            if is_speech:
                                self.speaking_frames_count += 1
                                if self.speaking_frames_count >= self.speaking_threshold_frames:
                                    self.is_speaking = True
                                    self.audio_buffer.extend(vad_frame_chunk)
                                    self.silence_frames_count = 0
                                    self.speaking_frames_count = 0
                                    print("VAD: Speech started")
                            else: self.speaking_frames_count = 0
                        else: # self.is_speaking is True
                            self.audio_buffer.extend(vad_frame_chunk)
                            if not is_speech:
                                self.silence_frames_count += 1
                                if self.silence_frames_count >= self.silence_threshold_frames:
                                    print(f"VAD: Speech ended. Buffer: {len(self.audio_buffer)} bytes")
                                    self.is_speaking = False
                                    # Chỉ đặt vào session_state nếu buffer có nội dung đáng kể
                                    if len(self.audio_buffer) > self.vad_bytes_per_frame * self.speaking_threshold_frames * 2: # Heuristic
                                        st.session_state.vad_detected_audio_bytes = bytes(self.audio_buffer) # Copy
                                        st.session_state.vad_triggered_stt = True
                                    self.audio_buffer.clear()
                                    self.silence_frames_count = 0
                                    # Không break, để tiếp tục nhận frame cho câu nói tiếp theo
                            else: self.silence_frames_count = 0
            except Exception as e:
                print(f"VADAudioProcessor recv error: {e}")
            return frame # Luôn trả về frame
else: # webrtc_components_available is False
    VADAudioProcessor = None # Để tránh lỗi nếu không import được


# --- Các hàm tải và cache model (STT, VQA) ---
# (Copy load_stt_whisper, load_vqa_hf_model, load_vqa_beit3_local từ code trước,
#  đảm bảo chúng dùng @st.cache_resource và print log thay vì st.info/error)
# Ví dụ đã có trong file bạn cung cấp trước đó.
@st.cache_resource
def load_stt_whisper(model_name=WHISPER_MODEL_NAME):
    if not whisper_available: return None
    print(f"INFO: Đang tải model STT Whisper '{model_name}'...")
    try:
        model = whisper.load_model(model_name)
        print(f"INFO: Model STT Whisper '{model_name}' đã tải xong.")
        return model
    except Exception as e:
        print(f"ERROR: Lỗi tải model Whisper '{model_name}': {e}.")
        return None

@st.cache_resource
def load_vqa_hf_model(model_key_name): # model_key_name từ models_vqa
    if model_key_name not in models_vqa or models_vqa[model_key_name][0] == "Gemini" or models_vqa[model_key_name] == "beit3_local_vqa_placeholder":
        return None, None
    print(f"INFO: Đang tải model VQA HF '{model_key_name}'...")
    processor_class, model_class, model_hf_name = models_vqa[model_key_name]
    try:
        processor = processor_class.from_pretrained(model_hf_name)
        model = model_class.from_pretrained(model_hf_name)
        print(f"INFO: Model VQA HF '{model_key_name}' đã tải xong.")
        return processor, model
    except Exception as e:
        print(f"ERROR: Lỗi tải model VQA HF '{model_key_name}': {e}")
        return None, None

@st.cache_resource
def load_vqa_beit3_local():
    print("INFO: Đang tải model BEiT3 (Local VQA)...")
    label_file = "label2answer.json"; spm_file = "beit3.spm"; checkpoint_file = "best.pth"
    if not os.path.exists(label_file) or not os.path.exists(checkpoint_file):
        missing = [f for f in [label_file, checkpoint_file] if not os.path.exists(f)]
        print(f"ERROR: Thiếu tệp cho BEiT3 (Local VQA): {', '.join(missing)}.")
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

# --- Hàm tiện ích STT, TTS, Dịch (Tương tự code trước) ---
def stt_whisper(audio_bytes, whisper_model_instance, lang_code_whisper="auto"):
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
        text = result.get("text", "")
        return text, None
    except Exception as e:
        return None, f"Lỗi STT (Whisper): {e}"
    finally:
        if tmp_audio_path and os.path.exists(tmp_audio_path):
            os.unlink(tmp_audio_path)

def tts_gtts(text, lang_code_gtts):
    # (Giữ nguyên hàm này từ phiên bản trước)
    if not gtts_available: return None, "Thư viện gTTS không khả dụng."
    if not text: return None, "Không có văn bản cho TTS."
    try:
        tts = gTTS(text=text, lang=lang_code_gtts, slow=False)
        fp = io.BytesIO(); tts.write_to_fp(fp); fp.seek(0)
        return fp.read(), None
    except Exception as e:
        return None, f"Lỗi TTS (gTTS API): {e}"

def translate_api(text, target_lang_code, source_lang_code="auto"):
    # (Giữ nguyên hàm này từ phiên bản trước)
    if not translator_available: return None, "Thư viện dịch không khả dụng."
    if not text: return None, "Không có văn bản để dịch."
    try:
        translator = Translator(to_lang=target_lang_code, from_lang=source_lang_code)
        translated_text = translator.translate(text)
        if translated_text is None:
             return None, f"Không thể dịch văn bản."
        return translated_text, None
    except Exception as e:
        return None, f"Lỗi dịch (API): {e}"

def speak_ui_message(message_text, lang_code_gtts, force_speak=False):
    # (Giữ nguyên hàm này từ phiên bản trước)
    if 'last_spoken_ui_msg' not in st.session_state: st.session_state.last_spoken_ui_msg = ""
    if not gtts_available: return False
    if message_text and (message_text != st.session_state.last_spoken_ui_msg or force_speak):
        audio_bytes, error = tts_gtts(message_text, lang_code_gtts)
        if error:
            st.warning(f"Lỗi TTS khi nói: '{message_text}': {error}")
            st.session_state.last_spoken_ui_msg = ""
            return False
        if audio_bytes:
            st.session_state.audio_bytes_to_play = audio_bytes
            st.session_state.last_spoken_ui_msg = message_text
            return True
    return False

# --- Hàm VQA (Tương tự code trước, đã copy vào đây) ---
def get_vqa_response(image_pil, question_text_en, selected_model_key_vqa):
    # (Giữ nguyên hàm get_vqa_response từ phiên bản trước)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    answer = None; error_msg = None
    try:
        if selected_model_key_vqa.startswith('Gemini'):
            if not gemini_available: raise NotImplementedError("GeminiVQA class không khả dụng.")
            _, gemini_instance, _ = models_vqa[selected_model_key_vqa]
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                image_pil.save(tmp, format="PNG"); tmp_path = tmp.name
            try:
                if "zeroshot" in selected_model_key_vqa:
                    answer = gemini_instance.ask_zeroshot(image_file=tmp_path, question=question_text_en)
                else:
                    answer = gemini_instance.ask_fewshot(image_file=tmp_path, question=question_text_en)
            finally: os.unlink(tmp_path)
        elif selected_model_key_vqa == 'BEiT3 (Local VQA)':
            proc, model, lbl2ans = load_vqa_beit3_local()
            if not all([proc, model, lbl2ans]): raise ValueError("Lỗi tải model BEiT3 local cho VQA.")
            data = proc(image_pil, question_text_en)
            img_t, lang_t, pad_m = data["image"].to(device), data["language_tokens"].to(device), data["padding_mask"].to(device)
            with torch.no_grad(): logits = model(image=img_t, question=lang_t, padding_mask=pad_m)
            idx = logits.argmax(-1).item(); answer = lbl2ans.get(str(idx), f"BEiT3: ID '{idx}' không tìm thấy trong label map.")
        else: # Các model HF khác
            proc, model = load_vqa_hf_model(selected_model_key_vqa)
            if not proc or not model: raise ValueError(f"Lỗi tải model HF VQA '{selected_model_key_vqa}'.")
            model.to(device); model.eval()
            if "Image Class." in selected_model_key_vqa:
                inputs = proc(images=image_pil, return_tensors="pt").to(device)
            else: # ViLT models
                inputs = proc(image_pil, question_text_en, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad(): outputs = model(**inputs)
            logits = outputs.logits; idx = logits.argmax(-1).item()
            if hasattr(model.config, 'id2label') and model.config.id2label:
                answer = model.config.id2label.get(idx, f"HF: ID '{idx}' không có trong id2label.")
            else: answer = f"HF: Class index: {idx} (không có id2label)."
    except NotImplementedError as e: error_msg = f"Lỗi triển khai VQA: {e}"
    except ValueError as e: error_msg = str(e)
    except Exception as e: error_msg = f"Lỗi chung trong VQA ({selected_model_key_vqa}): {e}\n{traceback.format_exc()}"
    return answer, error_msg


# --- Hàm Run chính của Streamlit ---
def run():
    # ... (Hiển thị startup_warnings) ...
    st.title("VQA Tự Động Nghe (Local STT, API TTS/Dịch)")

    # --- Tải model STT ---
    if 'stt_model' not in st.session_state:
        st.session_state.stt_model = None
        if whisper_available:
            with st.spinner(f"Đang tải model STT Whisper ({WHISPER_MODEL_NAME})..."):
                st.session_state.stt_model = load_stt_whisper(WHISPER_MODEL_NAME)
            if st.session_state.stt_model: st.success(f"Model STT Whisper ({WHISPER_MODEL_NAME}) đã sẵn sàng.")
            else: st.error("Không thể tải model STT Whisper.")
    
    # --- Khởi tạo Session State ---
    default_lang_name = list(SUPPORTED_LANGUAGES.keys())[0]
    if 'user_lang_name' not in st.session_state: st.session_state.user_lang_name = default_lang_name
    if 'current_pil_image' not in st.session_state: st.session_state.current_pil_image = None
    if 'processed_file_id' not in st.session_state: st.session_state.processed_file_id = None
    if 'q_original_text' not in st.session_state: st.session_state.q_original_text = ""
    if 'q_english_text' not in st.session_state: st.session_state.q_english_text = ""
    if 'ans_english_text' not in st.session_state: st.session_state.ans_english_text = ""
    if 'ans_original_lang_text' not in st.session_state: st.session_state.ans_original_lang_text = ""
    if 'ui_status_msg' not in st.session_state: st.session_state.ui_status_msg = ""
    if 'ui_error_msg' not in st.session_state: st.session_state.ui_error_msg = ""
    if 'audio_bytes_to_play' not in st.session_state: st.session_state.audio_bytes_to_play = None
    if 'last_stt_proc_time' not in st.session_state: st.session_state.last_stt_proc_time = 0
    
    # States cho VAD - Đảm bảo khởi tạo là None
    if 'listening_mode' not in st.session_state: st.session_state.listening_mode = False
    # QUAN TRỌNG: Khởi tạo vad_processor_instance ở đây
    if 'vad_processor_instance' not in st.session_state: 
        st.session_state.vad_processor_instance = None
        print("DEBUG: Initialized st.session_state.vad_processor_instance to None")

    if 'vad_triggered_stt' not in st.session_state: st.session_state.vad_triggered_stt = False
    if 'vad_detected_audio_bytes' not in st.session_state: st.session_state.vad_detected_audio_bytes = None

    # --- Sidebar ---
    st.sidebar.header("Cài đặt")
    prev_user_lang_name = st.session_state.user_lang_name
    st.session_state.user_lang_name = st.sidebar.selectbox(
        "Ngôn ngữ của bạn:",
        options=list(SUPPORTED_LANGUAGES.keys()),
        index=list(SUPPORTED_LANGUAGES.keys()).index(st.session_state.user_lang_name),
        key="user_language_selectbox_run" 
    )
    if st.session_state.user_lang_name != prev_user_lang_name:
        st.session_state.q_original_text = ""; st.session_state.q_english_text = ""
        st.session_state.ans_english_text = ""; st.session_state.ans_original_lang_text = ""
        st.session_state.ui_status_msg = f"Ngôn ngữ đã đổi thành {st.session_state.user_lang_name}."
        st.session_state.ui_error_msg = ""
        st.rerun()

    current_lang_config = SUPPORTED_LANGUAGES[st.session_state.user_lang_name]
    w_code = current_lang_config[0]
    t_g_code = current_lang_config[1]

    selected_vqa_key = st.sidebar.selectbox("Chọn Model VQA:", list(models_vqa.keys()), key="vqa_model_selectbox_run")

    if st.sidebar.button("🎤 Hướng dẫn sử dụng", key="help_button_sidebar_run"):
        help_text = "Chào mừng! Chọn ngôn ngữ, tải ảnh. Nếu bật chế độ tự động, hãy nói câu hỏi. Nếu không, dùng nút micro thủ công."
        st.session_state.ui_status_msg = help_text
        if gtts_available: speak_ui_message(help_text, t_g_code, force_speak=True)

    # Nút bật/tắt chế độ tự động lắng nghe trong sidebar
    if webrtc_components_available and VADAudioProcessor:
        new_listening_mode = st.sidebar.checkbox("👂 Bật chế độ tự động lắng nghe", 
                                                 key="toggle_vad_listening_cb_run", 
                                                 value=st.session_state.listening_mode)
        if new_listening_mode != st.session_state.listening_mode:
            st.session_state.listening_mode = new_listening_mode
            if st.session_state.listening_mode:
                st.info("Chế độ tự động lắng nghe ĐÃ BẬT.")
                # Sẽ tạo VAD processor ngay trước khi gọi webrtc_streamer nếu cần
            else:
                st.info("Chế độ tự động lắng nghe ĐÃ TẮT.")
                # Khi tắt, có thể cân nhắc việc "dừng" streamer nếu nó đang chạy,
                # hoặc giải phóng vad_processor_instance, nhưng streamlit-webrtc
                # thường tự quản lý vòng đời của component khi nó không còn được render.
                # st.session_state.vad_processor_instance = None # Cân nhắc
            st.rerun()

    # --- Giao diện chính ---
    st.header("1. Tải ảnh lên")
    # ... (Phần tải ảnh giữ nguyên) ...
    uploaded_img_file = st.file_uploader("Chọn một tệp ảnh", type=["jpg", "jpeg", "png"], key="image_uploader_key_run")
    new_image_was_processed = False
    if uploaded_img_file is not None:
        current_file_id = (uploaded_img_file.name, uploaded_img_file.size)
        if st.session_state.processed_file_id != current_file_id:
            try:
                st.session_state.current_pil_image = Image.open(uploaded_img_file).convert("RGB")
                st.session_state.processed_file_id = current_file_id
                new_image_was_processed = True
                st.session_state.ui_status_msg = "Ảnh mới đã được tải lên thành công."
            except Exception as e:
                st.session_state.current_pil_image = None; st.session_state.processed_file_id = None
                st.session_state.ui_error_msg = f"Lỗi khi mở ảnh: {e}"
    if new_image_was_processed:
        st.session_state.q_original_text = ""; st.session_state.q_english_text = ""
        st.session_state.ans_english_text = ""; st.session_state.ans_original_lang_text = ""
        st.session_state.ui_error_msg = ""
    if st.session_state.current_pil_image:
        st.image(st.session_state.current_pil_image, caption="Ảnh đã tải", use_column_width=True)

    st.header("2. Đặt câu hỏi")
    can_ask_question = st.session_state.current_pil_image and st.session_state.stt_model
    
    if not st.session_state.current_pil_image: st.info("Vui lòng tải ảnh lên ở Bước 1.")
    elif not st.session_state.stt_model: st.error("Model STT chưa sẵn sàng. Không thể nhận dạng giọng nói.")
    
    # Chế độ tự động lắng nghe
    if st.session_state.listening_mode:
        if not (webrtc_components_available and VADAudioProcessor):
            st.error("Chế độ tự động lắng nghe không khả dụng do thiếu thư viện.")
        elif can_ask_question:
            st.info(f"🎤 Đang ở chế độ tự động lắng nghe bằng **{st.session_state.user_lang_name}**... Hãy nói câu hỏi của bạn.")
            
            # Đảm bảo VAD processor được tạo TRƯỚC KHI streamer dùng nó
            if st.session_state.vad_processor_instance is None:
                print("DEBUG: Creating VADAudioProcessor instance FOR STREAMER.")
                st.session_state.vad_processor_instance = VADAudioProcessor()
            
            # Factory bây giờ chỉ đơn giản là trả về instance đã được quản lý trong session_state
            def vad_factory():
                # Thêm một lần kiểm tra nữa ở đây để cực kỳ an toàn,
                # mặc dù logic ở trên nên đã xử lý việc này.
                if st.session_state.vad_processor_instance is None:
                    print("CRITICAL DEBUG: vad_processor_instance was None INSIDE factory. Recreating.")
                    st.session_state.vad_processor_instance = VADAudioProcessor()
                return st.session_state.vad_processor_instance

            webrtc_ctx = webrtc_streamer(
                key="vad_auto_streamer_run", # Key mới
                mode=WebRtcMode.SENDRECV,
                audio_processor_factory=vad_factory, # SỬA Ở ĐÂY
                media_stream_constraints={"video": False, "audio": True, "echoCancellation": True},
                rtc_configuration=RTC_CONFIGURATION,
            )
            if not webrtc_ctx.state.playing:
                st.caption("Streamer VAD chưa hoạt động. Có thể cần tương tác để kích hoạt micro.")
            
    # Chế độ ghi âm thủ công
    elif audiorec_available and can_ask_question:
        st.write(f"Nhấn nút micro và nói câu hỏi bằng **{st.session_state.user_lang_name}**.")
        manual_audio_bytes = st_audiorec()
        if manual_audio_bytes:
            st.session_state.vad_detected_audio_bytes = manual_audio_bytes
            st.session_state.vad_triggered_stt = True
            st.session_state.last_stt_proc_time = 0 
            st.rerun() 
    elif not audiorec_available and can_ask_question:
        st.error("Không có phương thức ghi âm nào khả dụng.")


    # --- Xử lý audio được VAD hoặc ghi âm thủ công phát hiện ---
    # (Khối logic này giữ nguyên như phiên bản trước)
    if st.session_state.get('vad_triggered_stt') and st.session_state.get('vad_detected_audio_bytes'):
        audio_for_stt_processing = st.session_state.vad_detected_audio_bytes
        st.session_state.vad_triggered_stt = False 
        st.session_state.vad_detected_audio_bytes = None 
        current_proc_time = time.time()
        if current_proc_time - st.session_state.last_stt_proc_time > 0.5:
            st.session_state.last_stt_proc_time = current_proc_time
            st.session_state.ui_status_msg = "Đã phát hiện giọng nói, đang xử lý..."
            st.session_state.ui_error_msg = "" 
            with st.spinner(st.session_state.ui_status_msg):
                # 1. STT
                print(f"DEBUG: Processing STT for audio of length {len(audio_for_stt_processing)}")
                q_original, stt_err = stt_whisper(audio_for_stt_processing, st.session_state.stt_model, w_code)
                if stt_err: st.session_state.ui_error_msg = f"Lỗi STT: {stt_err}"
                elif not q_original or not q_original.strip(): st.session_state.ui_error_msg = "Không nhận dạng được nội dung giọng nói."
                else:
                    st.session_state.q_original_text = q_original.strip()
                    st.session_state.ui_status_msg = f"Đã nhận dạng: \"{st.session_state.q_original_text}\""
                    # 2. Dịch câu hỏi
                    q_for_vqa_model = st.session_state.q_original_text
                    if t_g_code != "en":
                        if not translator_available: st.session_state.ui_error_msg = "Lỗi: Thư viện dịch chưa cài đặt."; q_for_vqa_model = "" 
                        else:
                            q_en_trans, trans_q_err = translate_api(st.session_state.q_original_text, "en", t_g_code)
                            if trans_q_err: st.session_state.ui_error_msg = f"Lỗi dịch câu hỏi: {trans_q_err}"; q_for_vqa_model = ""
                            elif not q_en_trans: st.session_state.ui_error_msg = "Dịch câu hỏi không thành công."; q_for_vqa_model = ""
                            else: q_for_vqa_model = q_en_trans.strip()
                    st.session_state.q_english_text = q_for_vqa_model
                    # 3. VQA
                    if st.session_state.q_english_text:
                        ans_en, vqa_err = get_vqa_response(st.session_state.current_pil_image, st.session_state.q_english_text, selected_vqa_key)
                        if vqa_err: st.session_state.ui_error_msg = f"Lỗi VQA: {vqa_err}"
                        elif not ans_en: st.session_state.ui_error_msg = "Model VQA không trả về câu trả lời."
                        else:
                            st.session_state.ans_english_text = ans_en.strip()
                            final_ans_to_speak = st.session_state.ans_english_text
                            # 4. Dịch câu trả lời
                            if t_g_code != "en":
                                if not translator_available: st.session_state.ui_error_msg = "Lỗi: Thư viện dịch chưa cài đặt (cho câu trả lời)."; final_ans_to_speak = f"{st.session_state.ans_english_text} (Không thể dịch)"
                                else:
                                    ans_orig_trans, trans_a_err = translate_api(st.session_state.ans_english_text, t_g_code, "en")
                                    if trans_a_err: st.session_state.ui_error_msg = f"Lỗi dịch câu trả lời: {trans_a_err}"; final_ans_to_speak = f"{st.session_state.ans_english_text} (Lỗi dịch)"
                                    elif not ans_orig_trans: st.session_state.ui_error_msg = "Dịch câu trả lời không thành công."; final_ans_to_speak = f"{st.session_state.ans_english_text} (Lỗi dịch)"
                                    else: final_ans_to_speak = ans_orig_trans.strip()
                            st.session_state.ans_original_lang_text = final_ans_to_speak
                            ans_label_disp = "Answer"; 
                            if t_g_code != "en" and translator_available: lbl_trans, _ = translate_api("Answer", t_g_code, "en"); 
                            if lbl_trans: ans_label_disp = lbl_trans
                            st.session_state.ui_status_msg = f"{ans_label_disp}: \"{st.session_state.ans_original_lang_text}\""
                            if gtts_available: speak_ui_message(st.session_state.ans_original_lang_text, t_g_code, force_speak=True)
                    elif not st.session_state.ui_error_msg : st.session_state.ui_status_msg = "Không có câu hỏi hợp lệ để xử lý VQA."
            st.rerun()


    # --- Hiển thị kết quả và thông báo ---
    # ... (Phần hiển thị kết quả giữ nguyên) ...
    st.header("3. Kết quả và Thông báo")
    if st.session_state.ui_error_msg: st.error(st.session_state.ui_error_msg)
    elif st.session_state.ui_status_msg:
        if any(kwd in st.session_state.ui_status_msg for kwd in ["Ảnh mới", "Ngôn ngữ đã đổi", "Đã nhận dạng:", "Answer:", "Câu trả lời:", "Không có câu hỏi"]):
            st.info(st.session_state.ui_status_msg)

    if st.session_state.q_original_text:
        st.write(f"**Câu hỏi của bạn ({st.session_state.user_lang_name}):** {st.session_state.q_original_text}")
        if st.session_state.q_english_text and t_g_code != "en":
             st.caption(f"(Đã dịch sang Tiếng Anh: {st.session_state.q_english_text})")
    
    if st.session_state.ans_original_lang_text:
        st.success(f"**Câu trả lời ({st.session_state.user_lang_name}):** {st.session_state.ans_original_lang_text}")
        if st.session_state.ans_english_text and t_g_code != "en" and \
           "(Lỗi dịch" not in st.session_state.ans_original_lang_text and \
           "(Không thể dịch" not in st.session_state.ans_original_lang_text:
            st.caption(f"(Câu trả lời gốc Tiếng Anh: {st.session_state.ans_english_text})")
        
        if gtts_available and st.button(f"🔊 Nghe lại câu trả lời", key="replay_answer_button_run"):
            speak_ui_message(st.session_state.ans_original_lang_text, t_g_code, force_speak=True)

    if st.session_state.get('audio_bytes_to_play') and gtts_available:
        st.audio(st.session_state.audio_bytes_to_play, format="audio/mp3", autoplay=False)
        st.session_state.audio_bytes_to_play = None

    st.sidebar.markdown("---")
    st.sidebar.caption("VQA: Local STT, API TTS/Dịch, Tự động nghe (Thử nghiệm)")


# --- Điểm vào chính ---
if __name__ == "__main__":
    
    # Kiểm tra và cảnh báo về GEMINI_API_KEY nếu cần
    if gemini_available and (not GEMINI_API_KEY or GEMINI_API_KEY == "AIzaSyD22-9DA9oTtVVxQ1iOmrmx7Xre_kaLqdU"):
        startup_warnings.append("GEMINI API KEY: Vui lòng đặt GEMINI_API_KEY nếu muốn sử dụng model Gemini.")
    run()