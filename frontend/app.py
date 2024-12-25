import streamlit as st
from st_audiorec import st_audiorec
import os
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # app.py dir
AUDIO_SAVE_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "audio"))  # audio dir
os.makedirs(AUDIO_SAVE_PATH, exist_ok=True)

# Backend URL
BACKEND_URL = "http://127.0.0.1:8000/predict"

# Page layout and header
st.set_page_config(page_title="Audio Recorder", page_icon="🎙️", layout="centered")
st.title("🎙️ Voice Recognition Application")
st.subheader("Record and Save Your Audio")

# Add sidebar with "About Us"
with st.sidebar:
    st.header("About Us")
    st.write("""
        Chào mừng đến với ứng dụng nhận diện giọng nói của chúng tôi!
        Ứng dụng này được phát triển để hỗ trợ việc ghi âm, lưu trữ và nhận diện giọng nói.
        Chúng tôi sử dụng công nghệ **Whisper Model** để đạt được độ chính xác cao.

        **Liên hệ:**
        - 📧 Email: support@voicerecognition.com
        - 🌐 Website: [Voice Recognition](https://voicerecognition.com)
    """)

# File uploader for audio files
st.subheader("Kéo và thả file âm thanh")
uploaded_file = st.file_uploader("Chọn một file âm thanh (định dạng WAV)", type=["wav"])

if uploaded_file is not None:
    file_path = os.path.join(AUDIO_SAVE_PATH, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success(f"✅ File đã được tải lên thành công: `{file_path}`")
    st.audio(file_path, format="audio/wav")

    # Process the uploaded file
    if st.button("Nhận diện âm thanh từ file đã tải lên"):
        with open(file_path, "rb") as audio_file:
            files = {"file": (uploaded_file.name, audio_file, "audio/wav")}
            response = requests.post(BACKEND_URL, files=files)

            if response.status_code == 200:
                result = response.json().get("result", "Không có kết quả")
                st.write("**Kết quả nhận diện:**", result)
            else:
                st.error("Lỗi khi gửi file đến backend")

# Audio recording
st.subheader("Hoặc ghi âm trực tiếp")
wav_audio_data = st_audiorec()

if wav_audio_data is not None:
    if len(wav_audio_data) > 44:
        st.success("🎉 Ghi âm thành công! Đang phát lại âm thanh...")
        file_path = os.path.join(AUDIO_SAVE_PATH, "recorded_audio.wav")

        if st.button("Lưu file âm thanh"):
            # Lưu file âm thanh
            with open(file_path, "wb") as f:
                f.write(wav_audio_data)

            st.success(f"✅ File đã được lưu thành công tại: `{file_path}`")
            st.audio(file_path, format="audio/wav")

        # Kiểm tra nếu nút nhận diện được bấm
        if st.button("Nhận diện âm thanh từ ghi âm"):
            if os.path.exists(file_path):  # Đảm bảo file đã được lưu
                with open(file_path, "rb") as audio_file:
                    files = {"file": ("recorded_audio.wav", audio_file, "audio/wav")}
                    response = requests.post(BACKEND_URL, files=files)

                    if response.status_code == 200:
                        result = response.json().get("result", "Không có kết quả")
                        st.write("**Kết quả nhận diện:**", result)
                    else:
                        st.error("Lỗi khi gửi file đến backend")
            else:
                st.warning("⚠️ Vui lòng lưu file âm thanh trước khi nhận diện.")
    else:
        st.warning("⚠️ Không có dữ liệu ghi âm hợp lệ. Vui lòng thử lại.")
else:
    st.info("ℹ️ Vui lòng nhấn **'Start Recording'** và nói vào micro.")

# Footer
st.markdown("---")
st.caption("Ứng dụng nhận diện giọng nói dựa trên Whisper Model")
