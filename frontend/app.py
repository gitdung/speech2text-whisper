import streamlit as st
from st_audiorec import st_audiorec
import os
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # app.py dir
AUDIO_SAVE_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "audio"))  # audio dir
os.makedirs(AUDIO_SAVE_PATH, exist_ok=True)

# Backend URL
BACKEND_URL = "http://127.0.0.1:8001/predict"

# Page layout and header
st.set_page_config(page_title="Audio Recorder", page_icon="🎙️", layout="centered")
st.title("🎙️ Voice Recognition Application")
st.subheader("Record and Save Your Audio")

st.write("Nhấn **'Start Recording'** và nói vào micro để ghi âm:")

# Audio recording
wav_audio_data = st_audiorec()

# Check if audio was recorded
if wav_audio_data is not None:
    if len(wav_audio_data) > 44:
        st.success("🎉 Ghi âm thành công! Đang phát lại âm thanh...")

        if st.button("Lưu file âm thanh"):
            file_path = os.path.join(AUDIO_SAVE_PATH, "recorded_audio.wav")

            # Save audio file
            with open(file_path, "wb") as f:
                f.write(wav_audio_data)

            st.success(f"✅ File đã được lưu thành công tại: `{file_path}`")
            st.audio(file_path, format="audio/wav")

        if st.button("Nhận diện âm thanh"):
            file_path = os.path.join(AUDIO_SAVE_PATH, "recorded_audio.wav")

            if os.path.exists(file_path):
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
