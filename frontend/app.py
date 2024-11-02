import streamlit as st
from st_audiorec import st_audiorec
import os
import requests
import io

BACKEND_URL = "http://127.0.0.1:8001/predict"

# Page layout and header
st.set_page_config(page_title="Audio Recorder", page_icon="🎙️", layout="centered")
st.title("🎙️ Voice Recognition Application")
st.subheader("Record and Save Your Audio")

st.write("Nhấn **'Start Recording'** và nói vào micro để ghi âm:")

# Audio recording
wav_audio_data = st_audiorec()

if wav_audio_data is not None:
    if len(wav_audio_data) > 44:
        st.success("🎉 Ghi âm thành công! Đang phát lại âm thanh...")

        if st.button("Nhận diện giọng nói"):
            audio_bytes = io.BytesIO(wav_audio_data)

            files = {'file': ('recorded_audio.wav', audio_bytes, 'audio/wav')}
            response = requests.post(BACKEND_URL, files=files)

            if response.status_code == 200:
                result = response.json().get("result")
                st.write("**Transcription**:", result)
            else:
                st.error("❌ Xảy ra lỗi khi nhận diện giọng nói.")
    else:
        st.warning("⚠️ Không có dữ liệu ghi âm hợp lệ. Vui lòng thử lại.")
else:
    st.info("ℹ️ Vui lòng nhấn **'Start Recording'** và nói vào micro.")

# Footer
st.markdown("---")
st.caption("Ứng dụng nhận diện giọng nói dựa trên Whisper Model")
