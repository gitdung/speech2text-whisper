import streamlit as st
from st_audiorec import st_audiorec
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # app.py dir
AUDIO_SAVE_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "audio"))  # audio dir
os.makedirs(AUDIO_SAVE_PATH, exist_ok=True)

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

        if st.button("Lưu file âm thanh"):
            file_path = os.path.join(AUDIO_SAVE_PATH, "recorded_audio.wav")

            with open(file_path, "wb") as f:
                f.write(wav_audio_data)

            st.success(f"✅ File đã được lưu thành công tại: `{file_path}`")
            st.audio(file_path, format="audio/wav")
    else:
        st.warning("⚠️ Không có dữ liệu ghi âm hợp lệ. Vui lòng thử lại.")
else:
    st.info("ℹ️ Vui lòng nhấn **'Start Recording'** và nói vào micro.")

# Footer
st.markdown("---")
st.caption("Ứng dụng nhận diện giọng nói dựa trên Whisper Model")
