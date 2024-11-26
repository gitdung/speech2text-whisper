import os
import time
import requests
import gradio as gr

# URL backend để xử lý nhận diện giọng nói
BACKEND_URL = "http://127.0.0.1:8001/predict"


def handle_audio(audio_path):
    if audio_path is None:
        return "⚠️ Không có dữ liệu ghi âm. Vui lòng thử lại.", None

    try:
        # Gửi file đến backend để nhận diện
        with open(audio_path, "rb") as audio_file:
            files = {"file": (os.path.basename(audio_path), audio_file, "audio/wav")}
            response = requests.post(BACKEND_URL, files=files)
            response.raise_for_status()

            # Xử lý kết quả từ backend
            if response.status_code == 200:
                result = response.json().get("result", "Không có kết quả")
                return result, audio_path
    except requests.exceptions.RequestException as e:
        return f"❌ Lỗi khi gửi file đến backend: {e}", None

    return "❌ Xử lý thất bại.", None


def main(audio_path):
    result_text, recorded_audio = handle_audio(audio_path)
    return result_text, recorded_audio


# UI
with gr.Blocks(css="""
    body {
        background-color: #f4f4f4;
    }
    .gradio-container {
        background-color: #ffffff;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 16px;
    }
    #pred-btn {
        background-color: #4CAF50;
        color: white;
        font-size: 22px;
    }
""") as app:
    gr.Markdown(
        """
        <div style="text-align: center;">
            <h1 style="color: #43bce8;">🎙️ Ứng Dụng Nhận Diện Giọng Nói Tiếng Việt</h1>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown('<h2 style="color: #43bce8;">Âm Thanh Đầu Vào</h3>')
            audio_input = gr.Audio(type="filepath", interactive=True)

        with gr.Column(scale=1):
            gr.Markdown('<h2 style="color: #43bce8;">Kết Quả Nhận Diện</h3>')
            result_text = gr.Textbox(label="Kết quả", interactive=False)
            playback_audio = gr.Audio(label="Âm thanh đã ghi")

    with gr.Row():
        pred_btn = gr.Button("🔍 Nhận Diện", variant="primary", elem_id="pred-btn")

    # Gán nút xử lý sự kiện
    # pred_btn.click(fn=main, inputs=audio_input, outputs=[result_text])
    pred_btn.click(fn=main, inputs=audio_input, outputs=[result_text, playback_audio])

# Khởi chạy ứng dụng
if __name__ == "__main__":
    app.launch(share=True)
