import gradio as gr
import os
from datetime import datetime
import numpy as np
from scipy.io.wavfile import write as write_wav

# Create save directory (verify it exists)
RECORDINGS_DIR = os.path.abspath("recordings")
os.makedirs(RECORDINGS_DIR, exist_ok=True)


def save_audio(audio_input):
    try:
        if audio_input is None:
            return "Error: No audio data received"

        audio_data, sample_rate = audio_input
        if not isinstance(audio_data, np.ndarray) or sample_rate < 1000:
            return "Error: Invalid audio input format"

        # Convert to int16 format
        audio_data = (audio_data * 32767).astype(np.int16) if audio_data.dtype == np.float32 else audio_data.astype(
            np.int16)

        # Generate file path
        filename = os.path.join(RECORDINGS_DIR, f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
        write_wav(filename, sample_rate, audio_data)

        return f"Audio saved: {os.path.normpath(filename)}"  # Display path

    except Exception as e:
        return f"Operation failed: {str(e)}"


# Create interface (updated parameters)
interface = gr.Interface(
    fn=save_audio,
    inputs=gr.Audio(sources=["microphone"], type="numpy"),
    outputs="text",
    title="Audio Recording",
    description="Start recording, then stop to save the file",
    flagging_mode="never"  # replaces allow_flagging
)

# Launch with configuration
interface.launch(
    server_name="0.0.0.0",
    server_port=7863,
    show_error=True,
    share=False
)
