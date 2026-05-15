import gradio as gr
import os
import tos_appbk
import airsim_agent
import recognition_module

# Ensure target folder exists
output_dir = "recordings"
os.makedirs(output_dir, exist_ok=True)


def save_audio(audio):
    my_agent = airsim_agent.AirSimAgent(knowledge_prompt="prompts/aisim_lession52.txt")
    # audio is a tuple containing (sample_rate, audio_data)
    sample_rate, audio_data = audio

    # Generate unique filename
    file_name = f"recording_{len(os.listdir(output_dir)) + 1}.wav"
    file_path = os.path.join(output_dir, file_name)

    # Save audio file using scipy
    from scipy.io.wavfile import write
    write(file_path, sample_rate, audio_data)
    print(f"Recording saved: {file_path}")

    # Upload file to OSS
    oss_url = tos_appbk.upload_file(file_path, "mp3-audio", file_name)
    print(f"Recording saved: {file_path}, URL: {oss_url}")

    # Recognize speech
    text = recognition_module.process_mp3(oss_url)
    print(f"Recognition result: {text}")
    command = text
    python_code = my_agent.process(command, True)  # Execute code
    print("python_code: \n", python_code)
    #my_agent.process(text)
    return oss_url




# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## Record and Save Audio Locally")
    audio_input = gr.Audio(type="numpy", label="Recording")  # source parameter no longer needed
    output_text = gr.Textbox(label="Save status")

    # Bind button event
    record_button = gr.Button("Save Recording")
    record_button.click(save_audio, inputs=audio_input, outputs=output_text)

# Launch app
demo.launch()
