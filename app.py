import streamlit as st
from transformers import AutoTokenizer, AutoModelForTextToWaveform
import torch
import soundfile as sf

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
model = AutoModelForTextToWaveform.from_pretrained("facebook/mms-tts-eng")

# Streamlit UI setup
st.title("Text-to-Waveform Model")
st.write("Enter some text and generate the corresponding audio waveform.")

# Text input
text = st.text_input("Enter text:")

if text:
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt")

    # Generate waveform using the model's forward pass
    with torch.no_grad():
        output_waveform = model(**inputs).waveform

    # Convert the waveform tensor to numpy array
    output_waveform = output_waveform.squeeze().cpu().numpy()  # Convert to NumPy array

    # Save the waveform as a WAV file
    wav_filename = "output.wav"
    sf.write(wav_filename, output_waveform, samplerate=16000)

    # Play the audio file in the app
    st.audio(wav_filename)

    # Option to download the audio file
    st.download_button(label="Download Audio", data=open(wav_filename, 'rb'), file_name=wav_filename)

# Add a footer or additional information
# st.write("This model is based on the Facebook MMS-TTS model.")
