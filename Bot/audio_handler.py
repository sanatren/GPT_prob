import os
import tempfile
import streamlit as st
from openai import OpenAI
import soundfile as sf
import numpy as np
from datetime import datetime
import time
import torch
from transformers import pipeline
import base64

# Try to import the mic recorder, but provide a fallback if not available
try:
    from streamlit_mic_recorder import mic_recorder
    MIC_RECORDER_AVAILABLE = True
except ImportError:
    MIC_RECORDER_AVAILABLE = False

from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
hf_token = os.getenv("HF_TOKEN")  # Add this to your .env file if you have one

class AudioHandler:
    def __init__(self, api_key):
        """Initialize AudioHandler with OpenAI API key"""
        if not api_key or not api_key.startswith('sk-'):
            raise ValueError("Invalid OpenAI API key format. Key should start with 'sk-'")
            
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)
        
    def record_audio(self, duration=5):
        """Record audio using streamlit-mic-recorder or file uploader"""
        st.write("### Voice Input")
        
        # Check if mic recorder is available
        if MIC_RECORDER_AVAILABLE:
            # Create tabs for different input methods
            tab1, tab2 = st.tabs(["Record Audio", "Upload Audio"])
            
            with tab1:
                st.write("Click the microphone button below to start recording. Click again to stop.")
                
                # Add mic recorder component
                audio_data = mic_recorder(
                    key="voice_recorder",
                    start_prompt="Click to start recording",
                    stop_prompt="Click to stop recording",
                    just_once=False,
                    use_container_width=False
                )
                
                if audio_data:
                    # The mic_recorder returns a dictionary with audio data
                    # Extract the audio bytes from the dictionary
                    if isinstance(audio_data, dict) and 'bytes' in audio_data:
                        audio_bytes = audio_data['bytes']
                        
                        # Save the recorded audio to a temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                            tmp_file.write(audio_bytes)
                            audio_path = tmp_file.name
                        
                        # Display audio player for the recorded audio - using file path to avoid binary data issues
                        try:
                            with open(audio_path, 'rb') as audio_file:
                                st.audio(audio_file.read(), format="audio/wav")
                        except Exception as e:
                            st.warning(f"Could not display audio player: {e}. Processing will continue.")
                        
                        # Show success message
                        st.success("Recording captured! Processing...")
                        
                        # Return the path to the saved file
                        return audio_path, None
            
            with tab2:
                st.write("Upload an audio file (WAV, MP3, M4A)")
                
                # File uploader for audio
                uploaded_file = st.file_uploader("Upload audio file", 
                                                type=["wav", "mp3", "m4a"],
                                                key="audio_upload")
                
                if uploaded_file is not None:
                    # Display audio player for the uploaded file using file bytes
                    file_bytes = uploaded_file.getvalue()
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    try:
                        st.audio(file_bytes, format=f"audio/{file_extension}")
                    except Exception as e:
                        st.warning(f"Could not display audio player: {e}. Processing will continue.")
                    
                    # Save the uploaded file to a temporary location
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
                        tmp_file.write(file_bytes)
                        audio_path = tmp_file.name
                    
                    # Show success message
                    st.success("Audio uploaded successfully! Processing...")
                    
                    # Return the path to the saved file
                    return audio_path, None
        else:
            # Fallback to just file upload if mic recorder is not available
            st.write("Upload an audio file (WAV, MP3, M4A)")
            st.info("For a better experience, install the 'streamlit-mic-recorder' package to enable in-browser recording.")
            
            # File uploader for audio
            uploaded_file = st.file_uploader("Upload audio file", 
                                            type=["wav", "mp3", "m4a"])
            
            if uploaded_file is not None:
                # Display audio player for the uploaded file using file bytes
                file_bytes = uploaded_file.getvalue()
                file_extension = uploaded_file.name.split('.')[-1].lower()
                try:
                    st.audio(file_bytes, format=f"audio/{file_extension}")
                except Exception as e:
                    st.warning(f"Could not display audio player: {e}. Processing will continue.")
                
                # Save the uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
                    tmp_file.write(file_bytes)
                    audio_path = tmp_file.name
                
                # Show success message
                st.success("Audio uploaded successfully! Processing...")
                
                # Return the path to the saved file
                return audio_path, None
            
        return None, None

    def save_audio(self, recording, sample_rate):
        """Handle the audio file path"""
        # If recording is already a file path, just return it
        if isinstance(recording, str) and os.path.exists(recording):
            return recording
        
        # Otherwise, save the recording to a file (for backward compatibility)
        try:
            if recording is None:
                return None
                
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
                if isinstance(recording, np.ndarray) and sample_rate:
                    sf.write(temp_audio.name, recording, sample_rate)
                else:
                    # Just write the bytes directly
                    temp_audio.write(recording)
                return temp_audio.name
        except Exception as e:
            st.error(f"Error saving audio: {e}")
            return None

    def transcribe_audio(self, audio_path):
        """Transcribe audio using OpenAI API directly for reliability"""
        try:
            if not audio_path or not os.path.exists(audio_path):
                return None
            
            # Use OpenAI API for transcription
            with open(audio_path, "rb") as audio_file:
                with st.spinner("Converting speech to text using OpenAI API..."):
                    transcript = self.client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file
                    )
            return transcript.text
        except Exception as e:
            st.error(f"Transcription error: {str(e)}")
            return None
        finally:
            # Clean up temp file
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except Exception as e:
                    st.warning(f"Could not remove temporary file: {e}")

    def process_voice_input(self, duration=5):
        """Process voice input with minimal UI disruption"""
        try:
            recording, sample_rate = self.record_audio(duration)
            if recording is None:
                return None
                
            # If recording is already a file path, use it directly
            if isinstance(recording, str) and os.path.exists(recording):
                audio_path = recording
            else:
                audio_path = self.save_audio(recording, sample_rate)
                
            if audio_path is None:
                return None
                
            transcript = self.transcribe_audio(audio_path)
            return transcript
            
        except Exception as e:
            st.error(f"Voice input error: {str(e)}")
            return None