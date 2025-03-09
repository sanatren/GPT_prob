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

# Try to import the audio recorder, but provide a fallback if not available
try:
    from audio_recorder_streamlit import audio_recorder
    AUDIO_RECORDER_AVAILABLE = True
except ImportError:
    AUDIO_RECORDER_AVAILABLE = False

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
        self.whisper_model = None  # Lazy-load the model
        
    @staticmethod
    @st.cache_resource
    def load_whisper_model(_hf_token=None):
        """Load the Whisper model (cached to avoid reloading)"""
        try:
            return pipeline(
                "automatic-speech-recognition", 
                model="openai/whisper-small",
                chunk_length_s=30,
                device=0 if torch.cuda.is_available() else -1,
                token=_hf_token  # Pass token if available
            )
        except Exception as e:
            st.warning(f"Failed to load Whisper model: {e}")
            # Try with a smaller model as fallback
            try:
                return pipeline(
                    "automatic-speech-recognition", 
                    model="openai/whisper-tiny",
                    chunk_length_s=30,
                    device=0 if torch.cuda.is_available() else -1,
                    token=_hf_token
                )
            except:
                st.error("Could not load any Whisper model. Will use OpenAI API only.")
                return None
        
    def record_audio(self, duration=5):
        """Record audio using streamlit-audio-recorder or file uploader"""
        st.write("### Voice Input")
        
        # Check if audio recorder is available
        if AUDIO_RECORDER_AVAILABLE:
            # Create tabs for different input methods
            tab1, tab2 = st.tabs(["Record Audio", "Upload Audio"])
            
            with tab1:
                st.write("Click the microphone button below to start recording. Click again to stop.")
                
                # Add audio recorder component with compatible parameters
                audio_bytes = audio_recorder(
                    text="Click to record",
                    recording_color="#e8b62c",
                    neutral_color="#6aa36f"
                )
                
                if audio_bytes:
                    # Display audio player for the recorded audio
                    st.audio(audio_bytes, format="audio/wav")
                    
                    # Save the recorded audio to a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        tmp_file.write(audio_bytes)
                        audio_path = tmp_file.name
                    
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
                    # Display audio player for the uploaded file
                    st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
                    
                    # Save the uploaded file to a temporary location
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.name.split(".")[-1]}') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        audio_path = tmp_file.name
                    
                    # Show success message
                    st.success("Audio uploaded successfully! Processing...")
                    
                    # Return the path to the saved file
                    return audio_path, None
        else:
            # Fallback to just file upload if audio recorder is not available
            st.write("Upload an audio file (WAV, MP3, M4A)")
            st.info("For a better experience, install the 'streamlit-audio-recorder' package to enable in-browser recording.")
            
            # File uploader for audio
            uploaded_file = st.file_uploader("Upload audio file", 
                                            type=["wav", "mp3", "m4a"])
            
            if uploaded_file is not None:
                # Display audio player for the uploaded file
                st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
                
                # Save the uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.name.split(".")[-1]}') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
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
            
            # Skip local model and use OpenAI API directly for reliability
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
                os.remove(audio_path)

    def process_voice_input(self, duration=5):
        """Process voice input with minimal UI disruption"""
        try:
            st.write("Starting voice input processing...")
            recording, sample_rate = self.record_audio(duration)
            if recording is None:
                st.warning("No recording detected")
                return None
            
            st.write(f"Recording received: {type(recording)}")
            
            # If recording is already a file path, use it directly
            if isinstance(recording, str) and os.path.exists(recording):
                audio_path = recording
                st.write(f"Using existing audio path: {audio_path}")
            else:
                audio_path = self.save_audio(recording, sample_rate)
                st.write(f"Saved audio to: {audio_path}")
            
            if audio_path is None:
                st.warning("Failed to save audio")
                return None
            
            st.write("Transcribing audio...")
            transcript = self.transcribe_audio(audio_path)
            st.write(f"Transcript result: {transcript}")
            return transcript
            
        except Exception as e:
            st.error(f"Voice input error: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return None 