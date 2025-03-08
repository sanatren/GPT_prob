import os
import tempfile
import streamlit as st
from openai import OpenAI
import sounddevice as sd
import soundfile as sf
import numpy as np
from datetime import datetime
import time

from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

class AudioHandler:
    def __init__(self, api_key):
        """Initialize AudioHandler with OpenAI API key"""
        if not api_key or not api_key.startswith('sk-'):
            raise ValueError("Invalid OpenAI API key format. Key should start with 'sk-'")
            
        try:
            # Store API key and initialize client
            self.api_key = api_key
            self.client = OpenAI(api_key=self.api_key)
            
            # Audio settings
            self.sample_rate = 44100
            self.channels = 1
        except Exception as e:
            st.error(f"Error initializing OpenAI client: {e}")
            raise

    def record_audio(self, duration=5):
        """Record audio for specified duration"""
        try:
            # Simple status message
            status = st.empty()
            progress = st.progress(0)
            
            # Quick countdown
            for i in range(3, 0, -1):
                status.write(f"Starting in {i}...")
                time.sleep(0.5)
            
            # Record
            status.write("🎙️ Recording...")
            recording = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32
            )
            
            # Show progress
            for i in range(duration):
                progress.progress((i + 1) / duration)
                time.sleep(1)
            
            sd.wait()
            
            # Cleanup UI elements
            status.empty()
            progress.empty()
            
            # Validate recording
            if np.abs(recording).max() < 0.01:
                st.warning("No audio detected. Please speak louder.", icon="🔇")
                return None, None
                
            return recording, self.sample_rate
            
        except Exception as e:
            st.error("Microphone error. Please check permissions.", icon="🎤")
            return None, None

    def save_audio(self, recording, sample_rate):
        """Save recording to temporary file"""
        try:
            # Create temp file with wav extension
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
                sf.write(temp_audio.name, recording, sample_rate)
                return temp_audio.name
        except Exception as e:
            st.error(f"Error saving audio: {e}")
            return None

    def transcribe_audio(self, audio_path):
        """Transcribe audio using Whisper API"""
        try:
            with open(audio_path, "rb") as audio_file:
                with st.spinner("Transcribing audio..."):
                    # Create transcription without explicit api_key parameter
                    transcript = self.client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file
                    )
            return transcript.text
        except Exception as e:
            st.error(f"Error transcribing audio: {str(e)}")
            return None
        finally:
            # Clean up temp file
            if os.path.exists(audio_path):
                os.remove(audio_path)

    def process_voice_input(self, duration=5):
        """Process voice input with minimal UI disruption"""
        try:
            # Record
            recording, sample_rate = self.record_audio(duration)
            if recording is None:
                return None
                
            # Save and transcribe
            audio_path = self.save_audio(recording, sample_rate)
            if audio_path is None:
                return None
                
            # Transcribe
            transcript = self.transcribe_audio(audio_path)
            return transcript
            
        except Exception as e:
            st.error("Voice input failed. Please try again.", icon="❌")
            return None 