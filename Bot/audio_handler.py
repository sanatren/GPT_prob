import os
import tempfile
import streamlit as st
from openai import OpenAI
import soundfile as sf
import numpy as np
from datetime import datetime
import time
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import queue
import threading
import av

from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

class AudioHandler:
    def __init__(self, api_key):
        """Initialize AudioHandler with OpenAI API key"""
        if not api_key or not api_key.startswith('sk-'):
            raise ValueError("Invalid OpenAI API key format. Key should start with 'sk-'")
            
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)
        self.audio_queue = queue.Queue()
        self.recording = False
        
    def record_audio(self, duration=5):
        """Record audio using WebRTC"""
        try:
            status = st.empty()
            
            def audio_callback(frame):
                """Callback to receive audio frames"""
                if self.recording:
                    sound = frame.to_ndarray()
                    self.audio_queue.put(sound)
                return frame
            
            # WebRTC Configuration
            rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
            
            # Create WebRTC streamer
            ctx = webrtc_streamer(
                key="audio_recorder",
                mode=WebRtcMode.SENDONLY,
                rtc_configuration=rtc_config,
                media_stream_constraints={"video": False, "audio": True},
                audio_receiver_size=1024,
                async_processing=True,
                callback=audio_callback
            )
            
            if ctx.state.playing:
                status.write("🎙️ Recording...")
                self.recording = True
                time.sleep(duration)
                self.recording = False
                status.write("✅ Recording complete!")
                
                # Collect audio data
                audio_data = []
                while not self.audio_queue.empty():
                    audio_data.append(self.audio_queue.get())
                
                if audio_data:
                    # Concatenate audio chunks
                    recording = np.concatenate(audio_data, axis=0)
                    return recording, 48000  # WebRTC typically uses 48kHz
                
            return None, None
            
        except Exception as e:
            st.error("Microphone error. Please check permissions.", icon="🎤")
            return None, None

    def save_audio(self, recording, sample_rate):
        """Save recording to temporary file"""
        try:
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
                    transcript = self.client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file
                    )
            return transcript.text
        except Exception as e:
            st.error(f"Error transcribing audio: {str(e)}")
            return None
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

    def process_voice_input(self, duration=5):
        """Process voice input with minimal UI disruption"""
        try:
            recording, sample_rate = self.record_audio(duration)
            if recording is None:
                return None
                
            audio_path = self.save_audio(recording, sample_rate)
            if audio_path is None:
                return None
                
            transcript = self.transcribe_audio(audio_path)
            return transcript
            
        except Exception as e:
            st.error("Voice input failed. Please try again.", icon="❌")
            return None 