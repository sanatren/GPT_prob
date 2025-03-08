import os
import tempfile
import streamlit as st
from openai import OpenAI
import soundfile as sf
import numpy as np
from datetime import datetime
import time
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import queue
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
        self.audio_frames = []
        
    def record_audio(self, duration=5):
        """Record audio using WebRTC"""
        try:
            status = st.empty()
            self.audio_frames = []  # Reset frames
            
            def video_frame_callback(frame):
                """Process incoming audio frames"""
                try:
                    # Convert audio frame to numpy array
                    audio_data = frame.to_ndarray()
                    self.audio_frames.append(audio_data)
                except Exception as e:
                    st.error(f"Frame processing error: {e}")
                return frame
            
            # Create WebRTC streamer
            webrtc_ctx = webrtc_streamer(
                key="audio_recorder",
                mode=WebRtcMode.SENDONLY,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={
                    "audio": True,
                    "video": False
                },
                video_frame_callback=video_frame_callback,
                async_processing=True,
            )
            
            if webrtc_ctx.state.playing:
                status.write("🎙️ Recording... Speak now!")
                time.sleep(duration)  # Record for specified duration
                status.write("✅ Recording complete!")
                
                if self.audio_frames:
                    # Concatenate all audio frames
                    audio_data = np.concatenate(self.audio_frames, axis=0)
                    return audio_data, 48000  # WebRTC uses 48kHz sample rate
                    
            return None, None
            
        except Exception as e:
            st.error(f"Recording error: {str(e)}")
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
                with st.spinner("Converting speech to text..."):
                    transcript = self.client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file
                    )
            return transcript.text
        except Exception as e:
            st.error(f"Transcription error: {str(e)}")
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
            st.error(f"Voice input error: {str(e)}")
            return None 