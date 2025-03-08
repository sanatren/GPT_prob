import os
import sys
import uuid
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from supabase import create_client
import streamlit as st

# Constants
MAX_HISTORY_LENGTH = 20  # Maximum number of messages to keep in memory
MAX_TOKEN_LENGTH = 4000  # Maximum tokens for context window

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

supabase = create_client(supabase_url, supabase_key)
model = None  # Initialize as None

session_data = {}

def get_model(api_key):
    """Get or create ChatOpenAI model with the given API key"""
    global model
    if model is None or model.openai_api_key != api_key:
        model = ChatOpenAI(
            model="gpt-3.5-turbo", 
            openai_api_key=api_key,
            streaming=True,  # Enable streaming
            temperature=0.7
        )
    return model

def get_session_history(session_id: str):
    """Retrieves chat history for a session, initializing if necessary."""
    if session_id not in session_data:
        session_data[session_id] = {
            "history": [],
            "language": "English",
            "created_at": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat()
        }
    return session_data[session_id]["history"]

def trim_history(history, max_length=MAX_HISTORY_LENGTH):
    """Trims the chat history to prevent context window overflow."""
    if len(history) > max_length:
        return history[-max_length:]
    return history

def set_session_language(session_id: str, language: str = "English"):
    """Stores the preferred response language for a given session."""
    if session_id not in session_data:
        session_data[session_id] = {
            "history": [],
            "language": language,
            "created_at": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat()
        }
    else:
        session_data[session_id]["language"] = language
        session_data[session_id]["last_accessed"] = datetime.now().isoformat()

def save_session_to_supabase(session_id, session_name, language="English"):
    """Saves or updates session metadata in Supabase."""
    try:
        existing = supabase.table("sessions").select("*").eq("session_id", session_id).execute()
        
        if existing.data:
            # Update existing session
            data = {
                "name": session_name,
                "language": language
                # Let Postgres handle last_accessed timestamp
            }
            response = supabase.table("sessions").update(data).eq("session_id", session_id).execute()
        else:
            # Create new session
            data = {
                "session_id": session_id,
                "name": session_name,
                "language": language
                # Let Postgres handle created_at and last_accessed timestamps
            }
            response = supabase.table("sessions").insert(data).execute()
        return response
    except Exception as e:
        print(f"Error saving session: {e}")
        return None

def save_message_to_supabase(session_id, role, message):
    """Stores chat messages in Supabase with enhanced error handling."""
    try:
        # First ensure session exists to satisfy foreign key constraint
        save_session_to_supabase(session_id, "Untitled Chat")
        
        data = {
            "session_id": session_id,
            "role": role,
            "message": message
            # No user_id field
        }
        response = supabase.table("history").insert(data).execute()
        return response
    except Exception as e:
        print(f"Supabase Error: {e}")
        return None

def get_chat_history_from_supabase(session_id):
    """Retrieves chat history from Supabase."""
    try:
        response = supabase.table("history").select("*").eq("session_id", session_id).order("timestamp").execute()
        return response.data
    except Exception as e:
        print(f"Error retrieving chat history: {e}")
        return []

def delete_session(session_id):
    """Deletes a session and its messages."""
    try:
        # Due to ON DELETE CASCADE, we only need to delete the session
        supabase.table("sessions").delete().eq("session_id", session_id).execute()
        if session_id in session_data:
            del session_data[session_id]
        return True
    except Exception as e:
        print(f"Error deleting session: {e}")
        return False

def invoke_with_language(session_id: str, messages, language=None):
    """Handles chatbot invocation with memory management and language support."""
    try:
        api_key = st.session_state.get('openai_api_key') or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required")
            
        model = get_model(api_key)
        
        if session_id not in session_data:
            set_session_language(session_id, "English")

        if language:
            set_session_language(session_id, language)
        else:
            language = session_data[session_id]["language"]

        # Get chat history
        history = get_chat_history_from_supabase(session_id)
        
        # Format messages for the model
        formatted_messages = []
        
        # Add system message
        system_message = f"""IMPORTANT: You are an AI assistant that MUST ALWAYS respond in {language} only.
        
        Core Rules:
        1. ALWAYS respond in {language}, regardless of the input language
        2. If you cannot translate something perfectly, explain it simply in {language}
        3. Maintain conversation context and memory across language changes
        4. Remember user details and preferences
        5. Never switch to another language, even if explicitly asked
        
        Current conversation language: {language}
        """
        
        formatted_messages.append(HumanMessage(content=system_message))
        
        # Add chat history
        for msg in history[-MAX_HISTORY_LENGTH:]:
            if msg["role"] == "user":
                formatted_messages.append(HumanMessage(content=msg["message"]))
            else:
                formatted_messages.append(AIMessage(content=msg["message"]))
        
        # Add the current message
        formatted_messages.append(HumanMessage(content=messages[0].content))

        # Create placeholder for streaming response
        placeholder = st.empty()
        full_response = ""
        
        # Stream the response
        for chunk in model.stream(formatted_messages):
            if chunk.content:
                full_response += chunk.content
                # Update the response in real-time with cursor
                placeholder.markdown(full_response + "▌")
        
        # Show final response without cursor
        placeholder.markdown(full_response)
        
        return full_response

    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        print(error_msg)
        return error_msg