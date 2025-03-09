import os
from typing import List, Dict, Any
import streamlit as st
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from document_processor import DocumentProcessor

class RAGChain:
    """Retrieval-Augmented Generation chain for answering questions based on uploaded documents"""
    
    def __init__(self, api_key: str):
        """Initialize the RAG chain with OpenAI API key"""
        self.api_key = api_key
        self.document_processor = DocumentProcessor()
        
        # Initialize the language model
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            openai_api_key=api_key,
            temperature=0.2
        )
        
        # Create the prompt template with chat history and language support
        self.prompt_template = PromptTemplate(
            input_variables=["question", "context", "chat_history", "language"],
            template="""
            You are an AI assistant that answers questions based on provided context from documents.
            
            Context information is below:
            ---------------------
            {context}
            ---------------------
            
            Previous conversation history:
            ---------------------
            {chat_history}
            ---------------------
            
            Given the context information and the conversation history, answer the question: {question}
            
            If the answer cannot be determined from the context, you can use your general knowledge to provide a helpful response.
            
            IMPORTANT: You must respond in {language}. If you don't know how to speak {language}, do your best to translate your response to {language}.
            
            Provide a comprehensive answer and cite the specific parts of the documents you're using when applicable.
            """
        )
        
        # Create the chain
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
    
    def process_file(self, uploaded_file, session_id: str) -> bool:
        """Process an uploaded file"""
        return self.document_processor.process_file(uploaded_file, session_id)
    
    def answer_question(self, question: str, session_id: str, chat_history: List[Dict[str, str]], language: str = "English") -> Dict[str, Any]:
        """
        Answer a question based on the uploaded documents and chat history
        
        Args:
            question: The user's question
            session_id: The current session ID
            chat_history: List of previous chat messages
            language: Language to respond in
            
        Returns:
            Dictionary with answer and source documents
        """
        # Retrieve relevant documents
        docs = self.document_processor.query_documents(question, session_id)
        
        # Format chat history for context
        formatted_history = ""
        for msg in chat_history[-5:]:  # Use last 5 messages for context
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted_history += f"{role}: {msg['message']}\n"
        
        if not docs:
            # If no documents are available, use general knowledge
            response = self.llm.invoke(
                [{"role": "system", "content": f"You are a helpful assistant. Please respond in {language}."},
                 {"role": "user", "content": f"Previous conversation:\n{formatted_history}\n\nQuestion: {question}"}]
            )
            return {
                "answer": response.content,
                "sources": []
            }
        
        # Format the context from retrieved documents
        context = "\n\n".join([
            f"Document: {doc['metadata']['source']}\nContent: {doc['content']}"
            for doc in docs
        ])
        
        # Generate the answer using both document context and chat history
        response = self.chain.invoke({
            "question": question,
            "context": context,
            "chat_history": formatted_history,
            "language": language
        })
        
        return {
            "answer": response["text"],
            "sources": docs
        }
    
    def clear_documents(self, session_id: str) -> bool:
        """Clear all documents for a session"""
        return self.document_processor.clear_documents(session_id) 