import os
import base64
import logging
import importlib
import streamlit as st
from typing import List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Voice & Vision
from gtts import gTTS

# LangChain Imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.tools import tool

# --- SETUP & CONFIGURATION ---
load_dotenv() # Load API Key from .env file

# Configure Logging (Step 10)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# --- STEP 1: DEFINE ROLE AND GOAL ---
AGENT_ROLE = """
You are Dr. Agent, a specialized Medical Assistant AI. 
Your goal is to analyze patient symptoms to provide a preliminary diagnosis summary.
You assist doctors by summarizing findings, checking against medical databases, and speaking the results.
You are professional, empathetic, and precise. You always clarify that you are an AI and not a replacement for a human doctor.
"""

# --- STEP 2: STRUCTURED INPUT & OUTPUT ---
class MedicalDiagnosisOutput(BaseModel):
    summary: str = Field(description="Brief summary of the patient's condition")
    possible_conditions: List[str] = Field(description="List of potential diagnoses based on evidence")
    recommended_actions: List[str] = Field(description="Next steps (e.g., 'Refer to Cardiologist')")
    severity_score: int = Field(description="Urgency score from 1-10")

    def to_markdown(self):
        return f"""
        ### 📋 Diagnosis Report
        **Summary:** {self.summary}  
        **Severity Score:** {self.severity_score}/10  
        
        **Possible Conditions:**
        * {', '.join(self.possible_conditions)}
        
        **Recommended Actions:**
        * {', '.join(self.recommended_actions)}
        """

# --- STEP 6: RAG (KNOWLEDGE BASE) ---
def _load_faiss():
    try:
        module = importlib.import_module("langchain_community.vectorstores")
        return getattr(module, "FAISS", None)
    except Exception as e:
        logging.warning(f"FAISS unavailable: {e}")
        return None

# We initialize this globally so we don't reload it on every button press
@st.cache_resource
def setup_knowledge_base():
    # Mock Medical Textbook Data
    docs = [
        Document(page_content="Flu symptoms: high fever, muscle aches, dry cough, fatigue. Treatment: Rest, fluids.", metadata={"source": "med_book_1"}),
        Document(page_content="Bronchitis symptoms: persistent cough with mucus, shortness of breath, low fever. Treatment: Bronchodilators.", metadata={"source": "med_book_2"}),
        Document(page_content="Migraine indicators: unilateral pulsing head pain, light sensitivity, nausea. Treatment: Dark room, pain relievers.", metadata={"source": "med_book_3"}),
    ]
    # NOTE: You need an OpenAI API Key for Embeddings to work
    if os.getenv("OPENAI_API_KEY"):
        faiss_cls = _load_faiss()
        if faiss_cls:
            return faiss_cls.from_documents(docs, OpenAIEmbeddings())
    return None

# --- STEP 3, 4, 5: REASONING & ORCHESTRATION ---
def run_diagnosis_agent(symptoms: str, history: str, xray_analysis: str):
    """
    Simulates the Multi-Agent flow: 
    1. Retrieve Context (Researcher)
    2. Generate Diagnosis (Doctor)
    """
    
    # Check for API Key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return MedicalDiagnosisOutput(
            summary="API Key Missing. Please check your .env file.",
            possible_conditions=[],
            recommended_actions=[],
            severity_score=0
        )

    # 1. Researcher Step (RAG Retrieval)
    vector_db = setup_knowledge_base()
    context_text = "No context available."
    if vector_db:
        retriever = vector_db.as_retriever()
        docs = retriever.invoke(symptoms)
        context_text = "\n".join([d.page_content for d in docs])
    
    # 2.  LLM Call
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Structured Output enforcement
    structured_llm = llm.with_structured_output(MedicalDiagnosisOutput)
    
    system_prompt = f"{AGENT_ROLE}\n\nPROTOCOL: Use the Context provided to inform your diagnosis. Be conservative."
    
    user_prompt = f"""
    Patient Symptoms: {symptoms}
    Patient History: {history}
    X-Ray Analysis (Visual Data): {xray_analysis}
    Medical Context (RAG): {context_text}
    """
    
    messages = [
        ("system", system_prompt),
        ("human", user_prompt)
    ]
    
    # Invoke the chain
    result = structured_llm.invoke(messages)
    return result

# --- STEP 7: VOICE & VISION ---
def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en')
        tts.save("diagnosis.mp3")
        return "diagnosis.mp3"
    except Exception as e:
        st.error(f"Audio generation failed: {e}")
        return None

# --- STEP 9: UI (STREAMLIT APP) ---
def main():
    st.set_page_config(page_title="Dr. Agent", page_icon="🏥")
    
    st.title("🏥 Dr. Agent: AI Medical Assistant")
    st.markdown("Build reliable AI agents with **Reasoning, RAG, and Structure**.")
    
    # Sidebar for Inputs
    with st.sidebar:
        st.header("Patient Data Input")
        symptoms = st.text_area("Reported Symptoms", "Patient has a dry cough and high fever.")
        history = st.text_input("Medical History", "No known allergies. Smoker.")
        uploaded_file = st.file_uploader("Upload X-Ray (Optional)", type=["jpg", "png", "jpeg"])
    
    # Main Action Area
    if st.button("Analyze Patient Case", type="primary"):
        with st.spinner("Consulting Medical Knowledge Base & Analyzing Data..."):
            
            # Mock Vision Analysis if file is uploaded
            xray_notes = "No X-ray provided."
            if uploaded_file:
                st.image(uploaded_file, caption="Uploaded X-Ray", width=200)
                # In a real app, you would send this image to GPT-4o-Vision here.
                xray_notes = "Vision Model detects mild inflammation in upper lungs."
            
            # RUN THE AGENT (Steps 3-6)
            diagnosis = run_diagnosis_agent(symptoms, history, xray_notes)
            
            # LOGGING (Step 10)
            logging.info(f"Diagnosis Generated for: {symptoms[:20]}...")

            # DISPLAY OUTPUT (Step 8)
            st.markdown("---")
            st.markdown(diagnosis.to_markdown())
            
            # VOICE OUTPUT (Step 7)
            if diagnosis.summary:
                st.markdown("### 🔊 Audio Summary")
                audio_file = text_to_speech(diagnosis.summary)
                if audio_file:
                    st.audio(audio_file)

if __name__ == "__main__":
    main()