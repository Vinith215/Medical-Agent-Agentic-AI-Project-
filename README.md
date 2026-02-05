# Medical-Agent-Agentic-AI-Project

# 🏥 AI Medical Assistant

**Dr. Agent** is a specialized AI application capable of analyzing patient symptoms, reviewing medical history, and interpreting X-rays to provide a preliminary diagnosis summary. 

This project was built from scratch following the **"10 Steps to Build AI Agents"** framework by Dr. Maryam Miradi. It demonstrates how to orchestrate Reasoning, RAG (Retrieval-Augmented Generation), Vision, and Voice capabilities into a single cohesive agent.

---

## 🚀 Features (The 10 Steps)

This project implements the following architectural steps:

1.  **Role Definition:** Defined as a professional, empathetic medical assistant.
2.  **Structured I/O:** Uses **Pydantic** to enforce strict input schemas and JSON output.
3.  **Behavior Tuning:** System prompts ensure "Chain-of-Thought" reasoning and safety protocols.
4.  **Tool Use:** Equipped with capabilities to calculate severity and cross-reference data.
5.  **Multi-Agent Logic:** Simulates a "Researcher" (RAG) and "Doctor" (Decision Maker) workflow.
6.  **Memory & RAG:** Uses **FAISS** vector store to retrieve medical knowledge (mock textbooks).
7.  **Voice & Vision:**
    * **Vision:** (Mocked/Ready for GPT-4o) Interprets X-ray uploads.
    * **Voice:** Uses **gTTS** to speak the diagnosis summary.
8.  **Deliver Output:** Formats results into Markdown and standard JSON.
9.  **User Interface:** Wrapped in a clean **Streamlit** web app.
10. **Evaluation:** Includes logging to monitor agent reliability and output validity.

---

## 🛠️ Tech Stack

* **Language:** Python 3.9+
* **Orchestration:** LangChain
* **LLM:** OpenAI GPT-4o (via `langchain-openai`)
* **Validation:** Pydantic
* **Vector Store:** FAISS (Facebook AI Similarity Search)
* **Frontend:** Streamlit
* **Audio:** gTTS (Google Text-to-Speech)

---

## 📂 Project Structure

```text
medical-agent/
│
├── app.py                # Main application logic (AI + UI)
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (API Keys)
├── .gitignore            # Git exclusion rules
└── README.md             # Project documentation


**⚙️ Installation & Setup**

1. Prerequisites
Python installed on your machine.

An OpenAI API Key (required for the LLM and Embeddings).

2. Clone/Create the Project
Create a folder named medical-agent and navigate into it.

3. Create a Virtual Environment
It is recommended to use a virtual environment to manage dependencies.

# Windows
python -m venv venv
venv\Scripts\activate

4. Install Dependencies
pip install -r requirements.txt

5. Configure API Key
Create a file named .env in the root directory and add your OpenAI key:
OPENAI_API_KEY=sk-proj-your-actual-api-key-here

▶️ How to Run
1. Ensure your virtual environment is active.

2. Run the Streamlit application:
streamlit run app.py

3. The app will open automatically in your browser at http://localhost:8501.

🧪 Usage Guide
1. Enter Symptoms: In the sidebar, type symptoms (e.g., "Patient has high fever and muscle aches").
2. Add History: Add context (e.g., "No allergies, recent travel").
3. Upload X-Ray: (Optional) Upload a .png or .jpg image.
4. Click Analyze: The agent will:
    Retrieve similar cases from the vector database (RAG).
    Analyze the inputs using the LLM.
    Generate a structured report.
5. Listen: Click the audio player to hear the diagnosis.
