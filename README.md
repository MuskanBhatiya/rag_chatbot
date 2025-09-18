# RAG HR Chatbot

A Retrieval-Augmented Generation (RAG) chatbot to answer HR policy queries using natural language processing. Built with FastAPI (backend) and Streamlit (UI), leveraging a vector database from the HR policy PDF.

## Project Overview
The chatbot uses the HR policy document (in `data/HR-Policy (1).pdf`) to provide accurate responses. Key sections include:
- **Introduction**: HR policies as guidelines for recruitment, promotion, compensation, training, etc.
- **Principles**: Place right person in right place, train employees, ensure tools/supply, better conditions, security/incentives, plan ahead.
- **Objectives**: Continuity of service, better communication, orientation/focus, mentoring reference.
- **Employment Contracts**: Accord for working relationship, compliance with Indian laws (wage, labour, Contract Act 1872).
- **Code of Conduct**: Vision, ethics, mission; rules for equal rights, dress code, conflict of interest; breach reporting procedure.
- **Employee Wages**: Payroll management, competitive salaries, government compliance.

The vector database is pre-built in `vectorstore/`. Local setup is prioritized due to Docker challenges (see Notes).

## Prerequisites
- Windows 10/11 with internet.
- Python 3.11 (download from [python.org](https://www.python.org/downloads/)).
- Git (download from [git-scm.com](https://git-scm.com/downloads)).

## Installation Steps
1. **Clone the Repository**:
   - Open Command Prompt.
   - Run: `git clone https://github.com/MuskanBhatiya/rag_chatbot.git`
   - Navigate: `cd rag_chatbot`
2. **Set Up Virtual Environment**:
   - Run: `python -m venv venv`
   - Activate: `venv\Scripts\activate` (see `(venv)` in prompt).
3. **Install Dependencies**:
   - Run: `pip install -r requirements.txt` (2-5 minutes).
4. **Create Vector Database** (if `vectorstore/` empty):
   - Run: `python core.py`

## Running the Chatbot
1. **Start Backend (FastAPI)**:
   - Open new Command Prompt.
   - Navigate: `cd rag_chatbot` (or full path like `D:\rag_chatbot`).
   - Activate: `venv\Scripts\activate`
   - Run: `uvicorn backend:app --host 0.0.0.0 --reload`
   - Confirm: "Uvicorn running on http://0.0.0.0:8000".
2. **Start UI (Streamlit)**:
   - Open another Command Prompt.
   - Navigate: `cd rag_chatbot` (or `D:\rag_chatbot`).
   - Activate: `venv\Scripts\activate`
   - Run: `streamlit run app.py`
   - Confirm: "You can now view your Streamlit app in your browser. URL: http://localhost:8501".
3. **Access Chatbot**:
   - Browser: `http://localhost:8501`
   - Query: "What is the maternity leave policy?" (expect concise answer with sources).

## Files Overview
- `app.py`: Streamlit UI.
- `backend.py`: FastAPI API.
- `core.py`: Vector DB creation.
- `requirements.txt`: Dependencies.
- `data/`: HR policy PDF.
- `vectorstore/`: Embeddings.
- `Dockerfile`, `start.sh`: Incomplete Docker (see Notes).

## Security Notes
- **Exposure**: `0.0.0.0` allows network access; use `localhost` in production.
- **Data**: Anonymize `data/` HR info; don't commit sensitive `.env`.
- **Dependencies**: Update with `pip list --outdated`.
- **Auth**: Add login to `backend.py`/`app.py`.

## Troubleshooting
- **UI Not Loading**: `netstat -aon | findstr :8501` > `taskkill /PID <pid> /F`.
- **Errors**: Reinstall `pip install -r requirements.txt`; recreate DB with `python core.py`.
- **Contact**: MuskanBhatiya (email: [muskanbhatiya364@gmail.com]).

## Notes
Docker setup (`Dockerfile`, `start.sh`) was explored but skipped due to time constraints and persistent build errors on my local machine. These issues arose from compatibility challenges with my Windows environment and the need to meet project deadlines, making local setup a more reliable and faster option for demonstration.

## License
MIT License.