# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /rag_chatbot

# Copy dependency list
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Copy startup script
COPY start.sh .
RUN chmod +x start.sh

# Expose backend (FastAPI/Uvicorn) and frontend (Streamlit) ports
EXPOSE 8000
EXPOSE 8501

# Run startup script
CMD ["./start.sh"]
