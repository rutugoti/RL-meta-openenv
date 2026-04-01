FROM python:3.11-slim

WORKDIR /app

# install deps first (cached layer — faster rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy all project files
COPY . .

# HF Spaces requires port 7860
EXPOSE 7860

# start the FastAPI server
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1"]