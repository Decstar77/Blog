FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY runner/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy runner source and all project files (weights live alongside project scripts)
COPY runner/app.py .
COPY projects/ ./projects/

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
