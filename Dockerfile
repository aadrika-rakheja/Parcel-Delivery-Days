FROM python:3.12-slim

WORKDIR /app

# Copy requirements and install dependencies before copying code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code, package modules, and trained model
COPY src/ /app/src/
COPY app/ /app/app/
COPY models/ /app/models/
COPY params.yaml /app/params.yaml
COPY pytest.ini /app/pytest.ini

ENV PYTHONPATH=/app/src:/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
