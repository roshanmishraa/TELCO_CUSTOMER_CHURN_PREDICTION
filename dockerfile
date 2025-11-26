# ---------------------
# 1. Base Image
# ---------------------
FROM python:3.11-slim

# Create directory
WORKDIR /app

# ---------------------
# 2. Install Dependencies
# ---------------------
COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ---------------------
# 3. Copy Application Code
# ---------------------
COPY . .

# ---------------------
# 4. Copy Model Artifacts (IMPORTANT)
# ---------------------
# Your model files must exist in: models/
# These will be added to container as /app/model
RUN mkdir -p /app/model
COPY models/final_xgb_model.pkl /app/model/final_xgb_model.pkl
COPY models/feature_columns.txt /app/model/feature_columns.txt

# ---------------------
# 5. Environment Variables
# ---------------------
# Make src importable
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MODEL_DIR=/app/model

# ---------------------
# 6. Expose port for FastAPI/Gradio
# ---------------------
EXPOSE 8000

# ---------------------
# 7. Start FastAPI (with Gradio mounted)
# ---------------------
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
