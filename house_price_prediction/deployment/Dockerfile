FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY house_price_prediction/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install fastapi uvicorn

# Copy application code
COPY house_price_prediction/ house_price_prediction/

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "house_price_prediction.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
