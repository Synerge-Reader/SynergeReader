FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requiredInstall.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requiredInstall.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p chroma_db uploads

# Expose port
EXPOSE 5000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000", "--reload"]
