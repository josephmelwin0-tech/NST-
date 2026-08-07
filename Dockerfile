# Use official lightweight Python image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

# Set working directory inside container
WORKDIR /app

# Install system dependencies needed for image processing / OpenCV if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install PyTorch and Torchvision CPU versions first (extremely lightweight compared to CUDA versions)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Expose port
EXPOSE 7860

# Set running directory to where app.py is located
WORKDIR /app/NST_Code

# Run the Flask app using Gunicorn for production-ready performance
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app", "--timeout", "120"]
