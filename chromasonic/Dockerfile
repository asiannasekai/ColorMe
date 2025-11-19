FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libsndfile1-dev \
    libasound2-dev \
    portaudio19-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY setup.py .
COPY README.md .

# Install Chromasonic package
RUN pip install -e .

# Create necessary directories
RUN mkdir -p data/images data/audio models output

# Expose port for web interface
EXPOSE 5000

# Environment variables
ENV PYTHONPATH=/app/src
ENV FLASK_APP=chromasonic.web_interface.app
ENV FLASK_ENV=production

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# Default command
CMD ["python", "-m", "chromasonic.web_interface.app"]