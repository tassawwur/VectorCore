# VectorCore - High-Performance Vector Database
# Multi-stage Docker build for production deployment

FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

# Set metadata
LABEL maintainer="VectorCore Team"
LABEL description="High-Performance Vector Database for AI Applications"
LABEL version="1.0.0"

# Create non-root user for security
RUN groupadd -r vectorcore && useradd -r -g vectorcore vectorcore

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /home/vectorcore/.local

# Copy application code
COPY --chown=vectorcore:vectorcore . .

# Make sure scripts are executable
RUN chmod +x main.py

# Switch to non-root user
USER vectorcore

# Set Python path
ENV PATH=/home/vectorcore/.local/bin:$PATH
ENV PYTHONPATH=/app

# Expose the default port
EXPOSE 8888

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import socket; s=socket.socket(); s.settimeout(5); s.connect(('localhost', 8888)); s.close()" || exit 1

# Default command
CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "8888"] 