# # Use Python base image
# FROM python:3.12.11-slim

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     curl \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

# # Install Ollama (you'll need to adapt this based on your system architecture)
# RUN curl -L https://ollama.ai/download/ollama-linux-amd64 -o /usr/local/bin/ollama \
#     && chmod +x /usr/local/bin/ollama

# # Set working directory
# WORKDIR /app


# # Copy application code
# COPY . .

# # Expose the port your Flask app runs on
# EXPOSE 8080

# # Start Ollama service and your application
# CMD ["bash", "-c", "ollama serve & python app2.py"]