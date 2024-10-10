# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install the required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Set environment variables (you can replace these with values from your .env file or other configs)
ENV QDRANT_DB_URL='http://host.docker.internal:6333' \
    OLLAMA_BASE_URL='http://host.docker.internal:11434'

# Expose port 8000 for external access
EXPOSE 8000

# Command to run your application
CMD ["python", "api_server.py"]
