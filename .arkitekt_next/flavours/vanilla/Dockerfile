# Use an official Python runtime as a base image
FROM python:3.12-slim

# Set environment variables to prevent Python from generating .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Set the working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install "numpy==1.26.4"
RUN pip install "ashlarUC2" --no-dependencies
RUN pip install "tifffile"
# Copy the application code into the container


RUN pip install "scikit-image<=0.20"
RUN pip install blessed
RUN pip install scikit-learn
RUN pip install matplotlib


RUN pip install "arkitekt-next[all]==0.8.47"

COPY .arkitekt_next /app/.arkitekt_next
COPY api /app/api
COPY * /app/



RUN python app.py

# Expose any ports the app runs on
EXPOSE 8000
# additional port for Arkitekt?


