# Base image
FROM python:3.9-slim-buster

# Set the working directory
WORKDIR /app

# Import NVIDIA repository public key
RUN curl -sL https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -

# Set up the stable repository and update the package list
RUN echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    gnupg2 \
    curl \
    ca-certificates \
    cuda-drivers \
    cuda-toolkit-11-1 && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Flask app code
COPY . app

# Set environment variables
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:/usr/local/cuda/extras/CUPTI/lib64
ENV FLASK_APP=app/app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=8080

# Expose the Flask app port
EXPOSE 8080

# Start the Flask app
CMD ["flask", "run"]
