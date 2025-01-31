# Define a default base image for the NVIDIA CUDA runtime
ARG IMAGE_NAME=nvidia/cuda
ARG IMAGE_TAG=12.4.1-cudnn-devel-ubuntu22.04

# Use the ARG values in the FROM directive
FROM ${IMAGE_NAME}:${IMAGE_TAG} as base

# Set environment variables for cuDNN
# ENV NV_CUDNN_VERSION=9.5.1.17-1
# ENV NV_CUDNN_PACKAGE_NAME=libcudnn9-cuda-12
# ENV NV_CUDNN_PACKAGE=${NV_CUDNN_PACKAGE_NAME}=${NV_CUDNN_VERSION}

# Install cuDNN and Python dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    # && apt-mark hold ${NV_CUDNN_PACKAGE_NAME} \
    && rm -rf /var/lib/apt/lists/*

# Set working directory for your application
WORKDIR /usr/src/

# Copy and install Python dependencies globally
COPY ./requirements.txt .
ARG UPDATE_DEPS=false
RUN if [ "$UPDATE_DEPS" = "true" ]; then \
        pip install --upgrade -r requirements.txt; \
    else \
        pip install -r requirements.txt; \
    fi

# Copy the data and application files
WORKDIR /usr/src/data/
COPY ./app/data .

WORKDIR /usr/src/app/
COPY ./app .

# Define a volume for persistent data
VOLUME ./vectorstore

# Expose the Streamlit default port
EXPOSE 8501
CMD ["streamlit", "run", "auth.py"]