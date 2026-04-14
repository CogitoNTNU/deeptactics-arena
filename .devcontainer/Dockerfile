FROM python:3.12-slim

# System dependencies needed by OpenCV, Box2D, pygame, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libsdl2-dev \
    swig \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /workspace

# Copy dependency files first for layer caching
COPY pyproject.toml ./
COPY requirements.txt ./

# Install dependencies using uv into the system Python (no venv needed in container)
RUN uv pip install --system -r requirements.txt
