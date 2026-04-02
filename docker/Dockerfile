FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# X11 / OpenGL display forwarding
ENV DISPLAY=:0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

# Base dependencies + deadsnakes PPA for Python 3.8
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    wget \
    git \
    build-essential \
    libssl-dev \
    libffi-dev \
    # X11 and OpenGL libs for GUI forwarding
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglu1-mesa \
    libx11-6 \
    libxext6 \
    libxrender1 \
    x11-utils \
    x11-apps \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    python3.8-venv \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.8
RUN curl -sS https://bootstrap.pypa.io/pip/3.8/get-pip.py | python3.8

# Install virtualenv
RUN python3.8 -m pip install virtualenv

# Create workspace with open permissions so any UID can write
RUN mkdir -p /workspace/home && chmod 777 /workspace /workspace/home

WORKDIR /workspace

# Create virtualenv world-writable so the runtime UID can use it
RUN virtualenv phc -p python3.8 && chmod -R a+rX /workspace/phc

# Activate virtualenv by default in shell sessions
ENV PATH="/workspace/phc/bin:$PATH"
ENV VIRTUAL_ENV="/workspace/phc"

# Install torch + cuda dependencies (cuda 11.6 as per installation.md)
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# Copy PHC requirements and install (place requirement.txt in PHC/ before building)
COPY PHC/requirement.txt ./PHC/requirement.txt
RUN pip install -r PHC/requirement.txt

# Install Isaac Gym (must be present at third_party/isaacgym/ before building)
COPY third_party/isaacgym/python ./isaacgym_python
RUN pip install -e ./isaacgym_python

CMD ["/bin/bash"]
