FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# X11 / OpenGL — DISPLAY is injected at runtime by run.sh via docker-compose
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

# Force NVIDIA as the OpenGL/EGL/Vulkan provider
ENV __GLX_VENDOR_LIBRARY_NAME=nvidia
ENV __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
ENV VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json

# Base dependencies + deadsnakes PPA for Python 3.8
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    wget \
    git \
    build-essential \
    libssl-dev \
    libffi-dev \
    gdb \
    strace \
    # X11 libs
    libx11-6 \
    libxext6 \
    libxrender1 \
    x11-utils \
    x11-apps \
    mesa-utils \
    # GLVND (GL vendor-neutral dispatch) — lets NVIDIA driver override mesa at runtime
    libglvnd0 \
    libglvnd-dev \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    # Keep mesa only as fallback for GLU (not as primary GL)
    libglu1-mesa \
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

# Copy Isaac Gym EGL/Vulkan JSON configs (required by libnvf.plugin.so for NvfDeviceCreate)
RUN mkdir -p /usr/share/glvnd/egl_vendor.d /etc/vulkan/icd.d
COPY third_party/isaacgym/docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
COPY third_party/isaacgym/docker/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json

# Fix Isaac Gym + NumPy 1.20+ incompatibility (np.float → float64)
RUN sed -i 's/dtype=np\.float,/dtype=np.float64,/g' ./isaacgym_python/isaacgym/torch_utils.py && \
    sed -i 's/dtype=np\.float)/dtype=np.float64)/g' ./isaacgym_python/isaacgym/torch_utils.py

RUN pip install -e ./isaacgym_python

# Add Isaac Gym bundled libs to LD_LIBRARY_PATH (libmem_filesys.so, libboost_*.so.1.68.0, etc.)
ENV LD_LIBRARY_PATH="/workspace/isaacgym_python/isaacgym/_bindings/linux-x86_64:${LD_LIBRARY_PATH}"

CMD ["/bin/bash"]
