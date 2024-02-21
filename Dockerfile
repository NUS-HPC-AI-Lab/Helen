FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

# ==================================================================
# tools
# ------------------------------------------------------------------
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
    apt-utils \
    build-essential \
    ca-certificates \
    cmake \
    wget \
    git \
    vim \
	ffmpeg \
	libopenmpi-dev \
	tmux \
	htop \
	libosmesa6-dev \
	libgl1-mesa-glx \
	libglfw3 \
    imagemagick \
    curl \
    libjpeg-dev \
    libpng-dev \
    axel \
    zip \
    unzip

# ==================================================================
# python packages
# ------------------------------------------------------------------
RUN PIP_INSTALL="python -m pip --no-cache-dir install" && \
DEBIAN_FRONTEND=noninteractive $PIP_INSTALL \
    tqdm \
    pandas==1.3.5 \
    scipy==1.7.3 \
    scikit-learn==1.0.2 \
    h5py==3.7.0 \
    tensorboard==2.10.1 \
    hydra-core==1.2.0 \
    openpyxl==3.0.10


# ==================================================================
# config & cleanup
# ------------------------------------------------------------------

RUN ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*