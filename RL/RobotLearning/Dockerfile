FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# install Ubuntu dependencies
ENV DEBIAN_FRONTEND=noninteractive 
RUN apt-get update && \
    apt-get -y install \
        build-essential \
        cmake \
        curl \
        ffmpeg \
        git \
        libpython3-dev \
        libomp-dev \
        libopenblas-dev \
        libblas-dev \
        python3-dev \
        python3-opengl \
        python3-pip \
        wget \
        xvfb && \
	apt-get clean && rm -rf /var/lib/apt/lists/*
	
# creates symoblic link from python to python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# build PyTorch wheel file for SM8.9 from source
WORKDIR workspace
ENV TORCH_CUDA_ARCH_LIST="3.7;5.0;6.0;7.0;8.0;8.9;9.0"
ENV PATH="/root/.local/bin:${PATH}"
RUN pip install --upgrade pip setuptools wheel && \
    git clone --recursive https://github.com/pytorch/pytorch && \
    pip install -r pytorch/requirements.txt
RUN cd pytorch && python3 setup.py bdist_wheel && \
    mkdir -p /workspace/pytorch_wheel && \
    cp dist/*.whl /workspace/pytorch_wheel/ && \
    cd .. && rm -rf pytorch
    
# install PyTorch from wheel file
RUN pip install /workspace/pytorch_wheel/*.whl

# install python dependencies for CleanRL and MuJoCo
RUN git clone https://github.com/cdoswald/cleanrl-causal-explorer.git && \
	pip install -r cleanrl-causal-explorer/requirements/requirements-mujoco.txt
	
# install additional python dependencies
RUN pip install \
	ffmpeg \
	imageio-ffmpeg \
	typing_extensions \
	# torch --extra-index-url https://download.pytorch.org/whl/cu118
