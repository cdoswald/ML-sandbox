FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# install ubuntu dependencies (source: CleanRL package)
ENV DEBIAN_FRONTEND=noninteractive 
RUN apt-get update && \
    apt-get -y install python3-pip xvfb ffmpeg git build-essential python3-opengl && \
	apt-get clean
RUN ln -s /usr/bin/python3 /usr/bin/python

# install python dependencies
RUN pip install \
	typing_extensions \
	torch --extra-index-url https://download.pytorch.org/whl/cu118 \
	ffmpeg imageio-ffmpeg

# clone causal explorer repository and install CleanRL required packages
RUN mkdir workspace && cd workspace && \
	git clone https://github.com/cdoswald/cleanrl-causal-explorer.git && \
	cd cleanrl-causal-explorer && \
	pip install -r requirements/requirements-mujoco.txt

# Mount volume/copy local files