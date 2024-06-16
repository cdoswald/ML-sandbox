# Base image
FROM continuumio/miniconda3

# Copy environment file
COPY environment.yml /workspace/environment.yml

# Initialize Conda in shell and activate base
SHELL ["/bin/bash", "-c"]
RUN conda init bash && \
	echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
	echo "conda activate base" >> ~/.bashrc

# Create Conda environment
RUN conda env create -f /workspace/environment.yml --platform linux-64

# Activate Conda environment and clone Waymo repo
SHELL ["conda", "run", "-n", "waymo_open_env", "/bin/bash", "-c"]
RUN source ~/.bashrc && \
	conda activate waymo_open_env && \
	apt-get update && apt-get install -y git && \
	git clone https://github.com/waymo-research/waymo-open-dataset.git /workspace/waymo-od && \
	pip install waymo-open-dataset-tf-2-12-0==1.6.4

# Expose port
EXPOSE 8000