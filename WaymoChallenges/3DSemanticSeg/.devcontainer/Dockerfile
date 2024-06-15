# Base image
FROM python:3.9-slim

# Change working dir
WORKDIR /workspace

# Copy user files to working dir
COPY . .

# Install Git
RUN apt-get update && apt-get install -y git

# Clone Waymo Open Dataset repo
RUN git clone https://github.com/waymo-research/waymo-open-dataset.git waymo-od

# Install packages
RUN pip install waymo-open-dataset-tf-2-12-0==1.6.4 Pillow==9.2.0

# Expose port
EXPOSE 8000