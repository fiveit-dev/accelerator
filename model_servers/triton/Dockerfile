FROM nvcr.io/nvidia/tritonserver:24.01-py3

# Update and install necessary packages
RUN apt-get update && apt-get install -y python3 python3-pip ffmpeg

# Copy the requirements file
COPY requirements.txt /requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r /requirements.txt
