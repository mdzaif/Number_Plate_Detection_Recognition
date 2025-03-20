# Use a base image
FROM python:3.12

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# Set environment variables for EasyOCR
ENV HOME=/app
RUN chmod -R 777 $HOME

# Copy the requirements file and install dependencies
COPY require.txt ./
RUN pip install --no-cache-dir -r require.txt

# Copy application files
COPY error_img ./error_img
COPY webui ./webui
COPY weights ./weights
COPY TF-ESPCN ./TF-ESPCN

# Expose application port
EXPOSE 7860

# Add a new user with home directory set
RUN useradd -m -d /app app
USER app

# Run the application
CMD ["python3", "-u", "webui/detect_recog_cuda.py"]
