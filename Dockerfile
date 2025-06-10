# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install system dependencies required by OpenCV and other libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install the python packages from requirements.txt in one go
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
COPY . .

# Tell the container to expose port 7860 (the default for Hugging Face Spaces)
EXPOSE 7860

# Run the app using gunicorn when the container launches.
# This is the command that starts your server in production.
# We bind to 0.0.0.0 to make it accessible from outside the container.
# --timeout 600 allows for long processing times on large uploads.
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "600", "app:app"] 