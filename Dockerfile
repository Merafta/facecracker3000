# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install system dependencies needed to build 'dlib'
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake

# Install the big, compiled libraries first to manage memory.
# We install dlib and then face-recognition which depends on it.
RUN pip install --no-cache-dir dlib
RUN pip install --no-cache-dir face-recognition

# Now install the rest of the packages from the requirements file.
# pip will see that face-recognition is already installed and will skip it.
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