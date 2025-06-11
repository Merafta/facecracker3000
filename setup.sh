#!/bin/bash

# This script runs during the container build process.
# It pre-downloads all the model files needed by deepface
# so that they are available when the application starts.

# Set the home directory for deepface
export DEEPFACE_HOME=/.deepface

# Create the directory
mkdir -p $DEEPFACE_HOME/weights/

# Run a simple Python script to trigger the downloads
python -c "from deepface import DeepFace; DeepFace.build_model('ArcFace'); DeepFace.build_model('RetinaFace')"

echo "DeepFace models downloaded successfully."