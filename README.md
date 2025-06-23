# Celebrium Docker-Based Image Classification with ONNX

This project demonstrates deploying an image classification neural network (trained on ImageNet) using a **custom Docker-based ONNX model** on **Cerebrium's serverless platform**. The model predicts the class ID for an input image based on the 1000 ImageNet classes.

---

## 🚀 Overview

- **Model**: ResNet18-style classifier trained on ImageNet
- **Format**: Converted from `.pth` to `.onnx`
- **Deployment**: Docker container deployed on Cerebrium
- **Serving**: FastAPI app with `/predict/` endpoint
- **Input**: 224x224 RGB image
- **Output**: Predicted class ID (0–999)

---

## 📁 Folder Structure

.
├── assets/                # Sample images
├── convert_to_onnx.py     # Converts PyTorch model to ONNX with Celebrium deployment config
├── Dockerfile             # For building custom Docker container
├── model.onnx             # Converted ONNX model
├── model.py               # ONNX inference + preprocessing
├── app.py (or main.py)    # FastAPI server (entrypoint for Cerebrium)
├── requirements.txt       # Python dependencies
├── test.py                # Local inference testing
├── test_server.py         # Tests deployed Cerebrium endpoint
└── README.md              # This file

## 🛠️ Setup (Local)

### Install Dependencies

pip install -r requirements.txt

# Convert the Model to ONNX

python convert_to_onnx.py

# Local Testing

python test.py --image ./assets/n01440764_tench.jpeg

# Run with Docker Locally

## Build the Docker Image

docker build -t celebrium-docker-prod .

## Run the Container

docker run -p 8000:8000 celebrium-docker-prod

Access Swagger UI:
http://localhost:8000/docs

Use /predict/ endpoint to test images.

# Deploy to Cerebrium
## Requirements
Python 3.10+

# Cerebrium CLI

## Steps
## Install CLI

pip install cerebrium --upgrade

## Initialize Deployment

cerebrium init celebrium-docker-prod

## This creates a cerebrium.toml.
## Deploy

cerebrium deploy

## Once deployed, you'll get a dashboard URL and endpoint URL like:

GET https://api.cortex.cerebrium.ai/v4/p-xxxxx/celebrium-docker-prod/health_check
POST https://api.cortex.cerebrium.ai/v4/p-xxxxx/celebrium-docker-prod/predict/

## To test your deployed endpoint:

python test_server.py --image ./assets/n01667114_mud_turtle.jpeg
By default, the script uses your deployed /predict/ endpoint.

## Output will be:

✅ Success:
{'prediction': 35}

# test_server.py
## Supports:

--image: Path to local image
(Optional) --url and --api-key if auth is required

python test_server.py --image ./assets/sample.jpeg --url https://... --api-key sk_abc123
