# app.py

from fastapi import FastAPI, File, UploadFile
from model import ImagePreprocessor, OnnxModel
from io import BytesIO
from PIL import Image
import numpy as np

app = FastAPI()
model = OnnxModel()
preprocessor = ImagePreprocessor()

@app.get("/health")
def health_check():
    return {"status": "running"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        image_array = preprocessor.transform(image).unsqueeze(0).numpy()
        prediction = model.predict(image_array)
        return {"predicted_class": prediction}
    except Exception as e:
        return {"error": str(e)}
