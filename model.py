# model.py

import onnxruntime as ort
from PIL import Image
import numpy as np
import torchvision.transforms as transforms


class ImagePreprocessor:
    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def preprocess(self, image_path: str) -> np.ndarray:
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0)  # shape: [1, 3, 224, 224]
        return img_tensor.numpy()


class OnnxModel:
    def __init__(self, model_path: str = "model_output/model.onnx"):
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, input_array: np.ndarray) -> int:
        outputs = self.session.run([self.output_name], {self.input_name: input_array})
        predictions = outputs[0]  # shape: [1, 1000]
        predicted_class = int(np.argmax(predictions))
        return predicted_class
