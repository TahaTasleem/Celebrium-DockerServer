# test.py

from model import ImagePreprocessor, OnnxModel
import sys


def test_inference(image_path):
    print(f"Loading image: {image_path}")
    preprocessor = ImagePreprocessor()
    model = OnnxModel()

    input_array = preprocessor.preprocess(image_path)
    predicted_class = model.predict(input_array)

    print(f"Predicted class ID: {predicted_class}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test.py <path_to_image>")
    else:
        test_inference(sys.argv[1])
