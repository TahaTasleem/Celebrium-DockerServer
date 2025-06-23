import requests
import argparse


def run_test(image_path: str, url: str, api_key: str = None):
    url = "https://api.cortex.cerebrium.ai/v4/p-b09172cb/celebrium-docker-prod/predict/"
    files = {"file": open(image_path, "rb")}
    headers = {}

    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = requests.post(url, files=files, headers=headers)

    if response.ok:
        print("Success:")
        print(response.json())
    else:
        print("Failed:", response.status_code)
        print(response.text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to test image")
    parser.add_argument(
        "--url",
        type=str,
        required=False,
        default="https://api.cortex.cerebrium.ai/v4/p-b09172cb/celebrium-docker-prod/predict/",
        help="Deployed model endpoint URL",
    )
    parser.add_argument("--api-key", type=str, required=False, help="API Key")

    args = parser.parse_args()

    run_test(args.image, args.url, args.api_key)
