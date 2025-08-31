from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import requests

app = FastAPI()

# -------------------------------
# TensorFlow Serving URL
# -------------------------------
TF_SERVING_URL = "http://localhost:8501/v1/models/potatomodel:predict"

# -------------------------------
# Class names
# -------------------------------
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# -------------------------------
# Health check endpoint
# -------------------------------
@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive!"}

# -------------------------------
# Utility function
# -------------------------------
def read_file_as_image(data: bytes) -> np.ndarray:
    """
    Convert uploaded file bytes to a normalized RGB image array
    """
    image = Image.open(BytesIO(data)).convert("RGB")  # ensure 3 channels
    image = image.resize((256, 256))  # match model input size
    image = np.array(image) / 255.0   # normalize to [0,1]
    return image

# -------------------------------
# Prediction endpoint
# -------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        image = read_file_as_image(await file.read())
        image_batch = np.expand_dims(image, axis=0)  # batch of 1

        # Prepare payload for TensorFlow Serving
        payload = {"instances": image_batch.tolist()}

        # Call TensorFlow Serving
        response = requests.post(TF_SERVING_URL, json=payload)

        # Check HTTP response
        if response.status_code != 200:
            return {"error": "TensorFlow Serving error", "details": response.text}

        resp_json = response.json()
        if "predictions" not in resp_json:
            return {"error": "No predictions returned", "details": resp_json}

        # Extract prediction
        prediction = resp_json["predictions"][0]
        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        return {"prediction": predicted_class, "confidence": confidence}

    except Exception as e:
        return {"error": "Exception occurred during prediction", "details": str(e)}

# -------------------------------
# Run the app
# -------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
