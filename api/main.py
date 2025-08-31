from fastapi import FastAPI, File, UploadFile, Query
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Load Models
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
PROD_MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "1.keras")
BETA_MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "2.keras")

print("Loading Production Model from:", PROD_MODEL_PATH)
prod_model = tf.keras.models.load_model(PROD_MODEL_PATH)

print("Loading Beta Model from:", BETA_MODEL_PATH)
beta_model = tf.keras.models.load_model(BETA_MODEL_PATH)

# Classes
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# -------------------------------
# Utility Function
# -------------------------------
def read_file_as_image(data: bytes) -> np.ndarray:
    """Read image file into numpy array"""
    image = Image.open(BytesIO(data))
    image = image.resize((256, 256))  # resize if your model expects fixed size
    image = np.array(image) / 255.0   # normalize if trained with normalized data
    return image

# -------------------------------
# Endpoints
# -------------------------------
@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive!"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_type: str = Query("prod", enum=["prod", "beta"])  # choose model in request
):
    # Read image
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, axis=0)

    # Select model
    if model_type == "prod":
        model = prod_model
    else:
        model = beta_model

    # Predict
    predictions = model.predict(image_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = float(np.max(predictions[0]))

    print(f"Model: {model_type}, Class: {predicted_class}, Confidence: {confidence}")

    return {
        "model": model_type,
        "class": predicted_class,
        "confidence": confidence
    }


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
