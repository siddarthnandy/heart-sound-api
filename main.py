from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import librosa
import soundfile as sf
import tempfile
import os
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all websites (change to specific domains if needed)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Load the trained model
MODEL_PATH = "best_heart_sound_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Function to preprocess audio (ONLY what is necessary)
def preprocess_audio(file_path, n_mfcc=40):
    # Load audio and downsample from 44.1kHz to 22.05kHz (matches training)
    y, sr = librosa.load(file_path, sr=22050, duration=5.0)

    # Extract MFCC features (40 coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # Ensure correct input shape (40, 216, 1)
    mfcc = np.expand_dims(mfcc, axis=-1)  # Add channel dimension

    return np.expand_dims(mfcc, axis=0)  # Add batch dimension

# API route to check if the server is running
@app.get("/")
def home():
    return {"message": "CardioAI Heart Sound Analysis API is running!"}

# API route to upload and analyze audio
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(await file.read())
            temp_path = temp_audio.name

        # Preprocess audio (no filtering, no pre-emphasis, just MFCC)
        input_data = preprocess_audio(temp_path)

        # Make a prediction
        prediction = model.predict(input_data)[0, 0]  # Get scalar value
        status = "Abnormal" if prediction > 0.5 else "Healthy"

        # Clean up temporary files
        os.unlink(temp_path)

        return {"prediction": float(prediction), "status": status}

    except Exception as e:
        return {"error": str(e)}

# Start the FastAPI server on port 8000
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
