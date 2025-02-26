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
MODEL_PATH = "best_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Function to preprocess audio
def preprocess_audio(file_path, n_mfcc=40):
    y, sr = librosa.load(file_path, duration=5.0)
    y = librosa.effects.preemphasis(y)
    y = y / np.max(np.abs(y)) * 0.95  # Normalize amplitude
    
    # Feature extraction
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    combined_features = np.vstack((mfcc, mfcc_delta, mfcc_delta2)).T
    
    # Reshape for model input
    combined_features = combined_features[:216, :]  # Ensure fixed shape
    combined_features = np.expand_dims(combined_features, axis=0)  # Add batch dimension
    return combined_features

# Function to amplify audio
def amplify_audio(file_path, factor=5.0):
    y, sr = librosa.load(file_path, sr=None)
    y = np.clip(y * factor, -1.0, 1.0)  # Amplify and clip to avoid distortion
    amplified_path = file_path.replace(".wav", "_amplified.wav")
    sf.write(amplified_path, y, sr)
    return amplified_path

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

        # Amplify and preprocess audio
        amplified_path = amplify_audio(temp_path)
        input_data = preprocess_audio(amplified_path)

        # Make a prediction
        prediction = model.predict(input_data)[0, 0]  # Get scalar value
        status = "Abnormal" if prediction > 0.5 else "Healthy"

        # Clean up temporary files
        os.unlink(temp_path)
        os.unlink(amplified_path)

        return {"prediction": float(prediction), "status": status}

    except Exception as e:
        return {"error": str(e)}

# Start the FastAPI server on port 8000
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
