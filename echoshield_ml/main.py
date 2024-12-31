from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import torch
from fastapi.middleware.cors import CORSMiddleware
from scipy.stats import skew, kurtosis, median_abs_deviation
import shutil
import os
import uvicorn
import base64
import librosa
from datetime import datetime
from features import classify_audio, extract_features


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI()

origins = [
"*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from torchaudio.pipelines import WAV2VEC2_BASE
bundle = WAV2VEC2_BASE
model = bundle.get_model()
print("Model downloaded successfully!")


SAVE_DIR = "./audio"
os.makedirs(SAVE_DIR, exist_ok=True)
from collections import Counter
import wave

import os
import shutil
from datetime import datetime
import soundfile as sf
import librosa
import torch.nn.functional as F
import subprocess
from fastapi import UploadFile, File
from fastapi.responses import JSONResponse
from checking import UnifiedDeepfakeDetector

SAVE_DIR = './audio' 
def reencode_audio(input_path, output_path):
    command = [
        'ffmpeg', '-i', input_path, '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', output_path
    ]
    subprocess.run(command, check=True)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    print(f"Received file: {file.filename}")

    original_filename = file.filename.rsplit('.', 1)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_filename = os.path.join(SAVE_DIR, f"{timestamp}.wav")
    reencoded_filename = os.path.join(SAVE_DIR, f"{timestamp}_reencoded.wav")

    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(wav_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    reencode_audio(wav_filename, reencoded_filename)
    os.remove(wav_filename)
    print(f"File successfully re-encoded as: {reencoded_filename}")

    try:
        audio, sr = librosa.load(reencoded_filename, sr=None)  
        print("Loaded successfully with librosa")
    except Exception as e:
        print(f"Error loading re-encoded file: {e}")
    new_features = extract_features(reencoded_filename)
    prediction, entropy = classify_audio(new_features)
    with open(reencoded_filename, "rb") as audio_file:
        audio_data = audio_file.read()

    # audio_base64 = base64.b64encode(audio_data).decode('utf-8')
    os.remove(reencoded_filename)
    return JSONResponse(content={
        "prediction": bool(prediction),
        "entropy": float(entropy),
    })
    
    
@app.post("/upload_audio")
async def upload_file(file: UploadFile = File(...)):
    print(f"Received file: {file.filename}")

    original_filename = file.filename.rsplit('.', 1)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_filename = os.path.join(SAVE_DIR, f"{timestamp}.wav")
    reencoded_filename = os.path.join(SAVE_DIR, f"{timestamp}_reencoded.wav")

    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(wav_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    reencode_audio(wav_filename, reencoded_filename)
    
    os.remove(wav_filename)
    print(f"File successfully re-encoded as: {reencoded_filename}")

    try:
        audio, sr = librosa.load(reencoded_filename, sr=None)  
        print("Loaded successfully with librosa")
    except Exception as e:
        print(f"Error loading re-encoded file: {e}")
    new_features = extract_features(reencoded_filename)
    detector = UnifiedDeepfakeDetector()
    print(reencoded_filename)
    result = detector.analyze_audio_rf(reencoded_filename, model_choice="all")
    prediction, entropy = classify_audio(new_features)
    with open(reencoded_filename, "rb") as audio_file:
        audio_data = audio_file.read()
    result = list(result)
    result.append("FAKE" if float(entropy) < 150 else "REAL")
    print(result)
    r_normalized = [x.upper() for x in result]
    counter = Counter(r_normalized)

    most_common_element, _ = counter.most_common(1)[0]

    print(f"The most frequent element is: {most_common_element}") 
    

    audio_base64 = base64.b64encode(audio_data).decode('utf-8')
    print(f"Audio Data Length: {len(audio_data)}")

    os.remove(reencoded_filename)
    return JSONResponse(content={
        "filename": file.filename,
        "prediction": most_common_element.upper(),
        "entropy": float(entropy),
        "audio": audio_base64,
        "content_type": "audio/wav"
    })


    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)