import os
import numpy as np
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
# import gradio as gr
import datetime
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from transformers import pipeline

class UnifiedDeepfakeDetector:
    def __init__(self):
        self.input_shape = (224, 224, 3)
        self.vgg_model = self.build_vgg16_model()
        self.dense_model = tf.keras.models.load_model('deepfake_detection_model.h5')
        self.cnn_model = tf.keras.models.load_model('audio_deepfake_detection_model_cnn.h5')
        self.melody_machine = pipeline(model="MelodyMachine/Deepfake-audio-detection-V2")

    def build_vgg16_model(self):
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape)
        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        output = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=base_model.input, outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.0001),
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        return model

    def audio_to_spectrogram(self, file_path, plot=False):
        try:
            audio, sr = librosa.load(file_path, duration=5.0, sr=22050)
            spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=224, fmax=8000)
            spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

            if plot:
                plt.figure(figsize=(12, 6))
                librosa.display.specshow(spectrogram_db, y_axis='mel', x_axis='time', cmap='viridis')
                plt.colorbar(format='%+2.0f dB')
                plt.title('Mel Spectrogram Analysis')
                plot_path = 'spectrogram_plot.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                return plot_path

            spectrogram_norm = (spectrogram_db - spectrogram_db.min()) / (spectrogram_db.max() - spectrogram_db.min())
            spectrogram_rgb = np.stack([spectrogram_norm]*3, axis=-1)
            spectrogram_resized = tf.image.resize(spectrogram_rgb, (224, 224))
            return preprocess_input(spectrogram_resized * 255)

        except Exception as e:
            print(f"Spectrogram error: {e}")
            return None

    def analyze_audio_rf(self, audio_path, model_choice="all"):
        results = {}
        plots = {}
        r = []
        audio_features = {}

        try:
            # Load audio and extract basic features
            audio, sr = librosa.load(audio_path, res_type="kaiser_fast")
            audio_features = {
                "sample_rate": sr,
                "duration": librosa.get_duration(y=audio, sr=sr),
                "rms_energy": float(np.mean(librosa.feature.rms(y=audio))),
                "zero_crossing_rate": float(np.mean(librosa.feature.zero_crossing_rate(y=audio)))
            }

            # VGG16 Analysis
            if model_choice in ["VGG16", "all"]:
                spec = self.audio_to_spectrogram(audio_path)
                if spec is not None:
                    pred = self.vgg_model.predict(np.expand_dims(spec, axis=0))[0][0]
                    results["VGG16"] = {
                        "prediction": "FAKE" if pred > 0.5 else "REAL",
                        "confidence": float(pred if pred > 0.5 else 1 - pred),
                        "raw_score": float(pred)
                    }
                    plots["spectrogram"] = self.audio_to_spectrogram(audio_path, plot=True)
                    r.append("FAKE" if pred > 0.5 else "REAL")

            # Dense Model Analysis
            if model_choice in ["Dense", "all"]:
                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
                mfcc_scaled = np.mean(mfcc.T, axis=0).reshape(1, -1)
                pred = self.dense_model.predict(mfcc_scaled)
                results["Dense"] = {
                    "prediction": "FAKE" if np.argmax(pred[0]) == 0 else "REAL",
                    "confidence": float(np.max(pred[0])),
                    "raw_scores": pred[0].tolist()
                }
                r.append("FAKE" if np.argmax(pred[0]) == 0 else "REAL")

            # CNN Model Analysis
            if model_choice in ["CNN", "all"]:
                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
                mfcc_scaled = np.mean(mfcc.T, axis=0).reshape(1, 40, 1, 1)
                pred = self.cnn_model.predict(mfcc_scaled)
                results["CNN"] = {
                    "prediction": "FAKE" if np.argmax(pred[0]) == 0 else "REAL",
                    "confidence": float(np.max(pred[0])),
                    "raw_scores": pred[0].tolist()
                }
                r.append("FAKE" if np.argmax(pred[0]) == 0 else "REAL")

            # Melody Machine Analysis
            if model_choice in ["MelodyMachine", "all"]:
                result = self.melody_machine(audio_path)
                best_pred = max(result, key=lambda x: x['score'])
                results["MelodyMachine"] = {
                    "prediction": best_pred['label'].upper(),
                    "confidence": float(best_pred['score']),
                    "all_predictions": result
                }
                r.append(best_pred['label'].upper())

            return r

        except Exception as e:
            print(f"Analysis error: {e}")
            return None, None, None