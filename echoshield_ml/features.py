import torchaudio
import torch
import numpy as np
from scipy.stats import skew, kurtosis, median_abs_deviation
import os
import torch.nn.functional as F


from torchaudio.pipelines import WAV2VEC2_BASE
bundle = WAV2VEC2_BASE

model = bundle.get_model()
print("Model downloaded successfully!")


def extract_features(file_path):
    if os.path.exists(file_path):
        print(f"File successfully written: {file_path}")
    else:
        print("File writing failed.")
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=bundle.sample_rate)(waveform)

    with torch.inference_mode():
        features, _ = model.extract_features(waveform)

    pooled_features = []
    for f in features:
        if f.dim() == 3:
            f = f.permute(0, 2, 1)
            pooled_f = F.adaptive_avg_pool1d(f[0].unsqueeze(0), 1).squeeze(0)
            pooled_features.append(pooled_f)

    final_features = torch.cat(pooled_features, dim=0).numpy()
    final_features = (final_features - np.mean(final_features)) / (np.std(final_features) + 1e-10)

    return final_features

def additional_features(features):
    mad = median_abs_deviation(features)
    features_clipped = np.clip(features, 1e-10, None)
    entropy = -np.sum(features_clipped * np.log(features_clipped))
    return mad, entropy

def classify_audio(features):

    _, entropy = additional_features(features)
    print(entropy)

    if  entropy > 150:
        return True, entropy
    else:
        return False, entropy