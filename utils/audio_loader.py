import librosa
import numpy as np
from math import inf
from typing import List, Dict, Tuple
from tqdm import tqdm 
import os

def load_audio(
    audio_files: List[str],
    label_map: Dict[int, str],
    sampling_rate: int = 16000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a list of audio files, resample them to a target sampling rate,
    truncate all to the length of the shortest file, and segment each into 10 equal parts.

    Args:
        audio_files (List[str]): List of paths to audio files.
        label_map (Dict[int, str]): Mapping of genre names to integer labels.
        sampling_rate (int, optional): Target sampling rate. Defaults to 16000.

    Returns:
        np.ndarray: Array of shape (num_valid_files * 10, segment_length) containing segmented audio waveforms.
        np.ndarray: Array of shape (num_valid_files * 10,) containing labels for each segment.
    """
    min_length = inf
    valid_files = []

    # Find the minimum length among valid audio files
    for file in tqdm(audio_files, desc='Finding minimum length'):
        try:
            waveform, _ = librosa.load(file, sr=sampling_rate)
            min_length = min(min_length, len(waveform))
            valid_files.append(file)
        except:
            print(f'[WARNING] {os.path.basename(file)} is corrupted and will be skipped!')
            continue

    segment_length = min_length // 10
    min_length = segment_length * 10 

    data = np.zeros((len(valid_files) * 10, segment_length), dtype=np.float32)
    labels = np.zeros(len(valid_files) * 10, dtype=np.int32)
    genre_to_label = {v: k for k, v in label_map.items()}

    for idx, file in enumerate(tqdm(valid_files, desc='Loading and segmenting audio')):
        try:
            waveform, _ = librosa.load(file, sr=sampling_rate)
            waveform = waveform[:min_length]

            genre = os.path.basename(os.path.dirname(file))
            label = genre_to_label.get(genre, -1) 

            for segment_idx in range(10):
                start = segment_idx * segment_length
                end = start + segment_length
                data[idx * 10 + segment_idx] = waveform[start:end]
                labels[idx * 10 + segment_idx] = label 
        except:
            print(f'[WARNING] {os.path.basename(file)} encountered an error during segmentation!')
            continue

    return data, labels
