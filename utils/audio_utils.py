# This code inspired by : https://github.com/ruohoruotsi/LSTM-Music-Genre-Classification
# And : https://www.kaggle.com/code/ramoliyafenil/proper-eda-lstm-genre-classifier-for-audio-data
import os
from typing import List, Dict, Tuple, Optional
import numpy as np
import librosa
from tqdm import tqdm
import torch

class MFCCFeatureExtractor:
    """
    Extract MFCC + Spectral Centroid + Chroma + Spectral Contrast from audio files with fixed-length segmentation.
    """

    def __init__(
        self,
        n_mfcc: int = 13,
        n_fft: int = 4084,
        hop_length: int = 1024,
        num_segments: int = 10,
    ):
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_segments = num_segments

    def _extract_segment_features(self, segment: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """
        Extract combined features from a segment.

        Returns:
            np.ndarray: Feature matrix with shape [time, 33] or None if error occurs
        """
        try:
            mfcc = librosa.feature.mfcc(
                y=segment, sr=sr, n_mfcc=self.n_mfcc,
                n_fft=self.n_fft, hop_length=self.hop_length, center=False).T

            centroid = librosa.feature.spectral_centroid(
                y=segment, sr=sr,
                n_fft=self.n_fft, hop_length=self.hop_length, center=False).T

            chroma = librosa.feature.chroma_stft(
                y=segment, sr=sr,
                n_fft=self.n_fft, hop_length=self.hop_length, center=False).T

            contrast = librosa.feature.spectral_contrast(
                y=segment, sr=sr,
                n_fft=self.n_fft, hop_length=self.hop_length, center=False).T

            min_len = min(mfcc.shape[0], centroid.shape[0], chroma.shape[0], contrast.shape[0])

            mfcc = mfcc[:min_len]
            centroid = centroid[:min_len]
            chroma = chroma[:min_len]
            contrast = contrast[:min_len]

            combined = np.concatenate((mfcc, centroid, chroma, contrast), axis=1)

        except Exception as e:
            print(f"[Segment Extraction Error] {e}")
            return None
        
        return combined

    def _extract_features(self, signal: np.ndarray, sr: int) -> List[np.ndarray]:
        """
        Split signal into segments and extract combined features.

        Returns:
            List[np.ndarray]: List of shape [time, 33] for each segment
        """
        segment_length = len(signal) // self.num_segments
        features = []

        for i in range(self.num_segments):
            start = i * segment_length
            end = start + segment_length
            segment = signal[start:end]

            if len(segment) == segment_length:
                segment_features = self._extract_segment_features(segment, sr)
                if segment_features is not None:
                    features.append(segment_features)

        return features

    def process_and_export_npy(
        self,
        audio_paths: List[str],
        output_features_path: str,
        output_labels_path: str,
        label_map: Dict[int, str]
    ) -> None:
        """
        Process list of audio files, extract features, and save to .npy files.
        """
        all_features = []
        all_labels = []
        genre_to_label = {v: k for k, v in label_map.items()}
        
        for audio_path in tqdm(audio_paths, desc="Processing audio files"):
            try:
                signal, sr = librosa.load(audio_path, sr=None)
                signal = signal[:660000]

                genre = os.path.basename(os.path.dirname(audio_path))
                label = genre_to_label.get(genre)

                if label is None:
                    print(f"[Warning] Genre not in label map: {genre}")
                    continue

                feature_segments = self._extract_features(signal, sr)

                all_features.extend(feature_segments)
                all_labels.extend([label] * len(feature_segments))

            except Exception as e:
                print(f"[Load Error] {audio_path}: {e}")

        np.save(output_features_path, np.array(all_features, dtype=object))
        np.save(output_labels_path, np.array(all_labels))

        print(f"[INFO] Exported features to {output_features_path}, labels to {output_labels_path}")


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, features: np.ndarray, labels: List[int]):
        """
        Args:
            features (np.ndarray): Array of shape [N, T, F] or [N] if variable-length segments.
            labels (List[int]): List of genre labels.
        """
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        x_np = self.features[idx]
        x = torch.tensor(x_np, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
