from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np


def read_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    try:
        with wave.open(str(path), "rb") as wav:
            sample_rate = wav.getframerate()
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            frames = wav.readframes(wav.getnframes())

        if sample_width == 1:
            audio = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
            audio = (audio - 128.0) / 128.0
        elif sample_width == 2:
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 4:
            audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported WAV sample width {sample_width} bytes: {path}")
    except wave.Error:
        from scipy.io import wavfile

        sample_rate, raw_audio = wavfile.read(str(path))
        channels = 1 if raw_audio.ndim == 1 else raw_audio.shape[1]
        audio = raw_audio.astype(np.float32)
        if np.issubdtype(raw_audio.dtype, np.integer):
            max_value = float(np.iinfo(raw_audio.dtype).max)
            audio = audio / max_value

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    audio = audio - float(np.mean(audio))
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak
    return audio.astype(np.float32), sample_rate


def frame_audio(audio: np.ndarray, sample_rate: int, frame_ms: float = 25.0, hop_ms: float = 10.0) -> np.ndarray:
    frame_len = max(1, int(sample_rate * frame_ms / 1000.0))
    hop_len = max(1, int(sample_rate * hop_ms / 1000.0))
    if len(audio) < frame_len:
        audio = np.pad(audio, (0, frame_len - len(audio)))

    starts = np.arange(0, len(audio) - frame_len + 1, hop_len)
    frames = np.stack([audio[start : start + frame_len] for start in starts])
    return frames * np.hanning(frame_len).astype(np.float32)


def zero_crossing_rate(frames: np.ndarray) -> np.ndarray:
    signs = np.signbit(frames)
    return np.mean(signs[:, 1:] != signs[:, :-1], axis=1)


def spectral_features(frames: np.ndarray, sample_rate: int) -> dict[str, float]:
    spectrum = np.abs(np.fft.rfft(frames, axis=1))
    power = spectrum**2
    freqs = np.fft.rfftfreq(frames.shape[1], d=1.0 / sample_rate)
    total = np.maximum(power.sum(axis=1), 1e-12)

    centroid = (power * freqs[None, :]).sum(axis=1) / total
    bandwidth = np.sqrt(((freqs[None, :] - centroid[:, None]) ** 2 * power).sum(axis=1) / total)
    cumulative = np.cumsum(power, axis=1)
    rolloff_idx = np.argmax(cumulative >= 0.85 * total[:, None], axis=1)
    rolloff = freqs[rolloff_idx]
    flatness = np.exp(np.mean(np.log(np.maximum(spectrum, 1e-12)), axis=1)) / np.maximum(
        np.mean(spectrum, axis=1), 1e-12
    )

    return {
        "spectral_centroid_mean": float(np.mean(centroid)),
        "spectral_centroid_std": float(np.std(centroid)),
        "spectral_bandwidth_mean": float(np.mean(bandwidth)),
        "spectral_bandwidth_std": float(np.std(bandwidth)),
        "spectral_rolloff_mean": float(np.mean(rolloff)),
        "spectral_rolloff_std": float(np.std(rolloff)),
        "spectral_flatness_mean": float(np.mean(flatness)),
        "spectral_flatness_std": float(np.std(flatness)),
    }


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int = 26) -> np.ndarray:
    def hz_to_mel(hz: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_min = hz_to_mel(np.array([0.0]))[0]
    mel_max = hz_to_mel(np.array([sample_rate / 2.0]))[0]
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for mel_idx in range(1, n_mels + 1):
        left, center, right = bins[mel_idx - 1], bins[mel_idx], bins[mel_idx + 1]
        if center > left:
            filters[mel_idx - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            filters[mel_idx - 1, center:right] = (right - np.arange(center, right)) / (right - center)
    return filters


def dct_type_2(x: np.ndarray, n_coeffs: int) -> np.ndarray:
    n = x.shape[1]
    basis = np.cos(np.pi / n * (np.arange(n) + 0.5)[:, None] * np.arange(n_coeffs)[None, :])
    return x @ basis


def mfcc_features(frames: np.ndarray, sample_rate: int, n_coeffs: int = 13) -> dict[str, float]:
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    filters = mel_filterbank(sample_rate, frames.shape[1])
    mel_energy = np.maximum(power @ filters.T, 1e-12)
    coeffs = dct_type_2(np.log(mel_energy), n_coeffs)
    features: dict[str, float] = {}
    for idx in range(n_coeffs):
        features[f"mfcc_{idx + 1}_mean"] = float(np.mean(coeffs[:, idx]))
        features[f"mfcc_{idx + 1}_std"] = float(np.std(coeffs[:, idx]))
    return features


def estimate_pitch_track(frames: np.ndarray, sample_rate: int, fmin: float = 70.0, fmax: float = 350.0) -> np.ndarray:
    min_lag = max(1, int(sample_rate / fmax))
    max_lag = min(frames.shape[1] - 1, int(sample_rate / fmin))
    pitches = []
    for frame in frames:
        frame = frame - np.mean(frame)
        energy = np.sum(frame**2)
        if energy < 1e-6:
            continue
        corr = np.correlate(frame, frame, mode="full")[len(frame) - 1 :]
        corr[:min_lag] = 0
        lag = int(np.argmax(corr[: max_lag + 1]))
        if lag <= 0 or corr[lag] / max(corr[0], 1e-12) < 0.25:
            continue
        pitches.append(sample_rate / lag)
    return np.asarray(pitches, dtype=np.float32)


def summarize(values: np.ndarray, prefix: str) -> dict[str, float]:
    if values.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
    }


def extract_audio_features(path: str | Path) -> dict[str, float]:
    audio, sample_rate = read_wav_mono(path)
    frames = frame_audio(audio, sample_rate)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    zcr = zero_crossing_rate(frames)
    pitch = estimate_pitch_track(frames, sample_rate)

    features: dict[str, float] = {
        "duration_sec": float(len(audio) / sample_rate),
        "sample_rate": float(sample_rate),
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "rms_skew": float(_skew(rms)),
        "rms_kurtosis": float(_kurtosis(rms)),
        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr)),
        "pitch_voiced_ratio": float(len(pitch) / max(len(frames), 1)),
    }
    features.update(summarize(pitch, "pitch"))
    features.update(spectral_features(frames, sample_rate))
    features.update(mfcc_features(frames, sample_rate))

    if len(pitch) > 2:
        pitch_diff = np.abs(np.diff(pitch))
        features["jitter_proxy"] = float(np.mean(pitch_diff / np.maximum(pitch[:-1], 1e-12)))
    else:
        features["jitter_proxy"] = 0.0

    if len(rms) > 2:
        rms_diff = np.abs(np.diff(rms))
        features["shimmer_proxy"] = float(np.mean(rms_diff / np.maximum(rms[:-1], 1e-12)))
    else:
        features["shimmer_proxy"] = 0.0

    features["entropy"] = float(_entropy(audio))
    return features


def _skew(values: np.ndarray) -> float:
    std = float(np.std(values))
    if std == 0:
        return 0.0
    return float(np.mean(((values - np.mean(values)) / std) ** 3))


def _kurtosis(values: np.ndarray) -> float:
    std = float(np.std(values))
    if std == 0:
        return 0.0
    return float(np.mean(((values - np.mean(values)) / std) ** 4) - 3.0)


def _entropy(audio: np.ndarray, bins: int = 64) -> float:
    hist, _ = np.histogram(audio, bins=bins, range=(-1.0, 1.0), density=False)
    prob = hist.astype(np.float64) / max(int(hist.sum()), 1)
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob))) if prob.size else 0.0


def label_from_parent(path: Path) -> tuple[str, int]:
    parent = path.parent.name.upper()
    if parent.startswith("PD"):
        return "PD", 1
    if parent.startswith("HC"):
        return "HC", 0
    raise ValueError(f"Cannot infer label from parent folder: {path.parent}")


def subject_id_from_filename(path: Path) -> str:
    stem = path.stem
    if "_" in stem:
        parts = stem.split("_")
        if len(parts) >= 2 and parts[1]:
            return parts[1].split("-")[0]
    if "-" in stem:
        return stem.split("-")[0].replace("AH_", "")
    return stem
