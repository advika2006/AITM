"""
core/dtw_classifier.py
DTW (Dynamic Time Warping) sequence classifier for lip reading.

Why DTW beats single-peak KNN:
  "yes" and "no" may have similar max-opening values,
   but their movement TRAJECTORIES are completely different.
  DTW compares the entire time-series of lip shapes,
   finding the optimal alignment between sequences of different lengths.

Memory layout:
  { "water": [ seq1, seq2, … ],   # each seq = (T, 8) np.array
    "help":  [ seq1 ],
    … }

Prediction:
  1. For each stored word, compute DTW distance against input sequence.
  2. Pick the word with minimum average DTW distance across its samples.
  3. Reject if distance > threshold → returns (None, 0.0).
  4. Confidence = 1 - dist/threshold, mapped to [0,1].
"""
import json
import math
import numpy as np
from typing import Optional, Dict, List, Tuple

# ── Fast DTW (pure numpy, no extra deps) ─────────────────────────────────────

def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute DTW distance between two sequences.
    a: (Ta, D), b: (Tb, D)
    Uses Euclidean distance between feature vectors.
    O(Ta × Tb) time — fast enough for D=8, T<150.
    """
    Ta, D = a.shape
    Tb    = b.shape[0]

    # Cost matrix
    cost = np.full((Ta, Tb), np.inf, dtype=np.float32)
    cost[0, 0] = float(np.linalg.norm(a[0] - b[0]))

    for i in range(1, Ta):
        cost[i, 0] = cost[i-1, 0] + float(np.linalg.norm(a[i] - b[0]))
    for j in range(1, Tb):
        cost[0, j] = cost[0, j-1] + float(np.linalg.norm(a[0] - b[j]))

    for i in range(1, Ta):
        for j in range(1, Tb):
            d = float(np.linalg.norm(a[i] - b[j]))
            cost[i, j] = d + min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1])

    # Normalise by path length
    return cost[Ta-1, Tb-1] / (Ta + Tb)


def _resample(seq: np.ndarray, target_len: int = 30) -> np.ndarray:
    """
    Resample a sequence to a fixed length using linear interpolation.
    Ensures fair DTW comparison regardless of speaking speed.
    """
    T, D = seq.shape
    if T == target_len:
        return seq
    old_idx = np.linspace(0, T-1, T)
    new_idx = np.linspace(0, T-1, target_len)
    out = np.zeros((target_len, D), dtype=np.float32)
    for d in range(D):
        out[:, d] = np.interp(new_idx, old_idx, seq[:, d])
    return out


# ── Classifier ────────────────────────────────────────────────────────────────

class DTWClassifier:
    """
    Few-shot DTW classifier.

    Each word is represented by 1–N stored sequences.
    Prediction = nearest-neighbour in DTW distance space.
    """

    def __init__(
        self,
        max_samples_per_word: int = 5,
        distance_threshold:   float = 0.35,   # reject if avg DTW dist > this
        resample_length:      int   = 30,      # normalise all seqs to this len
        min_frames:           int   = 5,       # ignore very short utterances
    ):
        self.max_samples      = max_samples_per_word
        self.threshold        = distance_threshold
        self.resample_len     = resample_length
        self.min_frames       = min_frames
        self.memory: Dict[str, List[np.ndarray]] = {}

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, word: str, sequence: np.ndarray) -> bool:
        """
        Store a new lip-movement sequence for `word`.
        sequence: (T, 8) array captured during one utterance.
        Returns False if sequence is too short to be useful.
        """
        word = word.strip().lower()
        if not word:
            return False
        if len(sequence) < self.min_frames:
            return False

        seq_r = _resample(sequence, self.resample_len)
        samples = self.memory.setdefault(word, [])
        samples.append(seq_r)
        # Rolling window
        if len(samples) > self.max_samples:
            self.memory[word] = samples[-self.max_samples:]
        return True

    def forget(self, word: str):
        self.memory.pop(word.strip().lower(), None)

    def clear(self):
        self.memory.clear()

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, sequence: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Predict word from a lip-movement sequence.
        Returns (word, confidence) or (None, 0.0).
        """
        if not self.memory or len(sequence) < self.min_frames:
            return None, 0.0

        seq_r = _resample(sequence, self.resample_len)

        best_word = None
        best_dist = float('inf')
        all_dists: Dict[str, float] = {}

        for word, samples in self.memory.items():
            # Average DTW distance across all stored samples for this word
            dists = [_dtw_distance(seq_r, s) for s in samples]
            avg_d = float(np.mean(dists))
            all_dists[word] = avg_d
            if avg_d < best_dist:
                best_dist = avg_d
                best_word = word

        if best_dist > self.threshold:
            return None, 0.0

        # Confidence ∈ [0,1]
        confidence = max(0.0, 1.0 - best_dist / self.threshold)

        # Second-best margin (robustness check)
        sorted_d = sorted(all_dists.values())
        if len(sorted_d) > 1:
            margin = sorted_d[1] - sorted_d[0]
            # If margin is tiny, reduce confidence — ambiguous match
            if margin < 0.02:
                confidence *= 0.6

        return best_word, round(confidence, 3)

    def predict_top_k(self, sequence: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
        """Return top-k predictions sorted by confidence."""
        if not self.memory or len(sequence) < self.min_frames:
            return []

        seq_r = _resample(sequence, self.resample_len)
        results = []

        for word, samples in self.memory.items():
            dists = [_dtw_distance(seq_r, s) for s in samples]
            avg_d = float(np.mean(dists))
            conf  = max(0.0, 1.0 - avg_d / self.threshold)
            results.append((word, conf, avg_d))

        results.sort(key=lambda x: x[2])   # sort by distance
        return [(w, c) for w, c, _ in results[:k] if c > 0]

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "words":        len(self.memory),
            "total_samples": sum(len(v) for v in self.memory.values()),
            "vocabulary":   sorted(self.memory.keys()),
            "threshold":    self.threshold,
            "resample_len": self.resample_len,
        }

    def word_sample_counts(self) -> Dict[str, int]:
        return {w: len(s) for w, s in self.memory.items()}

    # ── Persistence ───────────────────────────────────────────────────────────

    def to_json(self) -> str:
        data = {
            "config": {
                "max_samples":    self.max_samples,
                "threshold":      self.threshold,
                "resample_len":   self.resample_len,
                "min_frames":     self.min_frames,
            },
            "memory": {
                word: [s.tolist() for s in samples]
                for word, samples in self.memory.items()
            }
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, raw: str) -> "DTWClassifier":
        data = json.loads(raw)
        cfg  = data.get("config", {})
        obj  = cls(
            max_samples_per_word=cfg.get("max_samples", 5),
            distance_threshold=  cfg.get("threshold",   0.35),
            resample_length=     cfg.get("resample_len", 30),
            min_frames=          cfg.get("min_frames",   5),
        )
        for word, seqs in data.get("memory", {}).items():
            obj.memory[word] = [np.array(s, dtype=np.float32) for s in seqs]
        return obj