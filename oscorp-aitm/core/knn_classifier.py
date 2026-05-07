import math
import json
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

@dataclass
class KNNMemory:
    data: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    max_samples_per_word: int = 5
    distance_threshold: float = 20.0

    def fetch_from_cloud(self, db):
        """Pull the global gesture memory from Firestore on startup."""
        if not db:
            return
        try:
            docs = db.collection("knn_memory").stream()
            self.data.clear()
            for doc in docs:
                word = doc.id
                samples = doc.to_dict().get("samples", [])
                
                # Convert list of dicts back to list of tuples for the local app
                parsed_samples = []
                for s in samples:
                    if isinstance(s, dict) and "v" in s and "h" in s:
                        parsed_samples.append((s["v"], s["h"]))
                    # Fallback just in case old data format is somehow present
                    elif isinstance(s, (list, tuple)) and len(s) >= 2:
                        parsed_samples.append((s[0], s[1]))
                        
                if parsed_samples:
                    self.data[word] = parsed_samples
        except Exception as e:
            print(f"Error fetching from cloud: {e}")

    def train(self, word: str, vert: float, horiz: float, db=None):
        word = word.strip().lower()
        if not word:
            return
        
        samples = self.data.setdefault(word, [])
        samples.append((vert, horiz))
        if len(samples) > self.max_samples_per_word:
            self.data[word] = samples[-self.max_samples_per_word:]
            
        if db:
            # Firestore fix: Convert list of tuples to list of dicts
            cloud_format = [{"v": s[0], "h": s[1]} for s in self.data[word]]
            db.collection("knn_memory").document(word).set({
                "samples": cloud_format
            })

    def retrain(self, word: str, vert: float, horiz: float, db=None):
        word = word.strip().lower()
        self.data[word] = [(vert, horiz)]
        if db:
            cloud_format = [{"v": vert, "h": horiz}]
            db.collection("knn_memory").document(word).set({
                "samples": cloud_format
            })

    def forget(self, word: str, db=None):
        word = word.strip().lower()
        self.data.pop(word, None)
        if db:
            db.collection("knn_memory").document(word).delete()

    def clear(self, db=None):
        if db:
            for word in list(self.data.keys()):
                db.collection("knn_memory").document(word).delete()
        self.data.clear()

    def predict(self, vert: float, horiz: float) -> Tuple[Optional[str], float]:
        if not self.data:
            return None, 0.0

        best_word: Optional[str] = None
        best_dist = float('inf')

        for word, samples in self.data.items():
            avg_v = sum(s[0] for s in samples) / len(samples)
            avg_h = sum(s[1] for s in samples) / len(samples)
            dist = math.hypot(vert - avg_v, horiz - avg_h)

            if dist < best_dist:
                best_dist = dist
                best_word = word

        if best_dist > self.distance_threshold:
            return None, 0.0

        confidence = max(0.0, 1.0 - best_dist / self.distance_threshold)
        return best_word, round(confidence, 3)

    def to_json(self) -> str:
        return json.dumps({
            "data": {w: list(s) for w, s in self.data.items()},
            "max_samples_per_word": self.max_samples_per_word,
            "distance_threshold": self.distance_threshold,
        }, indent=2)

    @classmethod
    def from_json(cls, raw: str) -> "KNNMemory":
        obj = json.loads(raw)
        mem = cls(
            max_samples_per_word=obj.get("max_samples_per_word", 5),
            distance_threshold=obj.get("distance_threshold", 20.0),
        )
        mem.data = {w: [tuple(s) for s in samples]
                    for w, samples in obj.get("data", {}).items()}
        return mem

    def stats(self) -> dict:
        return {
            "words": len(self.data),
            "total_samples": sum(len(v) for v in self.data.values()),
            "vocabulary": sorted(self.data.keys()),
        }