# 🤫 Silent Speech AI — HACKHIVE-2k26 Demo Guide

## ⚡ Setup (60 seconds)

```bash
pip install -r requirements.txt
echo "GEMINI_API_KEY=your_key_here" > .env
streamlit run app.py
```

---

## 🎬 Live Demo Script for Judges

### Step 1 — Teach a word (15 sec)
1. Type **`water`** in the sidebar text box → click **✍️ Train**
2. Raise ☝️ index finger at the camera
3. Silently mouth **"water"**
4. Lower your finger → word appears in Memory Bank

> Repeat for 2–3 more words: **`help`**, **`yes`**, **`no`**

### Step 2 — Speak silently (20 sec)
1. Raise ☝️ → mouth a trained word → lower finger
2. Watch the word appear in the **Recognised Words** panel
3. Repeat 2–3 times to build a sentence

### Step 3 — Gemini translate (10 sec)
1. Make a ✊ Fist
2. Gemini generates 3 natural sentences in the chosen language
3. Switch language (Hindi / Kannada) and fist again → instant translation

### Bonus — Retrain on the fly
- Judge asks for a random word → type it → click **✍️ Train** → mouth it → done in 5 seconds

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---------|-----|
| `Camera index out of range` | Change **Camera index** slider in Settings (try 1, 2) |
| `qt.qpa.xcb` crash | We use `opencv-python-headless` — no Qt needed |
| Gemini timeout | System auto-falls back to offline templates, no crash |
| Low prediction accuracy | Lower **Match sensitivity** slider, or retrain word |
| Word not detected | Check lip bounding box is green; train the word again |

---

## 🏗️ Architecture (for judges)

```
Camera frame (BGR)
      │
      ▼
MediaPipe FaceLandmarker ──► LipFeatures (vert, horiz)
MediaPipe HandLandmarker ──► GestureState (finger_up, fist…)
      │
      ▼
State Machine
  ├─ finger_up=True  → accumulate max(vert, horiz)
  ├─ finger_up→False → KNN.predict(vert, horiz) → word
  └─ fist=True       → Gemini.generate_sentences(words, language)
      │
      ▼
Streamlit UI (zero hallucination — only predicts on gesture trigger)
```

**Why KNN beats Deep Learning here:**
- ✅ Zero training time per word (one mouth, instant)
- ✅ Zero hallucination (only fires on gesture)
- ✅ Interpretable: distance threshold = transparency
- ✅ Works with 0 GPU, tiny CPU

**Why not the DL model:**
- ❌ Required massive dataset (100K+ samples)
- ❌ ~12% accuracy without that data
- ❌ CUDA issues on WSL/Linux

---

## 📊 Feature Matrix

| Feature | Status |
|---------|--------|
| Real-time lip tracking | ✅ MediaPipe FaceLandmarker |
| Silent speech-to-text | ✅ Few-shot KNN |
| Zero hallucination | ✅ State-triggered |
| Dynamic vocabulary | ✅ Train any word in 3s |
| Gesture recognition | ✅ ☝️ ✊ hand gestures |
| Multi-language | ✅ EN / HI / KN / TE / TA |
| LLM sentence completion | ✅ Gemini 1.5 Flash |
| Offline fallback | ✅ Never crashes |
| Confidence score | ✅ Per-prediction % |
| Memory export/import | ✅ JSON download |
| Accessibility UI | ✅ High-contrast, large text |
