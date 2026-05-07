import cv2
import time
import json
import streamlit as st
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="Silent Speech AI | HACKHIVE-2k26",
    page_icon="🤫",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.ui_utils import (
    apply_accessibility_css, render_hero,
    render_gesture_badge, render_word_strip,
    render_memory_chips, render_suggestions, render_confidence,
)
from core.vision_tracker import VisionTracker, open_camera, GestureState
from core.knn_classifier import KNNMemory
from core.llm_assistant import LLMAssistant, LANGUAGES
from core.cloud_db import get_db

apply_accessibility_css()

# ── Cached singletons ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🧠 Initialising MediaPipe…")
def load_tracker():
    return VisionTracker()

@st.cache_resource(show_spinner="🌐 Connecting to Groq…")
def load_llm():
    return LLMAssistant()

tracker = load_tracker()
llm     = load_llm()
db      = get_db()

# ── Session state bootstrap ───────────────────────────────────────────────────
def _init_session():
    defaults = {
        "word_buffer":      [],          
        "last_sentences":   [],          
        "last_confidence":  0.0,
        "last_word":        "",
        "total_words":      0,
        "total_predictions":0,
        "camera_idx":       None,        
        "train_mode":       False,
        "pending_word":     "",
        "last_gesture":     "READY",
        "session_log":      [],          
        "camera_cap":       None,
    }
    
    if "knn" not in st.session_state:
        st.session_state.knn = KNNMemory()
        if db:
            st.session_state.knn.fetch_from_cloud(db)
            
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session()

# ── Sidebar ───────────────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown('<div class="sec-header">🎯 Teach a Word</div>', unsafe_allow_html=True)

        new_word = st.text_input(
            "Word to teach",
            placeholder="e.g.  water, help, yes",
            label_visibility="collapsed",
        ).strip().lower()

        col_a, col_b = st.columns(2)
        activate_train = col_a.button("✍️ Train", use_container_width=True, type="primary",
                                       help="Raise ☝️ after clicking, mouth the word, lower finger to save")
        retrain_btn    = col_b.button("🔄 Retrain", use_container_width=True,
                                       help="Overwrite existing samples for this word")

        if activate_train and new_word:
            st.session_state.train_mode   = True
            st.session_state.pending_word = new_word
            st.success(f"Ready to train **{new_word}** — raise ☝️ and mouth it!")
        elif activate_train:
            st.warning("Type a word first.")

        if retrain_btn and new_word:
            st.session_state.knn.forget(new_word, db=db)
            st.session_state.train_mode   = True
            st.session_state.pending_word = new_word
            st.info(f"Re-training **{new_word}**…")

        st.divider()

        st.markdown('<div class="sec-header">🌍 Language</div>', unsafe_allow_html=True)
        language = st.selectbox("Output language", LANGUAGES, label_visibility="collapsed")

        st.divider()

        st.markdown('<div class="sec-header">⚙️ Settings</div>', unsafe_allow_html=True)

        camera_idx = st.number_input(
            "Camera index (0 = auto)", min_value=0, max_value=9,
            value=0,
            help="Change if you get 'Camera index out of range' on Linux"
        )
        threshold = st.slider(
            "Match sensitivity",
            min_value=5.0, max_value=50.0,
            value=float(st.session_state.knn.distance_threshold),
            step=1.0,
            help="Lower = stricter match required"
        )
        st.session_state.knn.distance_threshold = threshold

        max_samples = st.slider(
            "Samples per word", 1, 10,
            value=st.session_state.knn.max_samples_per_word,
            help="More samples = more robust, but you must train word N times"
        )
        st.session_state.knn.max_samples_per_word = max_samples

        st.divider()

        st.markdown('<div class="sec-header">📚 AI Memory Bank</div>', unsafe_allow_html=True)
        stats = st.session_state.knn.stats()
        st.metric("Words learned", stats["words"], delta=None)
        st.metric("Total samples", stats["total_samples"], delta=None)

        st.markdown(
            render_memory_chips(st.session_state.knn.data),
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)
        if col1.button("🗑️ Clear", use_container_width=True):
            st.session_state.knn.clear(db=db)
            st.session_state.word_buffer = []
            st.rerun()

        if col2.button("💾 Export", use_container_width=True):
            st.download_button(
                label="⬇ Download",
                data=st.session_state.knn.to_json(),
                file_name="knn_memory.json",
                mime="application/json",
            )

        uploaded = st.file_uploader("Import memory", type="json", label_visibility="collapsed")
        if uploaded:
            try:
                imported_knn = KNNMemory.from_json(uploaded.read().decode())
                # Sync imported data to cloud
                for w, samples in imported_knn.data.items():
                    for s in samples:
                        st.session_state.knn.train(w, s[0], s[1], db=db)
                st.success("Memory imported and synced!")
                st.rerun()
            except Exception as e:
                st.error(f"Import error: {e}")

        st.divider()
        llm_status = "🟢 Online (Groq)" if llm.is_online else "🟡 Offline (fallback)"
        st.caption(llm_status)
        db_status = "🟢 Cloud Synced" if db else "🟡 Local Mode (No DB)"
        st.caption(db_status)

    return language, camera_idx


# ── Main layout ───────────────────────────────────────────────────────────────
def main():
    render_hero()

    language, camera_idx = sidebar()

    cam_col, out_col = st.columns([1.3, 1], gap="large")

    with cam_col:
        st.markdown('<div class="sec-header">📷 Live Feed</div>', unsafe_allow_html=True)
        gesture_ph = st.empty()
        video_ph   = st.empty()

        st.markdown("""
        <div style='background:#0D1117;border:1px solid #1E2533;border-radius:10px;padding:12px 16px;margin-top:8px;'>
            <div style='font-size:0.78rem;color:#546E7A;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px;'>Gesture Guide</div>
            <table style='width:100%;font-size:0.88rem;'>
            <tr><td>☝️</td><td style='color:#A5D6A7;'>Index finger</td><td style='color:#546E7A;'>Listen / Train</td></tr>
            <tr><td>✊</td><td style='color:#E1BEE7;'>Fist</td><td style='color:#546E7A;'>Send to Groq</td></tr>
            <tr><td>👊</td><td style='color:#FFCDD2;'>No hand</td><td style='color:#546E7A;'>System idle</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with out_col:
        st.markdown('<div class="sec-header">💬 Recognised Words</div>', unsafe_allow_html=True)
        words_ph = st.empty()

        st.markdown('<div class="sec-header">🤖 Groq Sentence Completion</div>', unsafe_allow_html=True)
        suggestions_ph = st.empty()

        st.markdown('<div class="sec-header">📊 Session Stats</div>', unsafe_allow_html=True)
        stats_ph = st.empty()

        conf_ph  = st.empty()
        status_ph = st.empty()

        buf_c1, buf_c2 = st.columns(2)
        if buf_c1.button("🗑️ Clear words", use_container_width=True):
            st.session_state.word_buffer = []
            st.session_state.last_sentences = []
            st.rerun()
        if buf_c2.button("✊ Translate now", use_container_width=True,
                          help="Same as making a Fist gesture"):
            if st.session_state.word_buffer:
                st.session_state.last_sentences = llm.generate_sentences(
                    st.session_state.word_buffer, language
                )
                st.session_state.word_buffer = []
                st.rerun()

    start = st.sidebar.checkbox("🟢 Start Camera", value=True, key="cam_on")

    if not start:
        video_ph.markdown("""
        <div style='height:300px;display:flex;align-items:center;justify-content:center;
                    border:2px dashed #1E2533;border-radius:12px;color:#37474F;flex-direction:column;gap:12px;'>
            <span style='font-size:3rem;'>📷</span>
            <p style='margin:0;'>Enable camera in sidebar</p>
        </div>
        """, unsafe_allow_html=True)
        _refresh_output(words_ph, suggestions_ph, stats_ph, conf_ph, status_ph)
        return

    cap = st.session_state.camera_cap
    if cap is None or not cap.isOpened():
        cap = open_camera(camera_idx if camera_idx != 0 else None)
        st.session_state.camera_cap = cap

    if cap is None:
        gesture_ph.markdown(render_gesture_badge("ERROR"), unsafe_allow_html=True)
        status_ph.markdown(
            "<p class='status-err'>❌ Camera not found. Try changing the index in Settings.</p>",
            unsafe_allow_html=True
        )
        return

    was_finger_up  = False
    max_vert       = 0.0
    max_horiz      = 0.0
    gesture_locked = False   
    frame_count    = 0

    try:
        while st.session_state.get("cam_on", False):
            ret, frame = cap.read()
            if not ret:
                status_ph.markdown(
                    "<p class='status-err'>⚠ Camera read error — frame dropped</p>",
                    unsafe_allow_html=True
                )
                time.sleep(0.05)
                continue

            frame_count += 1
            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w  = frame.shape[:2]

            face_result, hand_result = tracker.process_frame(rgb)
            lip    = tracker.extract_lip_features(face_result, w, h)
            gesture = tracker.get_gesture(hand_result)

            mode        = "READY"
            overlay_col = (80, 80, 80)   

            if gesture.finger_up:
                is_train = st.session_state.train_mode and bool(st.session_state.pending_word)
                mode     = "TRAINING" if is_train else "LISTEN"
                overlay_col = (0, 140, 255) if is_train else (0, 230, 120)

                if lip.valid:
                    max_vert  = max(max_vert,  lip.vert)
                    max_horiz = max(max_horiz, lip.horiz)

                was_finger_up = True
                gesture_locked = False   

            elif was_finger_up and not gesture.finger_up:
                if max_vert > 5.0:       
                    feats = (max_vert, max_horiz)

                    if st.session_state.train_mode and st.session_state.pending_word:
                        word = st.session_state.pending_word
                        st.session_state.knn.train(word, *feats, db=db)
                        st.session_state.session_log.append(
                            (time.time(), f"Trained (Cloud): {word} V={max_vert:.1f} H={max_horiz:.1f}")
                        )
                        st.session_state.train_mode = False

                    else:
                        predicted, conf = st.session_state.knn.predict(*feats)
                        st.session_state.total_predictions += 1
                        if predicted:
                            st.session_state.word_buffer.append(predicted)
                            if len(st.session_state.word_buffer) > 8:
                                st.session_state.word_buffer.pop(0)
                            st.session_state.last_word       = predicted
                            st.session_state.last_confidence = conf
                            st.session_state.total_words    += 1
                            st.session_state.session_log.append(
                                (time.time(), f"Predicted: {predicted} ({conf:.0%})")
                            )

                max_vert  = 0.0
                max_horiz = 0.0
                was_finger_up = False

            elif gesture.fist and not gesture_locked:
                # ── FIST → Translate ──────────────────────────────────────────
                mode = "TRANSLATE"
                if st.session_state.word_buffer:
                    gesture_locked = True
                    st.toast(f"⏳ Sending '{', '.join(st.session_state.word_buffer)}' to Groq...", icon="🤖")
                    
                    # Call Groq
                    sentences = llm.generate_sentences(st.session_state.word_buffer, language)
                    
                    st.session_state.last_sentences = sentences
                    st.session_state.word_buffer    = []
                    st.session_state.session_log.append(
                        (time.time(), f"Translated → {language}: {sentences[0] if sentences else ''}")
                    )
                    st.toast("✅ Sentences generated!", icon="✨")

            elif not gesture.fist:
                gesture_locked = False

            mode_labels = {
                "READY":    "IDLE — raise finger to speak",
                "LISTEN":   "☝  LISTENING…",
                "TRAINING": f"📝  TRAINING: {st.session_state.pending_word}",
                "TRANSLATE":"✊  TRANSLATING…",
            }
            annotated = tracker.draw_overlay(
                frame, lip, gesture,
                mode_label=mode_labels.get(mode, ""),
                color=overlay_col,
            )

            if lip.valid and gesture.finger_up:
                bar_w = int(min(1.0, max_vert / 60) * 100)
                cv2.rectangle(annotated, (w-110, 10), (w-10, 28), (30,30,30), -1)
                cv2.rectangle(annotated, (w-110, 10), (w-110+bar_w, 28), overlay_col, -1)
                cv2.putText(annotated, "OPEN", (w-110, 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, overlay_col, 1, cv2.LINE_AA)

            gesture_ph.markdown(render_gesture_badge(mode), unsafe_allow_html=True)
            video_ph.image(annotated, channels="BGR", width="stretch")
            _refresh_output(words_ph, suggestions_ph, stats_ph, conf_ph, status_ph)

    finally:
        if cap is not None:
            cap.release()
        st.session_state.camera_cap = None


def _refresh_output(words_ph, suggestions_ph, stats_ph, conf_ph, status_ph):
    words_ph.markdown(
        render_word_strip(st.session_state.word_buffer),
        unsafe_allow_html=True
    )
    suggestions_ph.markdown(
        render_suggestions(st.session_state.last_sentences),
        unsafe_allow_html=True
    )

    w = st.session_state.total_words
    p = st.session_state.total_predictions
    acc = (w / p * 100) if p else 0
    stats_ph.markdown(f"""
    <div style='display:flex;gap:12px;flex-wrap:wrap;'>
        <div style='background:#0D1117;border:1px solid #1E2533;border-radius:8px;padding:10px 16px;flex:1;'>
            <div style='font-size:1.6rem;font-weight:800;color:#4FC3F7;'>{w}</div>
            <div style='font-size:0.72rem;color:#546E7A;text-transform:uppercase;letter-spacing:.08em;'>Words</div>
        </div>
        <div style='background:#0D1117;border:1px solid #1E2533;border-radius:8px;padding:10px 16px;flex:1;'>
            <div style='font-size:1.6rem;font-weight:800;color:#4FC3F7;'>{p}</div>
            <div style='font-size:0.72rem;color:#546E7A;text-transform:uppercase;letter-spacing:.08em;'>Attempts</div>
        </div>
        <div style='background:#0D1117;border:1px solid #1E2533;border-radius:8px;padding:10px 16px;flex:1;'>
            <div style='font-size:1.6rem;font-weight:800;color:#69F0AE;'>{acc:.0f}%</div>
            <div style='font-size:0.72rem;color:#546E7A;text-transform:uppercase;letter-spacing:.08em;'>Match rate</div>
        </div>
        <div style='background:#0D1117;border:1px solid #1E2533;border-radius:8px;padding:10px 16px;flex:1;'>
            <div style='font-size:1.6rem;font-weight:800;color:#CE93D8;'>{len(st.session_state.knn.data)}</div>
            <div style='font-size:0.72rem;color:#546E7A;text-transform:uppercase;letter-spacing:.08em;'>Vocab</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.last_word:
        c = st.session_state.last_confidence
        color = "#69F0AE" if c > 0.7 else "#FFD740" if c > 0.4 else "#FF5252"
        conf_ph.markdown(
            f"<div style='margin-top:6px;'>"
            + render_confidence(f"Last: '{st.session_state.last_word}'", c, color)
            + "</div>",
            unsafe_allow_html=True
        )

    mem = st.session_state.knn.stats()["words"]
    if mem == 0:
        status_ph.markdown(
            "<p class='status-warn'>⚠ Memory empty — teach some words first (sidebar)</p>",
            unsafe_allow_html=True
        )
    else:
        status_ph.markdown(
            f"<p class='status-ok'>✓ System ready · {mem} word(s) in memory · ☝ to speak · ✊ to translate</p>",
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    main()