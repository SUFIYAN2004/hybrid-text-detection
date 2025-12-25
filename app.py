import os
import warnings
import streamlit as st
import time
import joblib
import re
import string
import pickle
import numpy as np

# 1. SILENCE ALL WARNINGS & TENSORFLOW LOGS
warnings.filterwarnings("ignore") 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# TensorFlow (Deep Learning) - Safe Import
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except ImportError:
    tf = None

# NLTK
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# --- PAGE CONFIGURATION ---
st.set_page_config(layout="wide", page_title="AI Verifier | Professional", initial_sidebar_state="collapsed")

# --- ASSET LOADING & CACHING ---
@st.cache_resource
def load_assets():
    for res in ['punkt', 'stopwords', 'wordnet', 'punkt_tab']:
        nltk.download(res, quiet=True)
    
    assets = {
        'ml_model': None, 'vectorizer': None, 
        'dl_model': None, 'tokenizer': None,
        'lemmatizer': WordNetLemmatizer(),
        'stop_words': set(stopwords.words('english')),
        'status': {'ml': False, 'dl': False}
    }

    # Load Engines
    try:
        assets['ml_model'] = joblib.load('models/ai_essay_classifier.pkl')
        assets['vectorizer'] = joblib.load('models/tfidf_vectorizer.pkl')
        assets['status']['ml'] = True
    except: pass

    try:
        if tf:
            assets['dl_model'] = load_model('models/deeplearning_new_version.keras')
            with open('models/tokenizer_version.pkl', 'rb') as f:
                assets['tokenizer'] = pickle.load(f)
            assets['status']['dl'] = True
    except: pass
    
    return assets

assets = load_assets()

# --- UTILITY FUNCTIONS ---
def clean_text(text):
    text = str(text).lower().strip()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    tokens = [assets['lemmatizer'].lemmatize(word) for word in tokens if word not in assets['stop_words']]
    return ' '.join(tokens)

# CALLBACK FOR CLEARING
def clear_text_callback():
    st.session_state["main_input"] = ""

# --- 2. CSS LAYOUT ---
st.html("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        height: 100vh !important;
        overflow: hidden !important;
        margin: 0; padding: 0;
        background-color: #0f172a;
        background-image: 
            radial-gradient(at 0% 0%, hsla(253,16%,7%,1) 0, transparent 50%), 
            radial-gradient(at 100% 0%, hsla(339,49%,30%,1) 0, transparent 50%);
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    [data-testid="stMainBlockContainer"] {
        padding-top: 1.5rem !important;
        padding-bottom: 0rem !important;
        height: 100vh !important;
        display: flex;
        flex-direction: column;
    }

    header, footer { visibility: hidden; height: 0; position: absolute; }

    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 1.2rem;
        padding: 1.2rem;
    }

    div.stButton > button {
        background: linear-gradient(90deg, #2563eb 0%, #3b82f6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4) !important;
        width: 100%; height: 40px !important;
    }

    [data-testid="column"]:nth-of-type(2) button {
        background: rgba(255, 255, 255, 0.05) !important;
        color: #64748b !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: none !important;
    }

    .stTextArea textarea {
        background: transparent !important;
        color: #f8fafc !important;
        border: none !important;
        height: 310px !important;
        font-size: 1.05rem !important;
        resize: none !important;
    }
</style>
""")

# --- 3. UI LAYOUT ---
h1, h2 = st.columns([2, 1])
with h1:
    st.markdown('<div style="display: flex; align-items: center; gap: 10px;"><div style="background: #2563eb; width: 38px; height: 38px; border-radius: 8px; display: flex; align-items: center; justify-content: center;"><span style="color: white; font-size: 20px;">🧠</span></div><h1 style="font-size: 24px; font-weight: 800; color: white; margin: 0;">Detect<span style="color: #3b82f6;">AI</span></h1></div>', unsafe_allow_html=True)
with h2:
    st.markdown("<div style='text-align: right;'><span style='background: #1e293b; color: #64748b; padding: 5px 14px; border-radius: 99px; border: 1px solid #334155; font-size: 11px;'>v2.0 Professional</span></div>", unsafe_allow_html=True)

st.write("")

left, right = st.columns([1.8, 1], gap="medium")

# --- 4. INPUT AREA ---
with left:
    st.markdown('<div class="glass-card" style="padding: 0; border: 1px solid rgba(255,255,255,0.05);">', unsafe_allow_html=True)
    st.markdown('<div style="background: rgba(15, 23, 42, 0.5); padding: 10px 20px; border-bottom: 1px solid rgba(255,255,255,0.05); display: flex; justify-content: space-between;"><div style="display: flex; gap: 6px;"><div style="width: 8px; height: 8px; border-radius: 50%; background: #ef4444;"></div><div style="width: 8px; height: 8px; border-radius: 50%; background: #f59e0b;"></div><div style="width: 8px; height: 8px; border-radius: 50%; background: #10b981;"></div></div><span style="color: #475569; font-size: 9px; font-weight: 800; text-transform: uppercase;">Input Analysis</span></div>', unsafe_allow_html=True)
    
    # Text Area
    user_input = st.text_area("Input Text", placeholder="Paste text here...", label_visibility="collapsed", key="main_input")
    
    # Flattened columns for bottom bar
    f1, f2, f3 = st.columns([2, 1, 1])
    with f1:
        clean_val = user_input.strip()
        word_count = len(clean_val.split()) if clean_val else 0
        st.markdown(f"<p style='color: #475569; padding: 15px 0 0 20px; font-size: 13px;'>⌨️ {word_count} words</p>", unsafe_allow_html=True)
    with f2:
        st.write("")
        # USE CALLBACK INSTEAD OF DIRECT SETTING
        st.button("Clear", use_container_width=True, on_click=clear_text_callback)
    with f3:
        st.write("")
        trigger = st.button("Run Analysis", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Status Bar
    st.write("")
    m1, m2, m3 = st.columns(3)
    ml_stat = "Active" if assets['status']['ml'] else "Offline"
    dl_stat = "Active" if assets['status']['dl'] else "Offline"
    for col, (l, v, c) in zip([m1, m2, m3], [("ML ENGINE", ml_stat, "#22c55e"), ("DL ENGINE", dl_stat, "#a855f7"), ("SPEED", "~0.4s", "#3b82f6")]):
        col.markdown(f"<div class='glass-card' style='text-align: center; padding: 12px;'><p style='color: #475569; font-size: 9px; font-weight: 800; margin:0;'>{l}</p><p style='color: {c}; font-weight: 800; margin:0; font-size:14px;'>{v}</p></div>", unsafe_allow_html=True)

# --- 5. RESULT CARDS ---
with right:
    ml_score, dl_score, ml_label, dl_label = 0, 0, "Waiting...", "Waiting..."
    
    if trigger:
        if not user_input.strip():
            st.toast("⚠️ Empty input detected!", icon="🚫")
        else:
            with st.spinner(" "):
                input_data = user_input.strip()
                if assets['status']['ml']:
                    cleaned = clean_text(input_data)
                    vec = assets['vectorizer'].transform([cleaned])
                    prob = assets['ml_model'].predict_proba(vec)[0][1] * 100
                    ml_label = "AI Generated" if prob > 50 else "Human Written"
                    ml_score = prob if prob > 50 else (100 - prob)
                
                if assets['status']['dl']:
                    seq = assets['tokenizer'].texts_to_sequences([input_data])
                    padded = pad_sequences(seq, maxlen=300, padding='post')
                    prob = assets['dl_model'].predict(padded, verbose=0)[0][0] * 100
                    dl_label = "AI Generated" if prob > 50 else "Human Written"
                    dl_score = prob if prob > 50 else (100 - prob)

    # Render Cards
    for title, score, label, color, icon in [("Machine Learning", ml_score, ml_label, "#22c55e", "👤"), ("Deep Learning", dl_score, dl_label, "#a855f7", "🧠")]:
        status_color = "#4ade80" if "Human" in label else "#f87171" if "AI" in label else "#475569"
        st.markdown(f"""
            <div class="glass-card" style="margin-bottom: 15px; border-left: 4px solid {color if score > 0 else 'transparent'};">
                <p style="color: #475569; font-size: 9px; font-weight: 800; text-transform: uppercase;">{title} Model</p>
                <div style="text-align: center; padding: 10px 0;">
                    <div style="width: 50px; height: 50px; background: #0f172a; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 8px; border: 1px solid #334155;">
                        <span style="font-size: 20px;">{icon if score > 0 else '➖'}</span>
                    </div>
                    <h4 style="color: {status_color}; margin: 0; font-weight: 800; font-size: 18px;">{label}</h4>
                    <p style="color: #475569; font-size: 11px;">Confidence: {score:.1f}%</p>
                    <div style="width: 100%; background: #1e293b; height: 5px; border-radius: 10px; margin-top: 10px; overflow: hidden;">
                        <div style="width: {score}%; height: 100%; background: {status_color}; transition: width 1s ease;"></div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

st.markdown("<p style='text-align: center; color: #1e293b; font-size: 10px; margin-top: auto; padding-bottom: 10px;'>AI Detection System © 2025</p>", unsafe_allow_html=True)