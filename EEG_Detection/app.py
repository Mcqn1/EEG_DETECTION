import os
import warnings
import numpy as np
import mne
from scipy.signal import stft
from scipy.stats import kurtosis, skew
import joblib
import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="EEG Seizure Detection",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- Suppress Warnings ---
warnings.filterwarnings("ignore")
np.random.seed(42)

# --- Paths and Constants ---
MODEL_DIR = "UTIL_DYNAMIC"
ML_MODEL_PATH = os.path.join(MODEL_DIR, "dynamic_svc_model.pkl")
CHANNELS_LIST_PATH = os.path.join(MODEL_DIR, "common_channels.txt")
TARGET_SFREQ = 256
UPLOAD_FOLDER = "uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Function to load CSS from an external file ---
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Caching the Model and Channels ---
@st.cache_resource
def load_model_and_channels():
    if not os.path.exists(ML_MODEL_PATH) or not os.path.exists(CHANNELS_LIST_PATH):
        return None, None
    model = joblib.load(ML_MODEL_PATH)
    with open(CHANNELS_LIST_PATH, "r") as f:
        channels = [line.strip() for line in f]
    return model, channels

# --- Helper Functions ---
def load_and_standardize_raw(edf_path, target_sfreq):
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose='ERROR')
        raw.rename_channels(lambda ch: ch.strip().upper())
        if raw.info['sfreq'] != target_sfreq:
            raw.resample(target_sfreq, verbose='ERROR')
        return raw
    except Exception as e:
        return f"Error reading EDF file: {e}"

def compute_features(epoch, sfreq):
    feats = []
    for ch_data in epoch:
        mean, std, sk, ku = np.mean(ch_data), np.std(ch_data), skew(ch_data), kurtosis(ch_data)
        f, _, Zxx = stft(ch_data, fs=sfreq, nperseg=int(sfreq))
        Pxx = np.abs(Zxx)**2
        bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30)}
        band_powers = [np.sum(Pxx[np.where((f >= b[0]) & (f <= b[1]))[0], :]) for b in bands.values()]
        feats.extend([mean, std, sk, ku, *band_powers])
    return np.array(feats)

# --- Main Prediction Logic ---
def get_prediction(edf_path, model, common_channels):
    raw_or_error = load_and_standardize_raw(edf_path, TARGET_SFREQ)
    if isinstance(raw_or_error, str):
        return raw_or_error, None
    raw = raw_or_error

    if not set(common_channels).issubset(raw.ch_names):
        missing = set(common_channels) - set(raw.ch_names)
        return f"Error: The EDF file is missing required channels. Missing: {list(missing)}", None

    raw.pick(common_channels, verbose='ERROR')
    raw.filter(0.5, 48.0, fir_design="firwin", verbose="ERROR").set_eeg_reference("average", projection=False, verbose="ERROR")

    events = mne.make_fixed_length_events(raw, duration=5)
    epochs = mne.Epochs(raw, events, tmin=0, tmax=5 - 1/TARGET_SFREQ, preload=True, baseline=None, verbose="ERROR")
    
    X_feats = np.array([compute_features(epoch, TARGET_SFREQ) for epoch in epochs.get_data()])
    predictions = model.predict(X_feats)
    
    seizure_epochs = np.sum(predictions == 1)
    total_epochs = len(predictions)
    
    return seizure_epochs, total_epochs

# --- Streamlit UI ---

load_css("style.css")

st.title("🧠 EEG Seizure Detection")
st.markdown("<p>Analyze EDF files for seizure activity with an advanced machine learning model.</p>", unsafe_allow_html=True)

# Initialize session state variables
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'uploaded_filename' not in st.session_state:
    st.session_state.uploaded_filename = None

# Callback function to clear state when a new file is uploaded
def clear_state_on_new_upload():
    st.session_state.analysis_result = None
    st.session_state.processing = False
    st.session_state.uploaded_filename = st.session_state.get('file_uploader_key')

# We use columns to keep the content centered
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    model, common_channels = load_model_and_channels()

    if model is None or common_channels is None:
        st.error("Model or channel list not found. Please run the training script first.")
    else:
        uploaded_file = st.file_uploader(
            "Drag & Drop or Browse to Upload an EDF File",
            type=['edf'],
            label_visibility="collapsed",
            key='file_uploader_key',
            on_change=clear_state_on_new_upload
        )

        # Show the button only if a file is uploaded and we are not processing or showing results
        if uploaded_file is not None and not st.session_state.processing and st.session_state.analysis_result is None:
            if st.button("Analyze File", key="analyze_button"):
                st.session_state.processing = True
                st.rerun()

        # If processing, show the spinner and perform analysis
        if st.session_state.processing:
            with st.spinner('Analyzing brainwaves... This may take a moment.'):
                temp_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                seizure_epochs, total_epochs = get_prediction(temp_path, model, common_channels)
                
                # Store result and update state
                st.session_state.analysis_result = (seizure_epochs, total_epochs)
                st.session_state.processing = False
                st.rerun()

        # If analysis is complete, show the result
        if st.session_state.analysis_result is not None:
            seizure_epochs, total_epochs = st.session_state.analysis_result
            st.header("Analysis Complete")
            if total_epochs is None:
                st.error(f"❌ {seizure_epochs}")
            elif seizure_epochs > 0:
                percentage = (seizure_epochs / total_epochs) * 100
                #st.error(f"🚨 **SEIZURE DETECTED** in **{seizure_epochs}** of {total_epochs} windows ({percentage:.2f}%).")
                st.error(f"🚨 **SEIZURE DETECTED** ")
            else:
                #st.success(f"✅ **NO SEIZURES DETECTED** in {total_epochs} windows.")
                st.success(f"✅ **NO SEIZURES DETECTED** ")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #8B949E;'>Built with Streamlit and Scikit-learn</p>", unsafe_allow_html=True)