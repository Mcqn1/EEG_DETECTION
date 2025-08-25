import os
import glob
import warnings
import numpy as np
import mne
from datetime import datetime
from scipy.signal import stft
from scipy.stats import kurtosis, skew
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import joblib


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")
np.random.seed(42)


MODEL_DIR = "C://Websites//EEG_Detection//UTIL_DYNAMIC"
ML_MODEL_PATH = os.path.join(MODEL_DIR, "dynamic_svc_model.pkl")
CHANNELS_LIST_PATH = os.path.join(MODEL_DIR, "common_channels.txt") # Define path for the new file
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- Constants ----------------
TARGET_SFREQ = 256  # Hz



def parse_time_to_seconds(time_str):
    """Parses time from HH.MM.SS or seconds string to seconds."""
    if '.' in time_str or ':' in time_str:
        time_str = time_str.replace('.', ':')
        parts = list(map(int, time_str.split(':')))
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return int(time_str)

def get_seizure_annotations(summary_file, file_name):
    """Parses a summary file to get seizure times for a specific edf file."""
    with open(summary_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    file_sections = content.split('File Name:')[1:]
    for section in file_sections:
        if file_name in section:
            lines = section.strip().split('\n')
            num_seizures = 0
            for line in lines:
                if 'Number of Seizures in File' in line:
                    try:
                        num_seizures = int(line.split(':')[-1].strip())
                        break
                    except (ValueError, IndexError):
                        continue
            if num_seizures == 0:
                return None, None, None

            starts, ends = [], []
            if 'Seizure 1 Start Time' in section: # CHB format
                for i in range(1, num_seizures + 1):
                    for line in lines:
                        if f'Seizure {i} Start Time:' in line:
                            starts.append(float(line.split(':')[-1].strip().replace(' seconds', '')))
                        if f'Seizure {i} End Time:' in line:
                            ends.append(float(line.split(':')[-1].strip().replace(' seconds', '')))
            else: # PN format
                reg_start_sec = 0
                for line in lines:
                    if 'Registration start time' in line:
                        reg_start_sec = parse_time_to_seconds(line.split(': ')[1])
                        break
                for line in lines:
                    if 'Seizure start time' in line:
                        starts.append(parse_time_to_seconds(line.split(': ')[1]) - reg_start_sec)
                    if 'Seizure end time' in line:
                        ends.append(parse_time_to_seconds(line.split(': ')[1]) - reg_start_sec)

            onsets = np.array(starts)
            durations = np.array(ends) - onsets
            descriptions = ['seizure'] * len(onsets)
            return onsets, durations, descriptions
    return None, None, None

def load_and_standardize_raw(edf_path, target_sfreq):
    """Loads and standardizes an EDF file."""
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose='ERROR')
        raw.rename_channels(lambda ch: ch.strip().upper())
        raw.pick_types(eeg=True, exclude=['EKG', 'ECG']) # Keep only EEG channels
        if raw.info['sfreq'] != target_sfreq:
            raw.resample(target_sfreq, verbose='ERROR')
        return raw
    except Exception:
        return None

def discover_common_channels(base_data_path, target_sfreq):
    """Finds the intersection of channels across all EDF files."""
    print("[PASS 1] Discovering common channels...")
    patient_folders = [os.path.join(base_data_path, f) for f in os.listdir(base_data_path) if os.path.isdir(os.path.join(base_data_path, f))]
    common_channels = None
    for patient_folder in patient_folders:
        for edf_file in glob.glob(os.path.join(patient_folder, "*.edf")):
            raw = load_and_standardize_raw(edf_file, target_sfreq)
            if raw:
                current_channels = set(raw.ch_names)
                if common_channels is None:
                    common_channels = current_channels
                else:
                    common_channels.intersection_update(current_channels)
    if common_channels:
        print(f"[✓] Found {len(common_channels)} common channels.")
        return sorted(list(common_channels))
    raise RuntimeError("Could not find any common EEG channels across the dataset.")

def process_and_extract_features(base_data_path, common_channels, target_sfreq, window_sec=5):
    """Processes all data and extracts feature vectors and labels."""
    print("\n[PASS 2] Processing data and extracting features...")
    all_X_feats, all_y = [], []
    patient_folders = [os.path.join(base_data_path, f) for f in os.listdir(base_data_path) if os.path.isdir(os.path.join(base_data_path, f))]
    for patient_folder in patient_folders:
        print(f"\n[+] Processing patient: {os.path.basename(patient_folder)}")
        summary_file = next(glob.iglob(os.path.join(patient_folder, "*.txt")), None)
        raws = [load_and_standardize_raw(edf_file, target_sfreq) for edf_file in sorted(glob.glob(os.path.join(patient_folder, "*.edf")))]
        raws = [r for r in raws if r is not None and set(common_channels).issubset(set(r.ch_names))]
        if not raws or not summary_file:
            continue
        
        for raw in raws:
            raw.pick(common_channels, verbose='ERROR')
            onsets, durations, descriptions = get_seizure_annotations(summary_file, os.path.basename(raw.filenames[0]).replace('.edf', ''))
            if onsets is not None:
                raw.set_annotations(mne.Annotations(onset=onsets, duration=durations, description=descriptions))
        
        raw_combined = mne.concatenate_raws(raws)
        raw_combined.filter(0.5, 48.0, fir_design="firwin", verbose="ERROR").set_eeg_reference("average", projection=False, verbose="ERROR")

        events = mne.make_fixed_length_events(raw_combined, duration=window_sec)
        epochs = mne.Epochs(raw_combined, events, tmin=0, tmax=window_sec-1/target_sfreq, preload=True, baseline=None, verbose="ERROR")
        
        y = np.zeros(len(epochs))
        for i, epoch in enumerate(epochs):
            for ann in raw_combined.annotations:
                if ann['description'] == 'seizure' and (epochs.events[i, 0] / target_sfreq) < (ann['onset'] + ann['duration']) and (epochs.events[i, 0] / target_sfreq + window_sec) > ann['onset']:
                    y[i] = 1
                    break
        
        X_feats = np.array([compute_features(epoch, target_sfreq) for epoch in epochs.get_data()])
        all_X_feats.append(X_feats)
        all_y.append(y)
        
        num_seizure_epochs = int(np.sum(y))
        print(f"[✓] Extracted {len(X_feats)} samples. Seizure epochs found: {num_seizure_epochs}" if num_seizure_epochs > 0 else f"[✓] Extracted {len(X_feats)} samples. No seizures found.")

    return np.vstack(all_X_feats), np.concatenate(all_y)

def compute_features(epoch, sfreq):
    """Computes a feature vector for a single epoch."""
    feats = []
    for ch_data in epoch:
        mean, std, sk, ku = np.mean(ch_data), np.std(ch_data), skew(ch_data), kurtosis(ch_data)
        f, _, Zxx = stft(ch_data, fs=sfreq, nperseg=int(sfreq))
        Pxx = np.abs(Zxx)**2
        bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30)}
        band_powers = [np.sum(Pxx[np.where((f >= band[0]) & (f <= band[1]))[0], :]) for band in bands.values()]
        feats.extend([mean, std, sk, ku, *band_powers])
    return np.array(feats)


if __name__ == "__main__":
    BASE_DATA_PATH = "C://Websites//EEG_Detection//EEG_DATA//"
    
    final_common_channels = discover_common_channels(BASE_DATA_PATH, TARGET_SFREQ)
    X_features, y_labels = process_and_extract_features(BASE_DATA_PATH, final_common_channels, TARGET_SFREQ)

    print(f"\n--- Training ---")
    print(f"Total samples: {len(y_labels)}, Class distribution: 0={np.sum(y_labels==0)}, 1={np.sum(y_labels==1)}")

    X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, stratify=y_labels if len(np.unique(y_labels)) > 1 else None, random_state=42)

    model = Pipeline([("scaler", StandardScaler()), ("clf", SVC(probability=True, kernel='rbf', class_weight='balanced', random_state=42))])
    print("\nTraining the SVM model...")
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"\nModel Accuracy on Test Set: {accuracy * 100:.2f}%")

    joblib.dump(model, ML_MODEL_PATH)
    print(f"\n[✓] Trained model saved to: {ML_MODEL_PATH}")

    
    with open(CHANNELS_LIST_PATH, "w") as f:
        for channel in final_common_channels:
            f.write(f"{channel}\n")
    print(f"[✓] Common channels list saved to: {CHANNELS_LIST_PATH}")