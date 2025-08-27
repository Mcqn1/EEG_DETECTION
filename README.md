# EEG Seizure Detection using Machine Learning :

This project provides a complete, end-to-end pipeline to train a machine learning model for seizure detection from EEG data and serve it through a beautiful, interactive web interface built with Streamlit. The system is robustly designed to handle diverse EEG datasets in `.edf` format, automatically standardizing different channel configurations and sampling rates.



## ✨ Features

-   **Dynamic Data Handling**: Automatically processes `.edf` files with different channel montages and sampling rates.
-   **Automated Feature Extraction**: Extracts key statistical and frequency-based features from EEG signals.
-   **Machine Learning Pipeline**: Uses `scikit-learn` to train a Support Vector Machine (SVM) model.
-   **Interactive Web UI**: A modern, animated dark-themed web interface built with **Streamlit**.
-   **Clean & Organized Code**: A structured project with separate scripts for training and for the web application.

---

## 🛠️ Technologies Used

This project is built with a powerful stack of data science and web development tools:

-   **Backend & ML**:
    -   ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
    -   **MNE-Python**: For EEG data processing and feature extraction.
    -   **Scikit-learn**: For training the Support Vector Machine (SVM) model.
    -   **NumPy**: For numerical operations and data manipulation.
    -   **SciPy**: For signal processing tasks like STFT.
-   **Frontend & UI**:
    -   ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
    -   **HTML/CSS**: For custom styling of the Streamlit interface.

---

## 🚀 Setup and Installation

### Prerequisites

-   Python 3.8 or higher
-   `pip` (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Mcqn1/SEIZURE_DETECTION_MODEL.git][(https://github.com/Mcqn1/SEIZURE_DETECTION_MODEL.git)]
    cd SEIZURE_DETECTION_MODEL
    ```

2.  **Install the required Python packages:**
    ```bash
    pip install streamlit mne numpy scikit-learn joblib scipy
    ```

3.  **Prepare the Data:**https://github.com/Mcqn1/SEIZURE_DETECTION_MODEL/wiki
    -   Download the required datasets (see Data Acknowledgement section).
    -   Place the patient folders inside the `EEG_DATA` directory.

---
## ⚙️ How to Use

### 1. Train the Model

Before running the web app, you must train the model.

-   **Run the training script from your terminal:**
    ```bash
    python EEG_TRAIN_MODEL.py
    ```
-   This will create two essential files in the `UTIL_DYNAMIC` folder:
    -   `dynamic_svc_model.pkl`: The trained model.
    -   `common_channels.txt`: The list of channels the model was trained on.

### 2. Run the Web Application

Once the model is trained, launch the web UI.

1.  **Start the Streamlit server:**
    ```bash
    streamlit run streamlit_app.py
    ```
    *(If needed, use `python -m streamlit run streamlit_app.py`)*

2.  **Open your web browser** to the local URL provided in the terminal (usually `http://localhost:8501`).

3.  **Analyze a file** by uploading it and clicking the "Analyze" button.

---
## Data Acknowledgement & Resources

This project utilizes publicly available EEG datasets. Proper credit is given to the researchers and institutions who collected and shared the data.

-   **CHB-MIT Scalp EEG Database**:
    -   **Source**: [PhysioNet - CHB-MIT Database](https://physionet.org/content/chbmit/1.0.0/)
    -   **Citation**: Shoeb, A. H., & Guttag, J. V. (2010). Application of machine learning to epileptic seizure detection. *27th International Conference on Machine Learning (ICML)*.
