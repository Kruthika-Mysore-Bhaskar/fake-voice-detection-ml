# fake-voice-detection-ml
Machine learning system for detecting AI-generated and spoofed audio using LFCC features and deep learning (CNN). Achieved ~0.18% Equal Error Rate on ASVspoof 2019.


🔐 Fake Voice Detection using Machine Learning

AI-generated voices are increasingly being used for fraud, impersonation, and biometric attacks.
This project presents a machine learning–based system to detect fake (spoofed) audio using advanced feature engineering and deep learning.

🎓 MSc Data Science — Final Dissertation Project

🚀 Project Overview

Voice cloning and synthetic speech technologies have advanced rapidly, making it harder to trust voice-based authentication systems. This project addresses that challenge by building an audio
anti-spoofing system capable of distinguishing genuine human speech from AI-generated audio.
The system is trained and evaluated using the ASVspoof 2019 Logical Access (LA) dataset — a widely used benchmark in biometric security research.


🧠 Problem Statement

Traditional speaker verification systems are vulnerable to:
AI-generated voices (Text-to-Speech)
Voice conversion attacks
Synthetic audio impersonation
These attacks pose serious risks in banking, authentication systems, and fraud prevention.

This project focuses on detecting such attacks with high accuracy, strong generalisation, and real-time feasibility.



🛠️ Approach
🔹 Feature Engineering

Extracted Linear Frequency Cepstral Coefficients (LFCCs)
LFCCs preserve high-frequency artefacts often introduced by synthetic speech
Chosen over MFCCs due to better spoofing sensitivity

🔹 Models Implemented

Convolutional Neural Network (CNN) ✅ Best performer
Convolutional Recurrent Neural Network (CRNN)
Bidirectional LSTM (BiLSTM)
Classical baselines: GMM, SVM, Random Forest

🔹 Evaluation Metrics

Equal Error Rate (EER)
Accuracy, Precision, Recall, F1-score
ROC curves and confusion matrices


📊 Key Results

CNN achieved ~0.18% Equal Error Rate (EER)
Near-perfect separation between genuine and spoofed audio
Generalised better than more complex recurrent models
Real-time capable inference (~0.13 ms per sample)
Implemented a simple interface to test real audio inputs

🧪 System Pipeline

Audio preprocessing (resampling, normalisation, padding)
LFCC feature extraction
Deep learning–based classification
Evaluation using biometric security metrics
Real-time prediction via a simple interface

🧰 Tech Stack

Python
NumPy, SciPy
Librosa (audio processing)
Scikit-learn
TensorFlow / Keras
Matplotlib

📊 Dataset

This project uses the ASVspoof 2019 Logical Access (LA) dataset.
⚠️ Due to licensing restrictions, the dataset is not included in this repository.


🔮 Future Improvements

Transformer-based audio models
Replay attack detection (Physical Access dataset)
Model explainability (saliency maps)
Deployment as REST API or cloud service

The complete MSc dissertation report is available in the /report directory.

👤 Author

Kruthika Mysore Bhaskar
MSc Data Science
Open to Data Science / Machine Learning / AI Security roles

