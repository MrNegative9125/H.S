🤟 American Sign Language (ASL) Recognition using TensorFlow & Keras
📌 Project Overview

This project implements an American Sign Language (ASL) Hand Sign Recognition System using Deep Learning with TensorFlow and Keras.
The model is trained to recognize ASL alphabets/hand gestures from images or live camera input, helping bridge communication gaps for the deaf and hard-of-hearing community.

The system uses computer vision techniques and a Convolutional Neural Network (CNN) to accurately classify ASL hand signs.

🚀 Features

🧠 CNN-based Deep Learning model

✋ Recognizes ASL hand signs (alphabets/gestures)

📷 Supports image input and real-time webcam detection

⚡ Fast and accurate predictions

🌐 Deployable using Streamlit

🛠️ Technologies Used

Python 3.x

TensorFlow / Keras

OpenCV

NumPy

Pandas

Matplotlib

Streamlit

📂 Project Structure
ASL-Hand-Sign-Recognition/
│
├── model/
│   └── asl_model_final.keras
├── app.py                  # Streamlit web app
├── D.py                    # Model training script
├── requirements.txt
├── README.md

📊 Dataset

The dataset contains ASL hand sign images, organized by class labels.

Each folder represents an ASL alphabet or gesture.

Images are preprocessed (resizing, normalization, augmentation).

Common datasets used:

ASL Alphabet Dataset (Kaggle)

Custom captured ASL hand images

🧠 Model Architecture

Input Image Layer

Convolution + ReLU Layers

Max Pooling Layers

Fully Connected Dense Layers

Output Layer with Softmax Activation

Training Details:

Loss Function: categorical_crossentropy

Optimizer: Adam

Evaluation Metric: Accuracy

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/MrNegative9125/H.S
cd asl-hand-sign-recognition

2️⃣ Create Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

▶️ Run the Application

To start the Streamlit ASL Recognition App:

streamlit run app.py

📈 Results

High classification accuracy on validation data

Stable real-time ASL predictions

Performs well on unseen hand sign images
