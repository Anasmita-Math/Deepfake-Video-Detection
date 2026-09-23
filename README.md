# 🎭 Deepfake Video Detection with ResNeXt101 + PyTorch

A deepfake video detection system built using **ResNeXt101** and **PyTorch** that analyzes faces extracted from video frames and classifies them as **REAL** or **FAKE**.

The project includes an interactive **Streamlit web application** where users can upload a video, preview it, run deepfake detection, and view the final prediction, confidence score, and frame-by-frame REAL/FAKE probabilities.

The model was trained and evaluated using the **FaceForensics++ (FF++)** dataset.

---

## 📌 Features

- 🎥 Upload videos through a Streamlit web interface
- 👤 Detect faces from sampled video frames using OpenCV Haar Cascade
- 🧠 Classify detected faces using ResNeXt101
- 🔍 Predict whether a video is REAL or FAKE
- 📊 Display prediction confidence
- 📈 Display frame-by-frame REAL/FAKE probabilities
- ⚡ Automatically use GPU when CUDA is available
- 💻 Supports CPU inference
- 🎬 Supports MP4, AVI, MOV, and MKV formats
- 🧹 Automatically handles temporary uploaded video files

---

## 🏗️ Project Architecture

```text
                    Input Video
                        │
                        ▼
              ┌───────────────────┐
              │  Frame Sampling   │
              │   10 Frames       │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │   Face Detection  │
              │   Haar Cascade    │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │ Face Crop + Resize│
              │     128 × 128     │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │   Preprocessing   │
              │ Tensor + Normalize│
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │    ResNeXt101     │
              │      32x8d        │
              └─────────┬─────────┘
                        │
                        ▼
             REAL / FAKE Probability
                        │
                        ▼
              ┌───────────────────┐
              │ Frame Aggregation │
              └─────────┬─────────┘
                        │
                        ▼
             Final Video Prediction
              REAL / FAKE + Confidence
📁 Project Structure
Deepfake-Video-Detection/
│
├── app.py
│   └── Streamlit web application for deepfake detection
│
├── deep-fake-video-5-resnext101.ipynb
│   └── Model training, evaluation, and inference notebook
│
├── requirements.txt
│   └── Python dependencies
│
├── .gitignore
│   └── Files excluded from GitHub
│
├── README.md
│   └── Project documentation
│
└── resnext101_deepfake_faces.pth
    └── Trained model weights

Note: The trained .pth model file is approximately 336 MB and is excluded from normal GitHub commits using .gitignore.

🧠 How the Model Works

The project treats deepfake detection as a face-level image classification problem applied to video.

Step 1 — Video Input

The user uploads a video through the Streamlit interface.

Supported formats:

MP4
AVI
MOV
MKV
Step 2 — Frame Sampling

The application samples 10 evenly spaced frames from the uploaded video.

Step 3 — Face Detection

OpenCV's Haar Cascade frontal-face detector is used to locate faces.

The largest detected face from each sampled frame is selected.

Step 4 — Face Preprocessing

Each detected face is:

Cropped
Resized to 128 × 128
Converted to a PyTorch tensor
Normalized using ImageNet normalization
Step 5 — ResNeXt101 Classification

The processed face is passed through:

ResNeXt101 32x8d

The classifier produces two classes:

REAL
FAKE
Step 6 — Frame-Level Prediction

For every analyzed frame, the application calculates:

REAL probability
FAKE probability
Step 7 — Video-Level Prediction

The frame-level probabilities are aggregated to produce the final video prediction and confidence score.

Example:

Prediction: REAL
Confidence: 80.19%
🖥️ Streamlit Application

The project includes an interactive Streamlit interface.

Video Upload

Users can upload a supported video file.

Video Preview

The uploaded video can be previewed directly inside the application.

Deepfake Detection

Click:

🔍 Detect Deepfake

to start the analysis.

Final Prediction

The application displays:

REAL

or

FAKE

along with a confidence score.

Frame-by-Frame Analysis

The application also displays REAL and FAKE probabilities for each analyzed frame.

🧱 Model Architecture

The model uses:

ResNeXt101 32x8d

as the CNN backbone.

The original training configuration used a pretrained ImageNet ResNeXt101 model with a custom binary classification head.

The classifier head consists of:

nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 2)
)

The output classes are:

0 → REAL
1 → FAKE
📊 Dataset

The project uses the FaceForensics++ (FF++) dataset.

The expected dataset structure is:

FF++/
│
├── real/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
│
└── fake/
    ├── video1.mp4
    ├── video2.mp4
    └── ...

The training experiment used:

200 REAL videos
200 FAKE videos

Ten face frames were extracted from each successfully processed video.

⚙️ Requirements

The project requires:

Python 3.11
PyTorch
Torchvision
Streamlit
OpenCV
NumPy
Pillow

All required packages are listed in:

requirements.txt
🚀 Installation
1. Clone the repository
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd Deepfake-Video-Detection
2. Create a virtual environment
python -m venv .venv
3. Activate the environment
Windows
.venv\Scripts\activate
Linux / macOS
source .venv/bin/activate
4. Install dependencies
pip install -r requirements.txt
▶️ Run the Streamlit Application

Run:

python -m streamlit run app.py

The application will open in your browser.

Usually:

http://localhost:8501
🧪 Using the Application
Open the Streamlit application.
Upload a video.
Preview the uploaded video.
Click 🔍 Detect Deepfake.
Wait for the analysis to finish.
View:
Final REAL/FAKE prediction
Confidence score
Frame-by-frame probabilities
📈 Results

The original training run reported the following validation result:

Reported Validation Accuracy

88.16%

The original experiment used:

200 REAL videos
200 FAKE videos
10 frames per video
ResNeXt101 32x8d
40 training epochs
Frozen ResNeXt101 backbone
Custom classification head
Reported Class Metrics
Class	Precision	Recall	F1-score
REAL	0.85	0.91	0.88
FAKE	0.92	0.86	0.89
Reported Overall Accuracy

88.16%

Note: The 88.16% figure is the result reported by the original training experiment. Performance on new or unseen real-world videos may differ.

🎯 Sample Video-Level Predictions
Video	True Label	Predicted	Confidence
01_02__outside_talking_still_laughing__YVGY8LOK.mp4	FAKE	FAKE	0.98
01__podium_speech_happy.mp4	REAL	REAL	0.97
⚠️ Limitations
1. Limited Dataset

The experiment uses only:

200 REAL + 200 FAKE

The model may not generalize perfectly to all deepfake generation techniques.

2. Haar Cascade Face Detection

The application uses OpenCV Haar Cascade for face detection.

It may be less robust for:

Extreme poses
Occlusions
Low-resolution videos
Poor lighting
3. Frame-Based Classification

The current system classifies sampled faces independently.

It does not explicitly model temporal information between consecutive frames.

4. Real-World Generalization

Performance may vary for videos from different datasets, cameras, compression levels, or deepfake generation methods.

5. Model Confidence

The displayed confidence represents the model's prediction probability and should not be treated as absolute proof that a video is real or fake.

🔮 Future Improvements
Fine-tune deeper ResNeXt101 layers
Add stronger data augmentation
Use more training videos
Use a larger and more diverse dataset
Replace Haar Cascade with RetinaFace, MTCNN, or another modern face detector
Introduce temporal modeling
Use a proper video-level train/validation/test split
Apply learning-rate scheduling
Improve robustness against video compression
Evaluate on completely unseen datasets
Add explainability and visualization for suspicious facial regions
🛠️ Technologies Used
Python
PyTorch
Torchvision
ResNeXt101
OpenCV
NumPy
Streamlit
Jupyter Notebook
FaceForensics++
📌 Workflow
Video Upload
     ↓
Frame Sampling
     ↓
Face Detection
     ↓
Face Cropping
     ↓
Image Preprocessing
     ↓
ResNeXt101
     ↓
REAL / FAKE Probabilities
     ↓
Frame Aggregation
     ↓
Final Video Prediction
     ↓
Confidence + Frame Analysis

📄 Disclaimer

This project is an experimental deepfake detection system developed for educational and research purposes.

Predictions may contain errors, particularly for videos that differ significantly from the training data. The output should not be treated as definitive evidence of whether a video is authentic or manipulated.

📜 License

This project is intended for educational and research purposes.