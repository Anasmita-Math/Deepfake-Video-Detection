import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import tempfile
import os
# ============================================================
# 1. STREAMLIT PAGE
# ============================================================
st.set_page_config(
    page_title="Deepfake Video Detector",
    page_icon="🎭",
    layout="centered"
)
# ============================================================
# 2. DEVICE
# ============================================================
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
# ============================================================
# 3. LOAD TRAINED RESNEXT101 MODEL
# ============================================================
@st.cache_resource
def load_model():

    model = models.resnext101_32x8d(
        weights=None
    )
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 2)
    )
    model.load_state_dict(
        torch.load(
            "resnext101_deepfake_faces.pth",
            map_location=device
        )
    )
    model = model.to(device)
    model.eval()
    return model
model = load_model()
# ============================================================
# 4. IMAGE TRANSFORMATION
# ============================================================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])
# ============================================================
# 5. FACE DETECTOR
# ============================================================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)
# ============================================================
# 6. EXTRACT FACES FROM VIDEO
# ============================================================
def extract_faces_from_video(
    video_path,
    frame_count=10,
    output_size=(128, 128)
):
    cap = cv2.VideoCapture(video_path)
    total = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )
    if total <= 0:
        cap.release()
        return []
    step = max(total // frame_count , 1)
    faces = []
    for i in range(frame_count):
        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            i * step
        )
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )
        detections = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5
        )
        if len(detections) > 0:
            # Select largest detected face
            x, y, w, h = max(
                detections,
                key=lambda r: r[2] * r[3]
            )
            face = frame[
                y:y+h,
                x:x+w
            ]
            face = cv2.resize(
                face,
                output_size
            )
            faces.append(face)
    cap.release()
    return faces
# ============================================================
# 7. PREDICT REAL OR FAKE
# ============================================================
def predict_video(
    video_path,
    frame_count=10
):
    faces = extract_faces_from_video(
        video_path,
        frame_count
    )
    if len(faces) == 0:
        raise ValueError(
            "No face could be detected in the video."
        )
    real_probs = []
    fake_probs = []
    model.eval()
    with torch.no_grad():
        for face in faces:
            face_tensor = transform(face)
            face_tensor = face_tensor.unsqueeze(0)
            face_tensor = face_tensor.to(device)
            output = model(face_tensor)
            probs = torch.softmax(
                output,
                dim=1
            )
            real_probs.append(
                probs[0, 0].item()
            )
            fake_probs.append(
                probs[0, 1].item()
            )
    avg_real = (
        sum(real_probs)
        / len(real_probs)
    )
    avg_fake = (
        sum(fake_probs)
        / len(fake_probs)
    )
    if avg_fake > avg_real:
        label = "FAKE"
        confidence = avg_fake
    else:
        label = "REAL"
        confidence = avg_real
    return (
        label,
        confidence,
        real_probs,
        fake_probs
    )
# ============================================================
# 8. USER INTERFACE
# ============================================================
st.title("🎭 Deepfake Video Detector")
st.write(
    "Upload a video and the trained ResNeXt101 "
    "model will analyze detected faces and predict "
    "whether the video is REAL or FAKE."
)
st.info(
    f"Running on: {device}"
)
uploaded_file = st.file_uploader(
    "🎥 Upload a video",
    type=[
        "mp4",
        "avi",
        "mov",
        "mkv"
    ]
)
if uploaded_file is not None:
    st.video(uploaded_file)
    if st.button(
        "🔍 Detect Deepfake",
        type="primary"
    ):
        file_extension = os.path.splitext(
            uploaded_file.name
        )[1]
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_extension
        ) as temp_file:

            temp_file.write(
                uploaded_file.read()
            )
            temp_video_path = temp_file.name

        try:
            with st.spinner(
                "🔍 Analyzing video... "
                "Please wait."
            ):
                (
                    label,
                    confidence,
                    real_probs,
                    fake_probs
                ) = predict_video(
                    temp_video_path
                )
            st.success(
                "✅ Video analysis completed!"
            )
            st.subheader(
                "Prediction"
            )
            if label == "REAL":
                st.success(
                    f"✅ REAL\n\n"
                    f"Confidence: "
                    f"{confidence * 100:.2f}%"
                )
            else:
                st.error(
                    f"⚠️ FAKE\n\n"
                    f"Confidence: "
                    f"{confidence * 100:.2f}%"
                )
            st.subheader(
                "📊 Frame-by-Frame Analysis"
            )
            for i, (
                real,
                fake
            ) in enumerate(
                zip(
                    real_probs,
                    fake_probs
                ),
                start=1
            ):
                st.write(
                    f"**Frame {i}** — "
                    f"REAL: {real * 100:.2f}% | "
                    f"FAKE: {fake * 100:.2f}%"
                )
                st.progress(
                    max(real, fake)
                )
        except Exception as e:
            st.error(
                f"❌ Error while processing "
                f"video: {e}"
            )
        finally:
            if os.path.exists(
                temp_video_path
            ):
                os.remove(
                    temp_video_path
                )