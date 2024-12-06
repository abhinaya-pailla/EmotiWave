import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import streamlit as st
from ResEmoteNet import ResEmoteNet

# Initialize device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Emotion labels
emotions = ['happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'neutral']

# Load model
model = ResEmoteNet().to(device)
checkpoint = torch.load(
    r'C:\Users\paill\Coding\Projects_for_resume\Emotion_Detection\best_model.pth', 
    map_location=device
)
model.load_state_dict(checkpoint)
model.eval()

# Transformation pipeline
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Haar Cascade for face detection
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detect emotion
def detect_emotion(image):
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
    scores = probabilities.cpu().numpy().flatten()
    return [round(score, 2) for score in scores]

# Display all emotions
def display_emotions(x, y, w, h, image, scores):
    org = (x + w + 10, y)
    for index, value in enumerate(emotions):
        emotion_str = f'{value}: {scores[index]:.2f}'
        y_offset = org[1] + (index * 30)
        org_with_offset = (org[0], y_offset)
        cv2.putText(image, emotion_str, org_with_offset, cv2.FONT_ITALIC, 1, (0, 0, 0), 2, cv2.LINE_4)

# Process uploaded images
def process_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=8, minSize=(60, 60))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        crop_img = image[y: y + h, x: x + w]
        pil_crop_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
        scores = detect_emotion(pil_crop_img)
        if scores:
            max_index = np.argmax(scores)
            max_emotion = emotions[max_index]
            if scores[max_index] > 0.5:
                cv2.putText(image, max_emotion, (x, y - 10), cv2.FONT_ITALIC, 1, (255, 0, 0), 2, cv2.LINE_4)
            display_emotions(x, y, w, h, image, scores)
    return image

# Process uploaded video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()
    frame_skip = 3
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            processed_frame = process_image(frame)
            stframe.image(processed_frame, channels="BGR", use_column_width=False, width=1000)
        frame_count += 1
    cap.release()

# UI Configuration
st.set_page_config(page_title="𝓔𝓶𝓸𝓽𝓲𝓦𝓪𝓿𝓮: 𝓡𝓲𝓭𝓮 𝓽𝓱𝓮 𝓣𝓲𝓭𝓮 𝓸𝓯 𝓗𝓾𝓶𝓪𝓷 𝓔𝔁𝓹𝓻𝓮𝓼𝓼𝓲𝓸𝓷𝓼😊", page_icon="😊", layout="wide")

# Sidebar with mode selection
st.sidebar.image(r"C:\Users\paill\Coding\Projects_for_resume\Emotion_Detection\Interface_images\Icon.jpg", width=200)
st.sidebar.header("Choose Mode")
option = st.sidebar.radio(
    "",
    ["Home", "Image Upload", "Video Upload", "Real-Time Webcam"],
    index=0
)

# Home Mode
if option == "Home":
    st.markdown(
        "<h1 style='text-align: center; color: #4CAF50; font-size: 40px;'>𝓔𝓶𝓸𝓽𝓲𝓦𝓪𝓿𝓮: 𝓡𝓲𝓭𝓮 𝓽𝓱𝓮 𝓣𝓲𝓭𝓮 𝓸𝓯 𝓗𝓾𝓶𝓪𝓷 𝓔𝔁𝓹𝓻𝓮𝓼𝓼𝓲𝓸𝓷𝓼😊</h1>", 
        unsafe_allow_html=True
    )
    st.image(r"C:\Users\paill\Coding\Projects_for_resume\Emotion_Detection\Interface_images\home.jpg", use_column_width=False, width=1500)

# Image Upload Mode
elif option == "Image Upload":
    st.markdown("<h3 style='text-align: center;color: #007ACC;'>𝑰𝒎𝒂𝒈𝒆 𝑬𝒎𝒐𝒕𝒊𝒐𝒏 𝑫𝒆𝒕𝒆𝒄𝒕𝒊𝒐𝒏</h3>", unsafe_allow_html=True)
    st.image(r"C:\Users\paill\Coding\Projects_for_resume\Emotion_Detection\Interface_images\image.jpg", use_column_width=False, width=1600)
    uploaded_file = st.file_uploader("Choose an Image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        processed_image = process_image(image_np)
        st.image(processed_image, caption="Processed Image with Detected Emotions", use_column_width=False, width=800)

# Video Upload Mode
elif option == "Video Upload":
    st.markdown("<h3 style='text-align: center;color: #007ACC;'>𝑽𝒊𝒅𝒆𝒐 𝑬𝒎𝒐𝒕𝒊𝒐𝒏 𝑫𝒆𝒕𝒆𝒄𝒕𝒊𝒐𝒏</h3>", unsafe_allow_html=True)
    st.image(r"C:\Users\paill\Coding\Projects_for_resume\Emotion_Detection\Interface_images\video.jpg", use_column_width=False, width=1600)
    uploaded_file = st.file_uploader("Choose a Video", type=["mp4", "avi"])
    if uploaded_file is not None:
        temp_file_path = f"temp_video.{uploaded_file.name.split('.')[-1]}"
        with open(temp_file_path, 'wb') as f:
            f.write(uploaded_file.read())
        process_video(temp_file_path)

# Real-Time Webcam Mode
elif option == "Real-Time Webcam":
    st.markdown("<h3 style='text-align: center;color: #007ACC;'>𝑹𝒆𝒂𝒍-𝑻𝒊𝒎𝒆 𝑬𝒎𝒐𝒕𝒊𝒐𝒏 𝑫𝒆𝒕𝒆𝒄𝒕𝒊𝒐𝒏</h3>", unsafe_allow_html=True)
    st.image(r"C:\Users\paill\Coding\Projects_for_resume\Emotion_Detection\Interface_images\webcam.jpg", use_column_width=False, width=1600)
    start_webcam = st.button("Start Webcam")
    stop_webcam = st.button("Stop")

    if start_webcam:
        video_capture = cv2.VideoCapture(0)
        stframe = st.empty()
        counter = 0
        evaluation_frequency = 5

        while not stop_webcam:
            result, video_frame = video_capture.read()
            if not result:
                st.error("Error: Unable to access the webcam.")
                break

            processed_frame = process_image(video_frame)
            stframe.image(processed_frame, channels="BGR", use_column_width=False, width=1000)
            counter += 1
            if counter == evaluation_frequency:
                counter = 0

        video_capture.release()
        cv2.destroyAllWindows()

