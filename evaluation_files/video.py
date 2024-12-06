import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from ResEmoteNet import ResEmoteNet

# Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Emotion labels
emotions = ['happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'neutral']

# Load the model
model = ResEmoteNet().to(device)
checkpoint = torch.load(r'C:\Users\paill\Coding\Projects_for_resume\Emotion_Detection\best_model.pth', weights_only=True)
model.load_state_dict(checkpoint)
model.eval()

# Transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load Haar Cascade for face detection
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_classifier.empty():
    raise FileNotFoundError("Haar cascade XML file not loaded. Check the file path.")

# Function to detect emotion
def detect_emotion(image):
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
    scores = probabilities.cpu().numpy().flatten()
    return [round(score, 2) for score in scores]

# Function to detect faces and annotate frame
def detect_bounding_box(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=8, minSize=(60, 60))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        crop_img = frame[y:y + h, x:x + w]
        pil_crop_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
        scores = detect_emotion(pil_crop_img)
        if scores:
            max_index = np.argmax(scores)
            max_emotion = emotions[max_index]
            org = (x, y - 10 if y - 10 > 10 else y + 10)
            cv2.putText(frame, f"{max_emotion}", org, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,cv2.LINE_4)

            # Display all emotion probabilities next to the face
            for i, (emotion, score) in enumerate(zip(emotions, scores)):
                cv2.putText(frame, f"{emotion}: {score:.2f}", (x + w + 10, y + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2,cv2.LINE_4)
    return frame

# Input and processing
file_path = r"C:\Users\paill\Downloads\1122524-hd_1920_1080_25fps.mp4"
cap = cv2.VideoCapture(file_path)

# Process video
frame_skip = 3
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count % frame_skip == 0:
        processed_frame = detect_bounding_box(frame)
        cv2.imshow("Processed Video", processed_frame)
    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
