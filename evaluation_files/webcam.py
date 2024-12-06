import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
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
    rounded_scores = [round(score, 2) for score in scores]
    return rounded_scores

# Start webcam and process frames
def process_webcam():
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Unable to access the webcam.")
        return

    evaluation_frequency = 5  # Evaluate every 5 frames
    counter = 0
    max_emotion = ""

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Unable to read from the webcam.")
            break

        # Convert to grayscale and detect faces
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

        for (x, y, w, h) in faces:
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Crop the face for emotion detection
            cropped_face = frame[y:y + h, x:x + w]
            pil_image = Image.fromarray(cropped_face)

            if counter == 0:
                scores = detect_emotion(pil_image)
                max_index = np.argmax(scores)
                max_emotion = emotions[max_index]

            # Display the primary emotion and its confidence
            cv2.putText(
                frame,
                f"{max_emotion} ({scores[max_index]:.2f})",(x, y - 10),cv2.FONT_ITALIC,0.8,(255, 0, 0),2,cv2.LINE_4
            )

            # Display all emotions with probabilities
            org = (x + w + 10, y)
            for index, value in enumerate(emotions):
                emotion_str = f'{value}: {scores[index]:.2f}'
                cv2.putText(
                    frame,emotion_str,org,cv2.FONT_ITALIC,0.8,(0, 0, 0),2,cv2.LINE_4
                )
                org = (org[0], org[1] + 30)

        # Display the frame
        cv2.imshow("Real-Time Emotion Detection", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Update counter for evaluation frequency
        counter += 1
        if counter == evaluation_frequency:
            counter = 0

    video_capture.release()
    cv2.destroyAllWindows()

# Run the webcam processing
if __name__ == "__main__":
    process_webcam()
