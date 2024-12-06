<<<<<<< HEAD
# ResEmoteNet: Bridging Accuracy and Loss Reduction in Facial Emotion Recognition

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/resemotenet-bridging-accuracy-and-loss/facial-expression-recognition-on-fer2013)](https://paperswithcode.com/sota/facial-expression-recognition-on-fer2013?p=resemotenet-bridging-accuracy-and-loss)

A new network that helps in extracting facial features and predict the emotion labels.

The emotion labels in this project are:
 - Happiness 😀
 - Surprise 😦
 - Anger 😠
 - Sadness ☹️
 - Disgust 🤢
 - Fear 😨
 - Neutral 😐


## Table of Content:

 - [Installation](#installation)
 - [Usage](#usage)
 - [Checkpoints](#checkpoints)
 - [Results](#results)
 - [License](#license)


## Installation

1. Create a Conda environment.
```bash
conda create --n "fer"
conda activate fer
```

2. Install Python v3.8 using Conda.
```bash
conda install python=3.8
```

3. Clone the repository.
```bash
git clone https://github.com/ArnabKumarRoy02/ResEmoteNet.git
```

4. Install the required libraries.
```bash
pip install -r requirement.txt
```

## Usage

Run the file.
```bash
cd train_files
python ResEmoteNet_train.py
```

## Checkpoints
All of the checkpoint models for FER2013 can be found [here](https://drive.google.com/drive/folders/1Daxa6d1-XFxxpg6dyxYl4V-anfiHwtqK?usp=sharing).

## Results

 - FER2013:
   - Testing Accuracy: **79.79%** (SoTA - 76.82%)


## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
=======
# 
𝓔𝓶𝓸𝓽𝓲𝓦𝓪𝓿𝓮 : 𝓡𝓲𝓭𝓮 𝓽𝓱𝓮 𝓣𝓲𝓭𝓮 𝓸𝓯 𝓗𝓾𝓶𝓪𝓷 𝓔𝔁𝓹𝓻𝓮𝓼𝓼𝓲𝓸𝓷𝓼😊

📖 𝑶𝒗𝒆𝒓𝒗𝒊𝒆𝒘

This project showcases a Facial Emotion Detection System built using the ResEmotiNet model trained on the FER2013 dataset. The system can identify emotions from images, videos, and live webcam feeds, offering real-time emotion recognition functionality.

The system recognizes the following 7 emotions:

Happiness 😀

Surprise 😦

Anger 😠

Sadness ☹️

Disgust 🤢

Fear 😨

Neutral 😐

A Streamlit interface provides an intuitive user experience, allowing users to upload images or videos or process live webcam feeds seamlessly.

🔧 𝑭𝒆𝒂𝒕𝒖𝒓𝒆𝒔

Emotion Detection from Images: Upload an image to detect emotions in real-time.

Emotion Detection from Videos: Process video files to identify emotions in every detected face.

Live Webcam Feed: Use your webcam to detect and classify emotions dynamically.

Streamlit Interface: User-friendly web application for input and visualization.

Haar Cascade Classifier: Efficient face detection in images and videos.

Detailed Emotion Scores: Probability scores for each emotion displayed alongside the highest-confidence prediction.

🖥️ 𝑻𝒆𝒄𝒉𝒏𝒐𝒍𝒐𝒈𝒚 𝑺𝒕𝒂𝒄𝒌

Core Components

Deep Learning Framework: PyTorch

Model: ResEmotiNet (a custom deep learning model)

Dataset: FER2013, containing 7 emotion classes

𝑨𝒅𝒅𝒊𝒕𝒊𝒐𝒏𝒂𝒍 𝑳𝒊𝒃𝒓𝒂𝒓𝒊𝒆𝒔

OpenCV: For image and video processing, and webcam integration

Streamlit: Interactive user interface

Torchvision: For pre-processing and data transformations

🛠️ 𝑯𝒐𝒘 𝑰𝒕 𝑾𝒐𝒓𝒌𝒔

Face Detection:
The Haar Cascade Classifier detects faces in the input image or video frame.

Emotion Classification:
The detected face is passed through the ResEmotiNet model, which predicts emotion probabilities.

Visualization:
The detected emotion (with the highest confidence) and corresponding scores for all emotions are displayed on the output.

📝 𝑼𝒔𝒂𝒈𝒆 𝑰𝒏𝒔𝒕𝒓𝒖𝒄𝒕𝒊𝒐𝒏𝒔

1.Create a Conda environment.

```bash

𝗰𝗼𝗻𝗱𝗮 𝗰𝗿𝗲𝗮𝘁𝗲 --𝗻 "𝗳𝗲𝗿"
𝗰𝗼𝗻𝗱𝗮 𝗮𝗰𝘁𝗶𝘃𝗮𝘁𝗲 𝗳𝗲𝗿

```

2. Install Python v3.8 using Conda.
   
```bash

𝗰𝗼𝗻𝗱𝗮 𝗶𝗻𝘀𝘁𝗮𝗹𝗹 𝗽𝘆𝘁𝗵𝗼𝗻=𝟯.𝟴

```
3. Clone the Repository
   
```bash

𝗴𝗶𝘁 𝗰𝗹𝗼𝗻𝗲 𝗵𝘁𝘁𝗽𝘀://𝗴𝗶𝘁𝗵𝘂𝗯.𝗰𝗼𝗺/𝗮𝗯𝗵𝗶𝗻𝗮𝘆𝗮-𝗽𝗮𝗶𝗹𝗹𝗮/𝗘𝗺𝗼𝘁𝗶𝗪𝗮𝘃𝗲  
𝗰𝗱 𝗘𝗺𝗼𝘁𝗶𝗪𝗮𝘃𝗲

```

4. Install Dependencies

```bash

𝗽𝗶𝗽 𝗶𝗻𝘀𝘁𝗮𝗹𝗹 -𝗿 𝗿𝗲𝗾𝘂𝗶𝗿𝗲𝗺𝗲𝗻𝘁𝘀.𝘁𝘅𝘁

```

5. Run the file.

```bash

𝗰𝗱 𝘁𝗿𝗮𝗶𝗻_𝗳𝗶𝗹𝗲𝘀
𝗽𝘆𝘁𝗵𝗼𝗻 𝘁𝗿𝗮𝗶𝗻.𝗽𝘆

```

6. Run the Application

Launch the Streamlit app:

```bash

𝘀𝘁𝗿𝗲𝗮𝗺𝗹𝗶𝘁 𝗿𝘂𝗻 𝗘𝗺𝗼𝘁𝗶𝗪𝗮𝘃𝗲_𝗮𝗽𝗽.𝗽𝘆

```

🗂️ 𝑰𝒏𝒑𝒖𝒕 𝑴𝒐𝒅𝒆𝒔

Image Upload: Drag and drop an image file into the interface for analysis.

Video Upload: Upload a video to process frame-by-frame emotion detection.

Webcam Input: Activate your webcam and see real-time emotion detection.

📊 𝑫𝒂𝒕𝒂𝒔𝒆𝒕 𝑰𝒏𝒇𝒐𝒓𝒎𝒂𝒕𝒊𝒐𝒏

The FER2013 dataset is widely used for facial expression recognition tasks. 
It includes:
35,887 grayscale images (48x48 pixels each)
7 emotion classes: Happiness, Surprise, Anger, Sadness, Disgust, Fear, and Neutral

💻 𝑺𝒄𝒓𝒆𝒆𝒏𝒔𝒉𝒐𝒕𝒔

Image Input: Emotion detection and probabilities displayed on the processed image.

Video Input: Real-time video frame processing with emotion labels.

Webcam Feed: Live detection with bounding boxes and emotion predictions.

🌟 𝑨𝒄𝒌𝒏𝒐𝒘𝒍𝒆𝒅𝒈𝒆𝒎𝒆𝒏𝒕𝒔

FER2013 Dataset: FER2013 Kaggle

OpenCV Library for efficient image processing

Streamlit for an interactive interface
>>>>>>> 886913494ea0d0dbc178e72b91cc2989d7c0b93f
