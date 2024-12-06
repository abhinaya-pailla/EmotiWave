import os
import shutil

def move_images(source_folder, destination_folder):
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                source_path = os.path.join(root, file)
                destination_path = os.path.join(destination_folder, file)
                shutil.move(source_path, destination_path)

# Replace 'test_folder' with the path to your data folder
test_folder = r"C:\Users\paill\Coding\Projects_for_resume\Emotion_Detection\EmotiWave\data"

# Replace 'destination_folder' with the path to your destination folder
destination_folder = r"C:\Users\paill\Coding\Projects_for_resume\Emotion_Detection\EmotiWave\data_preprocessing\processed_dataset"
move_images(test_folder, destination_folder)