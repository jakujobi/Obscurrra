# obscurra-v0.1.py

"""
Obscurra v0.1

This script processes images in the current directory to detect and blur faces 
using OpenCV. It scans all .jpg files, detects faces using pre-trained Haar 
cascades, and applies a blur effect to each detected face.

Modules:
    os: Provides a way of using operating system dependent functionality like 
        reading or writing to the file system.
    glob: Used to retrieve files/pathnames matching a specified pattern.
    cv2: OpenCV module for Python, used for image processing.

Functions:
    None

Usage:
    To run the script, execute:
        python obscurra-v0.1.py

    Ensure OpenCV is installed and place .jpg images in the current directory. 
    The script will process these images and save the blurred versions in the 
    'blurred' directory within the current directory.

Dependencies:
    - OpenCV (cv2)
    - glob
    - os

License:
    This project is licensed under the HPL 3 License - see the LICENSE file for details.

Author: John Akujobi
Date: May 17, 2024
"""

import cv2
import glob
import os

# Get the current directory
current_directory = os.getcwd()

# Path to the input and output folders
# input_folder = current_directory
input_folder = current_directory
output_folder = os.path.join(current_directory, 'blurred')

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load pre-trained face detection models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

# Check if the cascade file loaded correctly
if face_cascade.empty():
    print("Error loading cascade file. Check the path to haarcascade_frontalface_default.xml.")
    exit()

# Iterate over all images in the input folder
for filename in glob.glob(os.path.join(input_folder, '*.jpg')):
    try:
        print(f"Processing {filename}")
        # Read the image
        img = cv2.imread(filename)
        
        # Check if image is loaded correctly
        if img is None:
            print(f"Error reading image {filename}. Skipping.")
            continue
        
        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        if len(faces) == 0:
            print(f"No faces detected in {filename}.")
        else:
            print(f"Detected {len(faces)} face(s) in {filename}.")
        
        # Blur faces
        for (x, y, w, h) in faces:
            # Apply blur effect to the face region
            img[y:y+h, x:x+w] = cv2.blur(img[y:y+h, x:x+w], (30, 30))
        
        # Save the modified image to the output folder
        output_path = os.path.join(output_folder, os.path.basename(filename))
        cv2.imwrite(output_path, img)
        print(f"Processed and saved {output_path}")
    
    except Exception as e:
        print(f"Error processing {filename}: {e}")

print("Face blurring complete.")
