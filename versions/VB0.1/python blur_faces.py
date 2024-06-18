import cv2
import glob
import os
import numpy as np

# Get the current directory
current_directory = os.getcwd()

# Path to the input and output folders
input_folder = current_directory
output_folder = os.path.join(current_directory, 'blurred')

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load pre-trained face detection models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

# Load pre-trained DNN face detection model
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Check if the cascade files loaded correctly
if face_cascade.empty() or profile_face_cascade.empty() or net.empty():
    print("Error loading detection models. Check the paths to the model files.")
    exit()

# Function to detect and blur faces using cascades
def detect_and_blur_faces_cascade(image, cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        image[y:y+h, x:x+w] = cv2.blur(image[y:y+h, x:x+w], (30, 30))
    return image

# Function to detect and blur faces using DNN
def detect_and_blur_faces_dnn(image):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            image[y:y1, x:x1] = cv2.blur(image[y:y1, x:x1], (30, 30))
    return image

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
        
        # Detect and blur faces with frontal and profile cascades
        img = detect_and_blur_faces_cascade(img, face_cascade)
        img = detect_and_blur_faces_cascade(img, profile_face_cascade)
        
        # Detect and blur faces using DNN
        img = detect_and_blur_faces_dnn(img)
        
        # Save the modified image to the output folder
        output_path = os.path.join(output_folder, os.path.basename(filename))
        cv2.imwrite(output_path, img)
        print(f"Processed and saved {output_path}")
    
    except Exception as e:
        print(f"Error processing {filename}: {e}")

print("Face blurring complete.")
