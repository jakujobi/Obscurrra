# obscurrra-v0.5.py

"""
Obscurrra v0.5

License:
    This project is licensed under the HPL 3 License - see the LICENSE file for details.

Author:
    John Akujobi (john@jakujobi.com)

Description:
    Obscurrra v0.5 is a Python-based image processing tool designed to detect and blur faces in images.
    It supports multiple face detection models including MTCNN, Haar Cascades for frontal and profile faces.
    The program processes all images in a specified directory, detects faces using the selected models, 
    and applies a blur effect to the detected faces before saving the modified images to an output directory.
    
Features:
    - Detects faces using MTCNN, Haar Cascades for frontal faces, and Haar Cascades for profile faces.
    - Processes images with various extensions including jpg, jpeg, png, and webp.
    - Resizes images to a maximum specified dimension before processing.
    - Applies a blur effect to detected faces and saves the processed images.
    
Usage:
    The main script runs the `MainProgram` class which handles the directory management and image processing. 
    It can be executed directly to process images in the current directory's test folder.

Dependencies:
    - OpenCV (cv2)
    - MTCNN
    - glob
    - os
    - logging
"""


# main_program.py
#region
# main_program.py

import logging
import os
# from directory_manager import DirectoryManager
# from image_processor import ImageProcessor

class MainProgram:
    OUTPUT_FOLDER_NAME = 'blurred'

    def __init__(self):
        self.directory_manager = DirectoryManager()
        self.image_processor = ImageProcessor()

    def run(self, models):
        try:
            current_directory = self.directory_manager.get_current_directory()
            input_folder = current_directory
            output_folder = os.path.join(current_directory, MainProgram.OUTPUT_FOLDER_NAME)
            self.directory_manager.create_output_directory(output_folder)
            self.image_processor.process_all_images(input_folder, output_folder, models)
        except Exception as e:
            logging.error(f"Error running the program: {e}")
            raise e
#endregion

#image_processor.py
#region
#image_processor.py

import cv2
import glob
import os
import logging
import time
from mtcnn import MTCNN
from concurrent.futures import ThreadPoolExecutor


class Preprocessor:
    @staticmethod
    def read_image(image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Error reading image {image_path}.")
        return img

    @staticmethod
    def resize_image(image, max_dimension):
        height, width = image.shape[:2]
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_size = (int(width * scale), int(height * scale))
            resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
            return resized_image
        return image

    @staticmethod
    def preprocess_image(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        return gray

class FaceDetection:
    FRONT_FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    PROFILE_FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_profileface.xml'

    def __init__(self):
        self._front_face_cascade = self._load_face_detection_model(self.FRONT_FACE_CASCADE_PATH)
        self._profile_face_cascade = self._load_face_detection_model(self.PROFILE_FACE_CASCADE_PATH)
        self._mtcnn_detector = MTCNN()

    @staticmethod
    def _load_face_detection_model(model_path):
        try:
            model = cv2.CascadeClassifier(model_path)
            return model
        except Exception as e:
            logging.error(f"Error loading face detection model: {e}")
            raise e

    @property
    def front_face_cascade(self):
        return self._front_face_cascade

    @property
    def profile_face_cascade(self):
        return self._profile_face_cascade

    @property
    def mtcnn_detector(self):
        return self._mtcnn_detector

    def choose_model(self, models, image, gray_image):
        faces = []
        if 'mtcnn' in models:
            faces.extend(self.detect_faces_mtcnn(image))
        if 'frontalface' in models:
            faces.extend(self.filter_faces(self.detect_faces(gray_image, self.front_face_cascade), faces))
        if 'profileface' in models:
            faces.extend(self.filter_faces(self.detect_faces(gray_image, self.profile_face_cascade), faces))
        return faces

    @staticmethod
    def filter_faces(new_faces, existing_faces):
        filtered_faces = []
        for face in new_faces:
            if not any(FaceDetection.is_same_face(face, existing_face) for existing_face in existing_faces):
                filtered_faces.append(face)
        return filtered_faces

    @staticmethod
    def is_same_face(face1, face2):
        x1, y1, w1, h1 = face1
        x2, y2, w2, h2 = face2
        return abs(x1 - x2) < w1 / 2 and abs(y1 - y2) < h1 / 2

    @staticmethod
    def detect_faces(gray, face_cascade):
        try:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            return faces
        except Exception as e:
            logging.error(f"Error detecting faces: {e}")
            raise e

    def detect_faces_mtcnn(self, image):
        try:
            results = self.mtcnn_detector.detect_faces(image)
            faces = [(result['box'][0], result['box'][1], result['box'][2], result['box'][3]) for result in results]
            return faces
        except Exception as e:
            logging.error(f"Error detecting faces with MTCNN: {e}")
            raise e

class FaceBlurrer:
    BLUR_EFFECT = (70, 70)

    @staticmethod
    def blur_faces(img, faces):
        try:
            for (x, y, w, h) in faces:
                img[y:y+h, x:x+w] = cv2.blur(img[y:y+h, x:x+w], FaceBlurrer.BLUR_EFFECT)
            return img
        except Exception as e:
            logging.error(f"Error blurring faces: {e}")
            raise e

class ImageProcessor:
    IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    _MAX_IMAGE_SIZE = 1000  # Maximum size of the image's largest dimension

    def __init__(self):
        self.preprocessor = Preprocessor()
        self.face_detection = FaceDetection()
        self.face_blurrer = FaceBlurrer()

    @property
    def max_image_size(self):
        return self._MAX_IMAGE_SIZE

    @max_image_size.setter
    def max_image_size(self, value):
        if value > 0:
            self._MAX_IMAGE_SIZE = value
        else:
            raise ValueError("Maximum image size must be greater than 0.")

    @staticmethod
    def get_output_path(image_path, output_folder):
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        return os.path.join(output_folder, f"{name}_b{ext}")

    def process_single_image(self, image_path, output_folder, models):
        try:
            logging.info(f"Processing {image_path}")
            original_img = self.preprocessor.read_image(image_path)
            resized_img = self.preprocessor.resize_image(original_img.copy(), self.max_image_size)  # Resize the copy for face detection
            gray = self.preprocessor.preprocess_image(resized_img)
            faces = self.face_detection.choose_model(models, resized_img, gray)

            # Scale factor for translating face coordinates back to original image size
            scale_factor = max(original_img.shape[:2]) / max(resized_img.shape[:2])

            if faces:
                logging.info(f"Detected {len(faces)} face(s) in {image_path}.")
                # Scale the face coordinates back to original image size
                faces = [(int(x*scale_factor), int(y*scale_factor), int(w*scale_factor), int(h*scale_factor)) for (x, y, w, h) in faces]
                self.face_blurrer.blur_faces(original_img, faces)
            else:
                logging.info(f"No faces detected in {image_path}.")
            output_path = self.get_output_path(image_path, output_folder)
            cv2.imwrite(output_path, original_img)
            logging.info(f"Processed and saved {output_path}")
            return {'faces': len(faces)}
        except Exception as e:
            logging.error(f"Error processing {image_path}: {e}")
            raise e

    def process_all_images(self, input_folder, output_folder, models):
        try:
            start_time = time.time()  # Start timing
            total_faces = 0
            total_images = 0
            with ThreadPoolExecutor() as executor:
                futures = []
                for extension in self.IMAGE_EXTENSIONS:
                    for filename in glob.glob(os.path.join(input_folder, extension)):
                        futures.append(executor.submit(self.process_single_image, filename, output_folder, models))
                for future in futures:
                    result = future.result()  # To catch any exceptions from threads
                    total_faces += result['faces']
                    total_images += 1
            end_time = time.time()  # End timing
            elapsed_time = end_time - start_time  # Calculate elapsed time
            logging.info(f"Face blurring complete. Time taken: {elapsed_time} seconds.")
            logging.info(f"Total images processed: {total_images}")
            logging.info(f"Total faces found: {total_faces}")
        except Exception as e:
            logging.error(f"Error processing all images: {e}")
            raise e

#endregion


#driectory_manager.py
#region
# directory_manager.py

import os
import logging

class DirectoryManager:
    TEST_DIR_SUFFIX = '/test'

    @staticmethod
    def get_current_directory():
        try:
            current_directory = os.path.dirname(os.path.realpath(__file__))
            logging.info(f"Getting current directory: {current_directory}")
            return current_directory + DirectoryManager.TEST_DIR_SUFFIX
        except Exception as e:
            logging.error(f"Error getting current directory: {e}")
            raise e

    @staticmethod
    def create_output_directory(output_folder):
        try:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
                logging.info(f"Output directory created at: {output_folder}")
        except Exception as e:
            logging.error(f"Error creating output directory {output_folder}: {e}")
            raise e
#endregion


#If name part
#region

import logging
from main_program import MainProgram

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    try:
        from Obscurrra import ObscurrraGUI
        app = ObscurrraGUI()
        app.mainloop()
    except ImportError as e:
        logging.error(f"Failed to import GUI: {e}")
        main_program = MainProgram()
        models_to_use = ['mtcnn', 'frontalface', 'profileface']  # Specify which models to use
        main_program.run(models_to_use)

#endregion
