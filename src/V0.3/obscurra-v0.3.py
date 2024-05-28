# obscurra-v0.3.py

"""
Obscurra v0.3

License:
    This project is licensed under the HPL 3 License - see the LICENSE file for details.

Author:
    John Akujobi (john@jakujobi.com)
"""

import cv2
import glob
import os
from mtcnn import MTCNN


class DirectoryManager:
    TEST_DIR_SUFFIX = '/test'

    @staticmethod
    def get_current_directory():
        try:
            current_directory = os.path.dirname(os.path.realpath(__file__))
            print("Getting current directory..." + current_directory)
            return current_directory + DirectoryManager.TEST_DIR_SUFFIX
        except Exception as e:
            print(f"Error getting current directory: {e}")
            raise e

    @staticmethod
    def create_output_directory(output_folder):
        try:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
        except Exception as e:
            print(f"Error creating output directory {output_folder}: {e}")
            raise e


class ImageProcessor:
    IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    BLUR_EFFECT = (70, 70)
    FRONT_FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    PROFILE_FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_profileface.xml'
    _MAX_IMAGE_SIZE = 1000  # Maximum size of the image's largest dimension
    
    def __init__(self):
        self._front_face_cascade = self._load_face_detection_model(self.FRONT_FACE_CASCADE_PATH)
        self._profile_face_cascade = self._load_face_detection_model(self.PROFILE_FACE_CASCADE_PATH)
        self._mtcnn_detector = MTCNN()

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
    def _load_face_detection_model(model_path):
        try:
            model = cv2.CascadeClassifier(model_path)
            return model
        except Exception as e:
            print(f"Error loading face detection model: {e}")
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
            faces_mtcnn = self.detect_faces_mtcnn(image)
            faces.extend(faces_mtcnn)
        if 'frontalface' in models:
            faces_frontal = self.detect_faces(gray_image, self.front_face_cascade)
            for face in faces_frontal:
                if not any(self.is_same_face(face, existing_face) for existing_face in faces):
                    faces.append(face)
        if 'profileface' in models:
            faces_profile = self.detect_faces(gray_image, self.profile_face_cascade)
            for face in faces_profile:
                if not any(self.is_same_face(face, existing_face) for existing_face in faces):
                    faces.append(face)
        return faces

    def is_same_face(self, face1, face2):
        x1, y1, w1, h1 = face1
        x2, y2, w2, h2 = face2
        return abs(x1 - x2) < w1 / 2 and abs(y1 - y2) < h1 / 2

    @staticmethod
    def detect_faces(gray, face_cascade):
        try:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            return faces
        except Exception as e:
            print(f"Error detecting faces: {e}")
            raise e

    def detect_faces_mtcnn(self, image):
        try:
            results = self.mtcnn_detector.detect_faces(image)
            faces = []
            for result in results:
                x, y, width, height = result['box']
                faces.append((x, y, width, height))
            return faces
        except Exception as e:
            print(f"Error detecting faces with MTCNN: {e}")
            raise e

    @staticmethod
    def read_image(image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Error reading image {image_path}.")
        return img

    @staticmethod
    def get_output_path(image_path, output_folder):
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        return os.path.join(output_folder, f"{name}_b{ext}")

    @staticmethod
    def preprocess_image(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        return gray
    
    @staticmethod
    def resize_image(image, max_dimension):
        height, width = image.shape[:2]
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_size = (int(width * scale), int(height * scale))
            resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
            return resized_image
        return image

    def process_single_image(self, image_path, output_folder, models):
        try:
            print(f"Processing {image_path}")
            original_img = self.read_image(image_path)
            resized_img = self.resize_image(original_img.copy(), self.max_image_size)  # Resize the copy for face detection
            gray = self.preprocess_image(resized_img)
            faces = self.choose_model(models, resized_img, gray)

            # Scale factor for translating face coordinates back to original image size
            scale_factor = max(original_img.shape[:2]) / max(resized_img.shape[:2])

            if len(faces) > 0:
                print(f"Detected {len(faces)} face(s) in {image_path}.")
                # Scale the face coordinates back to original image size
                faces = [(int(x*scale_factor), int(y*scale_factor), int(w*scale_factor), int(h*scale_factor)) for (x, y, w, h) in faces]
                self.blur_faces(original_img, faces)
            else:
                print(f"No faces detected in {image_path}.")
            output_path = self.get_output_path(image_path, output_folder)
            cv2.imwrite(output_path, original_img)
            print(f"Processed and saved {output_path}")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            raise e

    @staticmethod
    def blur_faces(img, faces):
        try:
            for (x, y, w, h) in faces:
                img[y:y+h, x:x+w] = cv2.blur(img[y:y+h, x:x+w], ImageProcessor.BLUR_EFFECT)
            return img
        except Exception as e:
            print(f"Error blurring faces: {e}")
            raise e

    def process_all_images(self, input_folder, output_folder, models):
        try:
            for extension in self.IMAGE_EXTENSIONS:
                for filename in glob.glob(os.path.join(input_folder, extension)):
                    self.process_single_image(filename, output_folder, models)
            print("Face blurring complete.")
        except Exception as e:
            print(f"Error processing all images: {e}")
            raise e


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
            print(f"Error running the program: {e}")
            raise e


if __name__ == "__main__":
    main_program = MainProgram()
    models_to_use = ['mtcnn', 'frontalface', 'profileface']  # Specify which models to use
    main_program.run(models_to_use)
