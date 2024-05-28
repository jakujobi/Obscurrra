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
import mtcnn
import tensorflow
from mtcnn import MTCNN


class DirectoryManager:
    TEST_DIR_SUFFIX = '/test'

    @staticmethod
    def get_current_directory():
        """
        Get the current directory.

        Returns:
            str: The current directory path.

        Raises:
            Exception: If there is an error getting the current directory.
        """
        try:
            current_directory = os.path.dirname(os.path.realpath(__file__))
            print("Getting current directory..." + current_directory)
            # return current_directory + '/test'
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
    BLUR_EFFECT = (30, 30)
    FRONT_FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    PROFILE_FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_profileface.xml'
    SCALE_FACTOR = 1.05
    MIN_NEIGHBORS = 3
    MIN_SIZE = (30, 30)

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

    @staticmethod
    def detect_faces(gray, face_cascade):
        try:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=ImageProcessor.SCALE_FACTOR, minNeighbors=ImageProcessor.MIN_NEIGHBORS, minSize=ImageProcessor.MIN_SIZE)
            return faces
        except Exception as e:
            print(f"Error detecting faces: {e}")
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
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Equalize histogram
        gray = cv2.equalizeHist(gray)

        # Resize image (if necessary)
        # gray = cv2.resize(gray, (new_width, new_height))

        return gray

    @staticmethod
    def process_single_image(image_path, output_folder, front_face_cascade, profile_face_cascade, mtcnn_detector):
        try:
            print(f"Processing {image_path}")
            img = ImageProcessor.read_image(image_path)
            gray = ImageProcessor.preprocess_image(img)  # Call the new method here
            faces_front = ImageProcessor.detect_faces(gray, front_face_cascade)
            faces_profile = ImageProcessor.detect_faces(gray, profile_face_cascade)

            # Check if both lists are not empty
            if len(faces_front) > 0 and len(faces_profile) > 0:
                faces = list(set(map(tuple, faces_front)) | set(map(tuple, faces_profile)))
                faces = [list(face) for face in faces]
            elif len(faces_front) > 0:
                faces = faces_front
            elif len(faces_profile) > 0:
                faces = faces_profile
            else:
                faces = []

            if len(faces) > 0:
                print(f"Detected {len(faces)} face(s) in {image_path}.")
                # Use MTCNN on the detected areas
                for (x, y, w, h) in faces:
                    face_img = img[y:y+h, x:x+w]
                    result = mtcnn_detector.detect_faces(face_img)
                    if result:  # If MTCNN detects a face
                        ImageProcessor.blur_faces(face_img, [(result[0]['box'][0], result[0]['box'][1], result[0]['box'][2], result[0]['box'][3])])
            else:
                print(f"No faces detected in {image_path}.")
            output_path = ImageProcessor.get_output_path(image_path, output_folder)
            cv2.imwrite(output_path, img)
            print(f"Processed and saved {output_path}")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            raise e

    @staticmethod
    def detect_faces(gray, face_cascade):
        try:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            return faces
        except Exception as e:
            print(f"Error detecting faces: {e}")
            raise e

    @staticmethod
    def blur_faces(img, faces):
        try:
            for (x, y, w, h) in faces:
                # Apply blur effect to the face region
                img[y:y+h, x:x+w] = cv2.blur(img[y:y+h, x:x+w], ImageProcessor.BLUR_EFFECT)
            return img
        except Exception as e:
            print(f"Error blurring faces: {e}")
            raise e

    @staticmethod
    def process_all_images(input_folder, output_folder, front_face_cascade, profile_face_cascade, mtcnn_detector):
        try:
            for extension in ImageProcessor.IMAGE_EXTENSIONS:
                for filename in glob.glob(os.path.join(input_folder, extension)):
                    ImageProcessor.process_single_image(filename, output_folder, front_face_cascade, profile_face_cascade, mtcnn_detector)
            print("Face blurring complete.")
        except Exception as e:
            print(f"Error processing all images: {e}")
            raise e

class MainProgram:
    OUTPUT_FOLDER_NAME = 'blurred'

    def __init__(self):
        self.directory_manager = DirectoryManager()
        self.image_processor = ImageProcessor()

    def run(self):
        try:
            # Get the current directory
            current_directory = self.directory_manager.get_current_directory()

            # Path to the input and output folders
            input_folder = current_directory
            output_folder = os.path.join(current_directory, MainProgram.OUTPUT_FOLDER_NAME)

            # Create the output folder
            self.directory_manager.create_output_directory(output_folder)

            # Process all images
            self.image_processor.process_all_images(input_folder, output_folder, self.image_processor.front_face_cascade, self.image_processor.profile_face_cascade, self.image_processor.mtcnn_detector)
        except Exception as e:
            print(f"Error running the program: {e}")
            raise e

if __name__ == "__main__":
    # Run the main program
    main_program = MainProgram()
    main_program.run()