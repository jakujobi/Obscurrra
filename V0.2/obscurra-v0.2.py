import cv2
import glob
import os

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
    FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    BLUR_EFFECT = (30, 30)

    @staticmethod
    def load_face_detection_models():
        try:
            face_cascade = cv2.CascadeClassifier(ImageProcessor.FACE_CASCADE_PATH)
            if face_cascade.empty():
                raise ValueError("Error loading cascade file. Check the path to haarcascade_frontalface_default.xml.")
            return face_cascade
        except Exception as e:
            print(f"Error loading face detection models: {e}")
            raise e

    @staticmethod
    def read_image(image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Error reading image {image_path}.")
        return img

    @staticmethod
    def convert_to_gray(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def get_output_path(image_path, output_folder):
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        return os.path.join(output_folder, f"{name}_b{ext}")

    @staticmethod
    def process_single_image(image_path, output_folder, face_cascade):
        try:
            print(f"Processing {image_path}")
            img = ImageProcessor.read_image(image_path)
            gray = ImageProcessor.convert_to_gray(img)
            faces = ImageProcessor.detect_faces(gray, face_cascade)
            if len(faces) > 0:
                print(f"Detected {len(faces)} face(s) in {image_path}.")
                ImageProcessor.blur_faces(img, faces)
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
    def process_all_images(input_folder, output_folder, face_cascade):
        try:
            for extension in ImageProcessor.IMAGE_EXTENSIONS:
                for filename in glob.glob(os.path.join(input_folder, extension)):
                    ImageProcessor.process_single_image(filename, output_folder, face_cascade)
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

            # Load pre-trained face detection models
            face_cascade = self.image_processor.load_face_detection_models()

            # Process all images
            self.image_processor.process_all_images(input_folder, output_folder, face_cascade)
        except Exception as e:
            print(f"Error running the program: {e}")
            raise e

if __name__ == "__main__":
    # Run the main program
    main_program = MainProgram()
    main_program.run()
