import cv2
import glob
import os

class DirectoryManager:
    @staticmethod
    def get_current_directory():
        current_directory = os.path.dirname(os.path.realpath(__file__))
        print("Getting current directory..." + current_directory)
        return current_directory + '/test'
    # def get_current_directory():
    #     print("Getting current directory..." + os.getcwd())
    #     return os.getcwd() + '/'
    
    @staticmethod
    def create_output_directory(output_folder):
        try:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
        except Exception as e:
            print(f"Error creating output directory {output_folder}: {e}")
            raise e


class ImageProcessor:
    @staticmethod
    def load_face_detection_models():
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        profile_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        if face_cascade.empty():
            print("Error loading cascade file. Check the path to haarcascade_frontalface_default.xml.")
            exit()
        return face_cascade

    @staticmethod
    def process_single_image(image_path, output_folder, face_cascade):
        try:
            print(f"Processing {image_path}")
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Error reading image {image_path}.")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = ImageProcessor.detect_faces(gray, face_cascade)
            if len(faces) > 0:
                print(f"Detected {len(faces)} face(s) in {image_path}.")
                ImageProcessor.blur_faces(img, faces)
            else:
                print(f"No faces detected in {image_path}.")
            output_path = os.path.join(output_folder, os.path.basename(image_path))
            cv2.imwrite(output_path, img)
            print(f"Processed and saved {output_path}")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            raise e

    @staticmethod
    def detect_faces(gray, face_cascade):
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces

    @staticmethod
    def blur_faces(img, faces):
        for (x, y, w, h) in faces:
            # Apply blur effect to the face region
            img[y:y+h, x:x+w] = cv2.blur(img[y:y+h, x:x+w], (30, 30))
        return img
    
    @staticmethod
    def process_all_images(input_folder, output_folder, face_cascade):
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
        for extension in extensions:
            for filename in glob.glob(os.path.join(input_folder, extension)):
                ImageProcessor.process_single_image(filename, output_folder, face_cascade)
        print("Face blurring complete.")

class MainProgram:
    def __init__(self):
        self.directory_manager = DirectoryManager()
        self.image_processor = ImageProcessor()

    def run(self):
        # Get the current directory
        current_directory = self.directory_manager.get_current_directory()

        # Path to the input and output folders
        input_folder = current_directory
        output_folder = os.path.join(current_directory, 'blurred')

        # Create the output folder
        self.directory_manager.create_output_directory(output_folder)

        # Load pre-trained face detection models
        face_cascade = self.image_processor.load_face_detection_models()

        # Process all images
        self.image_processor.process_all_images(input_folder, output_folder, face_cascade)

if __name__ == "__main__":
    # Run the main program
    main_program = MainProgram()
    main_program.run()
