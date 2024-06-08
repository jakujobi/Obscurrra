# src/main_program.py

import logging
import os
from directory_manager import DirectoryManager
from image_processor import ImageProcessor

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
