# main_program.py

"""
MainProgram v0.4

License:
    This project is licensed under the HPL 3 License - see the LICENSE file for details.

Author:
    John Akujobi (john@jakujobi.com)

Description:
    MainProgram class orchestrates the directory management and image processing tasks for Obscurrra v0.4.
    It initializes the required components and runs the image processing workflow.

Dependencies:
    - logging
    - os
    - directory_manager.DirectoryManager
    - image_processor.ImageProcessor
"""


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
