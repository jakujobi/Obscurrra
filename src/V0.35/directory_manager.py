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
