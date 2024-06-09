# obscurrra-v0.3.py

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

import logging
from main_program import MainProgram

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    main_program = MainProgram()
    models_to_use = ['mtcnn', 'frontalface', 'profileface']  # Specify which models to use
    main_program.run(models_to_use)
