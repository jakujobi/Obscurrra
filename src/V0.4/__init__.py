# __init__.py

"""
Obscurrra v0.4 Package

License:
    This project is licensed under the HPL 3 License - see the LICENSE file for details.

Author:
    John Akujobi (john@jakujobi.com)

Description:
    This package contains the modules required for the Obscurrra v0.4 image processing tool.
    It includes modules for directory management, image processing, face detection, and face blurring.

Dependencies:
    - directory_manager.DirectoryManager
    - image_processor.ImageProcessor
    - main_program.MainProgram
"""


from .directory_manager import DirectoryManager
from .image_processor import ImageProcessor
from .main_program import MainProgram