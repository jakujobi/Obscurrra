# Obscurrra Project Documentation

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Timeline](#timeline)
- [Achievements](#achievements)
- [Challenges and Bugs](#challenges-and-bugs)
- [Methodology](#methodology)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [User Guide](#user-guide)
- [Developer Guide](#developer-guide)
- [Future Improvements](#future-improvements)
- [Contributors](#contributors)

## Project Overview

**Obscurrra** is a software application designed to detect faces in images and apply a blur effect to anonymize them. The project aims to enhance privacy by automatically processing images to obscure identifiable facial features. The application is built using Python and provides a graphical user interface (GUI) for easy interaction.

## Objectives

- Develop an application to detect faces in images using machine learning models.
- Apply a blur effect to detected faces to anonymize them.
- Create a user-friendly GUI for selecting images and managing the processing workflow.
- Ensure the application is portable and can be compiled into a standalone executable.

## Timeline

### Phase 1: Research and Planning (January 2024 - February 2024)

- **January 2024:** Conducted research on face detection models and image processing techniques.
- **February 2024:** Finalized project requirements and defined the scope and objectives.

### Phase 2: Development (March 2024 - June 2024)

- **March 2024:** Started developing the face detection and image processing modules.
- **April 2024:** Implemented the initial version of the GUI using Tkinter.
- **May 2024:** Integrated multiple face detection models (Haar Cascade and MTCNN).
- **June 2024:** Completed basic functionality for selecting images and applying blur effects.

### Phase 3: Testing and Optimization (July 2024)

- **July 2024:** Conducted extensive testing to identify and fix bugs.
- Optimized the performance of the face detection models and image processing workflow.

### Phase 4: Deployment and Documentation (August 2024)

- **August 2024:** Compiled the application into a standalone executable using PyInstaller.
- Created comprehensive documentation for users and developers.

## Achievements

- Successfully developed a functional application for face detection and anonymization.
- Implemented a user-friendly GUI to facilitate image processing tasks.
- Achieved portability by compiling the application into a standalone executable.
- Integrated multiple face detection models to enhance accuracy and flexibility.

## Challenges and Bugs

### Challenges

- **Model Integration:** Integrating multiple face detection models and ensuring compatibility was challenging. The team had to ensure that both Haar Cascade and MTCNN models were properly initialized and used as needed.
- **Performance Optimization:** Optimizing the performance of face detection and image processing required extensive testing and tuning.
- **GUI Design:** Designing a user-friendly and intuitive GUI was crucial for the project's success. The team had to iterate on the design multiple times to improve usability.

### Bugs

1. **MTCNN Model Initialization Failure:** The MTCNN model sometimes failed to initialize due to missing dependencies. The issue was resolved by updating the package requirements and ensuring all necessary files were included.
2. **Image Loading Errors:** Some images failed to load due to unsupported formats or file path issues. The team implemented error handling and provided feedback to users.
3. **Blur Effect Intensity:** Initial versions of the application did not allow users to adjust the blur effect intensity. The GUI was updated to include a slider for this feature.

## Methodology

The development of **Obscurrra** followed an agile methodology, with iterative cycles of development, testing, and optimization. Key steps included:

1. **Research:** Conducted research on existing face detection algorithms and image processing techniques.
2. **Design:** Designed the architecture of the application, including modules for face detection, image processing, and GUI components.
3. **Implementation:** Developed the application using Python and integrated the selected face detection models.
4. **Testing:** Conducted testing to identify and fix bugs, optimize performance, and ensure accuracy.
5. **Deployment:** Compiled the application into an executable and conducted final testing on different platforms.

## Technologies Used

- **Programming Language:** Python
- **GUI Framework:** Tkinter
- **Face Detection Models:** Haar Cascade, MTCNN
- **Image Processing Library:** OpenCV
- **Machine Learning Library:** TensorFlow
- **Executable Packaging:** PyInstaller

## Project Structure

The **Obscurrra** project is organized as follows:

```
Obscurrra/
├── src/
│   ├── Obscurrra.py        # Main application script
│   ├── icon.ico            # Application icon
│   ├── obscuRRRa Logo Full.png  # Full logo image
│   ├── obscuRRRa Logo Monogram.png # Monogram logo image
│   ├── obscuRRRa Profile image.png # Profile logo image
│   ├── haarcascade_frontalface_default.xml # Haar Cascade model for frontal face detection
│   ├── haarcascade_profileface.xml # Haar Cascade model for profile face detection
│   ├── mtcnn_weights.npy   # MTCNN model weights
└── dist/                   # Compiled executable directory
```

## User Guide

### Installation

To use **Obscurrra**, follow these steps:

1. **Download the Executable:** Obtain the compiled executable from the release section.
2. **Run the Application:** Double-click the executable to launch the GUI application.

### Usage

1. **Select Input Folder:** Click "Browse" to select the folder containing images for processing.
2. **Select Output Folder:** Choose the destination folder where processed images will be saved.
3. **Select Face Detection Models:** Choose the models to use for face detection (MTCNN, Frontal Face, Profile Face).
4. **Adjust Preferences:** Set the maximum image size and blur effect intensity as desired.
5. **Start Processing:** Click "Start Processing" to begin face detection and blurring.
6. **View Log and Progress:** Monitor the progress in the log display and view the processed images in the preview section.

## Developer Guide

### Prerequisites

To develop or modify **Obscurrra**, you need the following:

- Python 3.x
- Virtual environment tool (e.g., `venv`)
- Required Python packages (specified in `requirements.txt`)

### Setting Up the Development Environment

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/jakujobi/obscurrra.git
   cd obscurrra
   ```
2. **Create a Virtual Environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Application:**

   ```bash
   python src/Obscurrra.py
   ```

### Building the Executable

1. **Install PyInstaller:**

   ```bash
   pip install pyinstaller
   ```
2. **Compile the Application:**

   ```bash
   pyinstaller --onefile --windowed src/Obscurrra.py
   ```
3. **Find the Executable:**

   The compiled executable will be located in the `dist/` directory.

## Future Improvements

- **Enhanced Face Detection:** Explore advanced models for more accurate face detection.
- **Cross-Platform Compatibility:** Ensure seamless operation on different operating systems.
- **Real-Time Processing:** Implement real-time video processing capabilities.
- **Customization Options:** Allow users to customize additional processing parameters.

## Contributors

- [John Doe](https://github.com/johndoe) - Project Lead
- [Jane Smith](https://github.com/janesmith) - Developer
- [Alice Brown](https://github.com/alicebrown) - UI/UX Designer

---

This documentation provides a complete overview of the **Obscurrra** project, including its development history, technical details, and usage instructions. This information should assist both users and developers in understanding and working with the application effectively.
