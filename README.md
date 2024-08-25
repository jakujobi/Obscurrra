# Obscurra

Obscurra is a Python-based image processing tool designed to detect faces in images and apply a blur effect to anonymize them. The project aims to enhance privacy by automatically processing images to obscure identifiable facial features through a user-friendly graphical interface.

This repository contains the source code, documentation, and models used by the tool.

## Features

- Detects faces in images using Haar cascades.
- Blurs detected faces to anonymize them.
- Supports multiple image formats: JPG, JPEG, PNG, and WEBP.
- Modular design for easy maintenance and extensibility.

## Objectives

- Develop a tool to detect and blur faces in images.
- Provide a simple GUI for image processing.
- Compile the application into a portable executable.

## Timeline

### Phase 1: Planning (January - February 2024)

- **Research:** Investigated face detection models and image processing techniques.
- **Requirements:** Defined project scope and objectives.

### Phase 2: Development (March - June 2024)

- **Core Features:** Implemented face detection using Haar Cascade and MTCNN.
- **GUI:** Developed a Tkinter-based user interface.

### Phase 3: Testing and Optimization (July 2024)

- **Testing:** Identified and resolved bugs.
- **Optimization:** Enhanced performance and accuracy.

### Phase 4: Deployment (August 2024)

- **Executable:** Packaged application using PyInstaller.
- **Documentation:** Created user and developer guides.

## Achievements

- Integrated multiple face detection models.
- Developed a portable executable with a comprehensive GUI.
- Enhanced user experience with real-time image processing updates.

## Challenges and Bugs

- **Model Integration:** Ensured compatibility between different face detection models.
- **Performance:** Required optimization for processing speed and accuracy.
- **GUI Usability:** Improved interface design based on user feedback.

## Methodology

The project followed an agile development process with iterative cycles of development, testing, and refinement, focusing on usability and performance.

## Technologies Used

- **Python**: Core language
- **Tkinter**: GUI framework
- **OpenCV**: Image processing
- **TensorFlow**: MTCNN model
- **PyInstaller**: Executable packaging

## Project Structure

```
Obscurrra/
├── src/
│   ├── Obscurrra.py        # Main script
│   ├── icon.ico            # Application icon
│   ├── haarcascade_frontalface_default.xml # Frontal face detection model
│   ├── haarcascade_profileface.xml # Profile face detection model
│   ├── mtcnn_weights.npy   # MTCNN model weights
└── dist/                   # Compiled executable
```

## User Guide

1. **Download the Executable:** Obtain from the release section.
2. **Run the Application:** Launch the GUI executable.
3. **Process Images:**
   - Select input and output folders.
   - Choose detection models and set preferences.
   - Click "Start Processing."

## Developer Guide

### Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/jakujobi/obscurrra.git
   cd obscurrra
   ```
2. **Create a Virtual Environment and Install Dependencies:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```
3. **Run the Application:**

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

## Future Improvements

- Explore advanced face detection models.
- Add real-time video processing capabilities.
- Enhance cross-platform compatibility.

## Contributors

- **John Akujobi**- Project Lead, UI Designer, Software Engineer

---

This documentation provides a comprehensive yet concise overview of the **Obscurrra** project, highlighting key features, challenges, and future directions for both users and developers.
