# Obscurra

Obscurra is a Python-based image processing tool that detects and blurs faces in images using OpenCV. This repository contains the source code, documentation, and models used by the tool.

## Features

- Detects faces in images using Haar cascades.
- Blurs detected faces to anonymize them.
- Supports multiple image formats: JPG, JPEG, PNG, and WEBP.
- Modular design for easy maintenance and extensibility.

## Installation

To use Obscurra, you need to have Python and OpenCV installed. Follow these steps to set up the environment:

1. Clone the repository:

   ```sh
   git clone https://github.com/yourusername/obscurra.git
   cd obscurra
   ```
2. Create and activate a virtual environment:

   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. Install the required dependencies:

   ```sh
   pip install -r requirements.txt
   ```

## Usage

### Command Line

1. Place the images you want to process in the input directory.
2. Run the script:

   ```sh
   python src/obscurra/obscurra_v0_2.py
   ```
3. Processed images with blurred faces will be saved in the output directory.

### Example

To see an example of how to use Obscurra, you can use the sample images provided in the `examples` directory:

```sh
python src/obscurra/obscurra_v0_2.py --input_dir examples/input --output_dir examples/output
```

## Contributing
We welcome contributions! Please see CONTRIBUTING.md for more details on how to contribute to this project.

## License
This project is licensed under the HPL3 License. See the LICENSE.md file for more details.

## Contact
For any questions or suggestions, please contact:
- John Akujobi: john@jakujobi.com
```
