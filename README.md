# Distance Calculator

This script processes RealSense `.bag` files to detect objects using YOLO, calculates distances between detected objects, and saves the results to a video file and a CSV file. It is particularly designed for clinical tests involving a computer vision app through the NAO robot for the treatment of autism.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Functionality](#functionality)
- [Output](#output)
- [License](#license)

## Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/your-username/Distance-Calculator.git
    cd Distance-Calculator
    ```

2. **Create a virtual environment (optional but recommended):**

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Run the script:**

    ```sh
    python Distance_calculator.py [path_to_bag_file]
    ```

    If no `.bag` file path is provided as an argument, a file dialog will open to select the file.

2. **Press `ESC` to exit the video display.**

## Dependencies

- pyrealsense2
- numpy
- open3d
- opencv-python
- ultralytics
- tkinter

You can install these dependencies using the provided `requirements.txt` file:

```sh
pip install -r requirements.txt
```

## Functionality

The script performs the following tasks:

1. **File Selection:**
    - Opens a file dialog to select a `.bag` file if no file path is provided as a command-line argument.

2. **RealSense Pipeline:**
    - Initializes a RealSense pipeline and configures it to process the selected `.bag` file.

3. **YOLO Model:**
    - Loads a YOLO model pre-trained on specific classes ("child", "children", "robot", "robotNAO", "NAO").

4. **Object Detection:**
    - Detects objects in the video frames using YOLO.
    - Calculates 3D distances between detected objects.
    - Draws bounding boxes around detected objects and displays the distances between them.

5. **Output:**
    - Saves the processed video with annotated distances to an `.mp4` file.
    - Saves detected objects and their distances to a CSV file.

## Output

1. **Video:**
    - An output video file (`output.mp4`) showing the processed frames with detected objects and distances annotated.

2. **CSV File:**
    - A CSV file (`detected_objects_distances.csv`) containing the detected objects and their 3D coordinates.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
