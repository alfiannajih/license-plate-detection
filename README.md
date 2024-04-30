
<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">
<h3 align="center">License Plate Detection</h3>
  <p align="center">
    Developing a machine learning model to detect license plates and recognize it as a text.
    <br />
    <br />
    <!--<a href="#">View Demo</a>-->
  </p>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#basic-example">Basic Example</a></li>
        <li><a href="#result-example">Result Example</a></li>
      </ul>
    </li>
    <li><a href="#To Do">To Do</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This project aims to demonstrate the machine learning workflow for building a computer vision project, specifically focusing on license plate detection.

![Training Process](images/training_process.png)
As shown in the image above, the training process for both object detection and text recognition involves data collection, data annotation, image processing, data augmentation, and fine-tuning the model. Here are tech that I used for this training process:
- Roboflow, for collecting and annotating the data (The dataset can be viewed [here](https://universe.roboflow.com/alfian-fc0es/indonesia-license-plate/dataset/10)).
- Goolge Colab, for fine tuning the model.
- YOLOv8 Nano model, for faster performance in real time object detection.
- Convolutional Recurrent Neural Networks (CRNN) model, for text recognition.

![Inference Process](images/inference_process.png)
For the inference process, the input frame will be fed into an object detection model that has been fine-tuned. The detected license plate will then be processed (i.e. converted to grayscale, binarization, etc.) and fed into a text recognition model. Once the text has been detected, it will be combined with the original frame as the final result.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!--
<!-- GETTING STARTED -->
## Getting Started

Here are a few steps to follow in order to run the project locally on your computer.

### Prerequisites
Before doing the installation, make sure you have Python installed on your computer, I used `python 3.12.1` during the development.

### Installation
1. Clone the repo
    ```sh
    git clone https://github.com/alfiannajih/license-plate-detection
    ```
2. Change your directory
    ```sh
    cd license-plate-detection
    ```
3. Install the dependencies
    ```sh
    pip install -r requirements.txt
    ```
4. Run the sample to check if the program work correctly
    ```sh
    python main.py --mode image --file_path sample_input.jpg
    ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- USAGE EXAMPLES -->
## Usage
For the usage guide, you can access the help menu by running the following command in the command line:
```sh
python main.py --help
```
```
options:
  -h, --help            show this help message and exit
  --mode {image,video,webcam}
                        Mode to run
  --webcam_id WEBCAM_ID
                        ID of the webcam
  --file_path FILE_PATH
                        Path to image file
  --plate_detection_model_path PLATE_DETECTION_MODEL_PATH
                        Path to object detection (YOLO) model file
  --text_recognition_model_path TEXT_RECOGNITION_MODEL_PATH
                        Path to text recognition model directory
  --height_threshold HEIGHT_THRESHOLD
                        Height threshold for filtering text detection
```
### Basic Example
There are three modes in which the program can be used: `image`, `webcam`, and `video`.
1. Image mode
    
    Run the following command:
    ```sh
    python main.py --mode image --file_path sample_input.jpg
    ```
    You can replace `sample_input.jpg` with the name of your image file. The above commad will use `sample_input.jpg` as an input, just as in the installation process example.
    
    output example:
    ![Image mode output example](images/sample_output_image.png)

2. Webcam mode

    Run the following command:
    ```sh
    python main.py --mode webcam --webcam_id 0
    ```
    It will open your webcam, allowing you to use it as the input frame.

    output example:
    ![Webcam mode example](images/sample_output_webcam.png)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Result Example
There are four possible outcomes for the results:
- Both object detection and text recognition work perfectly.
![Good Example](images/good_result.png)
  
- Object detection works perfectly, but text recognition misses some letters/numbers.
![Bad Example 1](images/miss_leter.png)

- Object detection works perfectly, but text recognition fails to detect letters/numbers entirely.
![Bad Example 2](images\miss_all_letter.png)

- Object detection fails to detect the plate partially or entirely..
![Bad Example 3](images\partly_detected.png)
![Bad Example 4](images\not_detected.png)

<!-- To Do -->
## To Do

- [x] Collect data
- [x] Object detection model
- [x] Combine model with Optical Character Recognition (OCR)
- [x] Fine tune OCR model


<p align="right">(<a href="#readme-top">back to top</a>)</p>