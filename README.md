# Helmet-and-Number-Plate-Detection-and-Recognition
Motorcycle Accidents have been rapidly growing throughout the years in many countries. The helmet is the main safety equipment of motorcyclists. There was need to propose an automated system that monitors motorcycles and detects the persons wearing helmet or not and a system to detect number plates. 

This system proposes an automated system for detecting motorcyclists who do not wear a helmet and a system for retrieving motorcycle number plates.So, our model detects the helmet of the rider. If the two-wheeler rider does not wear the helmet, it detects the number plate of the vehicle.To detect the objects, this deep learning algorithm uses CNN (Convolutional Neural Network) that recognizes specific objects in videos, live images or feeds.

This project is a Streamlit-based application for detecting helmets, bikes, and recognizing number plates in a video stream. It uses the YOLOv3 object detection model for detecting bikes and helmets and a CNN model for helmet detection. Additionally, it recognizes number plates in real-time video.

# Installation
1. Clone the Repository

    Clone this repository to your local machine.
3. Install Dependencies :

   Navigate to the project directory and install the required dependencies listed in the requirements.txt file using pip :
   
        pip install -r requirements.txt
5. Download YOLO Weights and Configuration :
   
    Download the YOLOv3 weights (yolov3-custom_7000.weights) and configuration (yolov3-custom.cfg) files. You can obtain these files from your YOLOv3 training or a pre-trained YOLOv3           model. Place these files in the project directory.
7. Download Helmet Detection Model :
   
    Download the helmet detection model (helmet-nonhelmet_cnn.h5) and place it in the project directory. You can train this model using your dataset or use a pre-trained one.
   
# Usage
1. Run the Streamlit App

  To run the Streamlit app, use the following command:

    streamlit run source.py

  This will start the Streamlit development server and open the app in your default web browser.

2. Upload a Video File

  On the Streamlit app, use the file uploader to select a video file (e.g., MP4 or AVI) that you want to process.

3. View the Detection and Recognition Results

  The app will display the video with real-time detections of bikes and helmets. If a helmet is detected, it will be labeled as "Helmet" or "No Helmet" based on the helmet detection model's prediction. Number plates, if present, will also be recognized and displayed.

4. Interact with the App

  You can pause, resume, and navigate through the video using the app's interface. Observe the real-time results as the video plays.
  
# File Structure

* source.py: The main Streamlit app code for helmet, bike, and number plate detection and recognition.
* requirements.txt: A list of required Python packages and their versions.
* yolov3-custom_7000.weights: YOLOv3 custom-trained weights for object detection.
* yolov3-custom.cfg: YOLOv3 custom model configuration file.
* helmet-nonhelmet_cnn.h5: Helmet detection CNN model weights.
  # ScreenShots
  ![ss](https://github.com/FatimaSidra/Helmet-and-Number-Plate-Detection-and-Recognition/assets/112679516/dc00805f-2ce6-457b-b152-5b97f4b497bd)
  ![ss2](https://github.com/FatimaSidra/Helmet-and-Number-Plate-Detection-and-Recognition/assets/112679516/3254988d-1fd7-4cba-a53e-bd3efea3ce12)
  ![ss3](https://github.com/FatimaSidra/Helmet-and-Number-Plate-Detection-and-Recognition/assets/112679516/a98592c1-06e0-4933-ac5d-5fb4110a7190)
  
# Acknowledgements
* This project uses YOLOv3 for object detection. You can find more information about YOLOv3 here.

      https://pjreddie.com/darknet/yolo/
  
* The helmet detection model is a CNN-based model used for detecting helmets on bike riders.Number plate recognition is performed in real-time to identify and display number plates.
* Special thanks to the Streamlit community for creating an easy-to-use web framework for data science applications. Visit Streamlit's official website for more information.
  
Feel free to customize and extend this project to suit your specific needs or explore other object detection and recognition tasks using Streamlit.
