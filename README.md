# Openmv_Image_classification
image classification using Openmv H7plus camera

This project is a real-time object detection and classification system that uses OpenMV Cam H7 and TensorFlow Lite. 
The system detects movement using the frame differencing method and sends the detected object to a pre-trained TensorFlow Lite model for classification. The classification results are then displayed on the OpenMV IDE.
The project uses a pre-trained TensorFlow Lite model to classify objects, and the dataset used for training the model is the Amazon Arm-Bench dataset.
The dataset was recently released by Amazon, and it helps train pick-and-place robots. 
The dataset and the article can be found on http://armbench.s3-website-us-east-1.amazonaws.com/ and https://www.therobotreport.com/amazon-armbench-dataset-helps-train-pick-and-place-robots/, respectively.
