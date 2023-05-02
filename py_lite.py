import cv2
import numpy as np
import tensorflow as tf

# Load the object classification model
interpreter = tf.lite.Interpreter(model_path="lite_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Create a background subtractor using the Mixture of Gaussians method
fgbg = cv2.createBackgroundSubtractorMOG2()

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    
    # Apply the background subtractor to detect movement
    fgmask = fgbg.apply(frame)
    
    # Threshold the difference image to create a binary image
    _, binary = cv2.threshold(fgmask, 50, 255, cv2.THRESH_BINARY)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Loop over the contours
    for contour in contours:
        # Draw a bounding box around the contour
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Extract the ROI (region of interest) from the frame
        roi = frame[y:y+h, x:x+w]
        
        # Preprocess the ROI for classification
        # (resize, convert to grayscale, normalize pixel values, etc.)
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(roi, (96, 96))
        roi = np.array(roi, dtype=np.float32) / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)
        
        # Create an input tensor with the expected shape
        input_shape = input_details[0]['shape']
        input_data = np.zeros(input_shape, dtype=np.int8)
        input_data[0] = roi
        
        # Classify the ROI using the object classification model
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])
        class_id = np.argmax(predictions)
        
        # Print the predicted class label
        print("Predicted class label: {}".format(class_id))
    
    # Show the frame with bounding boxes and classification results
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
