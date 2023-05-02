# Untitled - By: ashkan - Mon May 1 2023

import sensor
import image
import time
import os
import gc
import ujson as json
import tensorflow as tf
import numpy as np

# Set up the camera
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.skip_frames(time = 2000)
clock = time.clock()

# Load the object classification model
interpreter = tf.lite.Interpreter(model_path="class.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Create a background subtractor using the Mixture of Gaussians method
fgbg = image.BackgroundSubtractor()

while(True):
    clock.tick()

    # Capture a frame from the camera
    img = sensor.snapshot()

    # Apply the background subtractor to detect movement
    fgmask = fgbg.apply(img)

    # Threshold the difference image to create a binary image
    binary = fgmask.binary([50, 50], invert=True)

    # Find blobs in the binary image
    blobs = binary.find_blobs(area_threshold=100, pixels_threshold=100, merge=True)

    # Loop over the blobs
    for blob in blobs:
        # Draw a bounding box around the blob
        img.draw_rectangle(blob.rect(), color=(0, 255, 0), thickness=2)

        # Extract the ROI (region of interest) from the frame
        roi = img.crop(blob.rect())

        # Preprocess the ROI for classification
        # (resize, convert to grayscale, normalize pixel values, etc.)
        roi = roi.to_grayscale()
        roi = roi.resize((96, 96))
        roi = roi.stretch_histogram()
        roi = roi.invert()
        roi = roi.normalise()
        roi = roi.mean_pool(kernel=2, stride=2)
        roi = np.asarray(roi, dtype=np.float32).reshape(1, 64, 64, 1)

        # Classify the ROI using the object classification model
        interpreter.set_tensor(input_details[0]['index'], roi)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])
        class_id = np.argmax(predictions)

        # Print the predicted class label
        print("Predicted class label: {}".format(class_id))

    # Draw the FPS on the image
    img.draw_string(0, 0, "FPS: {}".format(clock.fps()), color=(255, 0, 0))

    # Show the image on the OpenMV IDE
    img.flush()

    # Collect garbage to free up memory
    gc.collect()
