
import tensorflow as tf
import numpy as np
import sys
from tensorflow.keras.preprocessing import image

# Load the model
model = tf.keras.models.load_model('../model/image_classifier.h5')

# Load and preprocess the image
img_path = sys.argv[1]
img = image.load_img(img_path, target_size=(64, 64))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Predict the class of the image
prediction = model.predict(img_array)
print("Prediction:", "Class 1" if prediction[0] > 0.5 else "Class 0")
