import cv2
import numpy as np
from keras.models import load_model
import joblib
import time

# Open the webcam
cap = cv2.VideoCapture(0) # might have to change index to 1

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Failed to open webcam")
    exit()

# Set the capture interval in seconds
capture_interval = 7  # Capture an image every 5 seconds -- CHANGE AS DESIRED
last_capture_time = 0

# Capture a frame from the webcam
ret, frame = cap.read()

# Check if the frame was successfully captured
if not ret:
    print("Failed to capture frame")
    # break

# Convert the frame to RGB format
image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Display the image using OpenCV
cv2.imshow('Webcam Image', image)

# Check if it's time to capture an image
# current_time = time.time()
# if current_time - last_capture_time >= capture_interval:
    # Save the image
image_filename = "capturedimage{int(current_time)}.jpg"
cv2.imwrite(image_filename, frame)
print(f"Image captured and saved as {image_filename}")
    #last_capture_time = current_time

# Wait for a key press
# if cv2.waitKey(1) == ord('q'):  # Press 'q' to quit
#     break

# Release the webcam and close OpenCV windows
cap.release()

# Load the trained models
models = {
    'Colour': load_model('trained_model_Colour.h5'),
    'Left 1': load_model('trained_model_Left 1.h5'),
    'Left 2': load_model('trained_model_Left 2.h5'),
    'Left 3': load_model('trained_model_Left 3.h5'),
    'Right 1': load_model('trained_model_Right 1.h5'),
    'Right 2': load_model('trained_model_Right 2.h5'),
    'Right 3': load_model('trained_model_Right 3.h5'),
    'Right 4': load_model('trained_model_Right 4.h5')
}

# Load the label encoders
label_encoders = {
    'Colour': joblib.load('label_encoder_Colour.pkl'),
    'Left 1': joblib.load('label_encoder_Left 1.pkl'),
    'Left 2': joblib.load('label_encoder_Left 2.pkl'),
    'Left 3': joblib.load('label_encoder_Left 3.pkl'),
    'Right 1': joblib.load('label_encoder_Right 1.pkl'),
    'Right 2': joblib.load('label_encoder_Right 2.pkl'),
    'Right 3': joblib.load('label_encoder_Right 3.pkl'),
    'Right 4': joblib.load('label_encoder_Right 4.pkl')
}

# Load and preprocess the image
# image_path = 'testData\\White_Video_7_107.jpg'  # Replace with the path to your image
img = cv2.imread(image_filename)
img = cv2.resize(img, (128, 128))  # Resize the image to match the model's expected input size
img = img.flatten()
img = np.array([img])

# Predict the values
predicted_classes = {}
for label, model in models.items():
    prediction = np.argmax(model.predict(img), axis=-1)[0]  # Get the class with the highest probability
    predicted_label = label_encoders[label].inverse_transform([prediction])[0]
    predicted_classes[label] = predicted_label

# Print the predicted class labels
print('Predicted Class Labels:')
for label, prediction in predicted_classes.items():
    print(f'{label}: {prediction}')
