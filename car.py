import cv2
import glob
import numpy as np
import pyttsx3
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Dataset Preparation
data_dir = "mydata/cars/"
image_size = (224, 224)
random_state = 42

# Load and preprocess car images
image_paths = []
flat_labels = []

for image_path in glob.glob(data_dir + "*.jpg"):
    flat_name = image_path.split("\\")[-1].split("-")[0]  # Extract the flat name from the image file path
    image_paths.append(image_path)
    flat_labels.append(flat_name)

# Split dataset into training and validation sets
train_paths, valid_paths, train_labels, valid_labels = train_test_split(
    image_paths, flat_labels, test_size=0.2, random_state=random_state
)

# Model Training
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Freeze initial layers
for layer in base_model.layers:
    layer.trainable = False

# Add new classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(208, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Compile and train the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Real-time Inference System
def preprocess_image(image):
    image = image.resize(image_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize pixel values between 0 and 1
    return image

def predict_flat(image):
    image = preprocess_image(image)
    predictions = model.predict(image)
    flat_number = np.argmax(predictions) + 1  # Add 1 to match flat numbering
    return flat_number

# Initialize text-to-speech engine
engine = pyttsx3.init()

###car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_car.xml")

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ##cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    # Perform car detection on frame
    # car_detected = detect_car(frame)  # Implement your car detection algorithm here

    # If a car is detected
    cars = "falsefalsef"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if len(cars) > 10:
        flat_number = predict_flat(frame)
        cv2.putText(frame, f"Flat {flat_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Car Recognition", frame)
        cv2.waitKey(1)

        # Initialize the text-to-speech engine
        engine = pyttsx3.init()
        engine.say(f"The car belongs to flat {flat_number}")
        engine.runAndWait()

    else:
        cv2.imshow("Car Recognition", frame)
        cv2.waitKey(1)

        # Initialize the text-to-speech engine
        engine = pyttsx3.init()
        engine.say("This car does not belong to VBHC")
        engine.runAndWait()
        a = input("press any key to check the car")

cap.release()
cv2.destroyAllWindows()
