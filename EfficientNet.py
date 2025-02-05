import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input

# ✅ Define Custom Objects (Register Loss Function)
custom_objects = {"mse": tf.keras.losses.MeanSquaredError()}

# ✅ Load the Model with Custom Objects
model_path = "/Users/prathikpittala/Documents/RESNETSYMMETRY/symmetry_model.h5"
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

print("✅ Model loaded successfully!")



IMG_SIZE = (224, 224)

def preprocess_frame(frame):
    """Resize, normalize, and format frame for EfficientNetB0 model."""
    frame_resized = cv2.resize(frame, IMG_SIZE)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    frame_array = np.expand_dims(frame_rgb, axis=0)  # Add batch dimension
    return preprocess_input(frame_array)  # Normalize for EfficientNet

# ✅ Start Webcam Capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ✅ Preprocess Frame
    input_frame = preprocess_frame(frame)

    # ✅ Predict Symmetry Score
    symmetry_score = model.predict(input_frame)[0][0]  # Extract single value

    # ✅ Determine Symmetry Label (Threshold = 50%)
    is_symmetrical = "Symmetrical" if symmetry_score >= 50 else "Non-Symmetrical"

    # ✅ Display Results
    cv2.putText(frame, f"Symmetry Score: {symmetry_score:.2f}%", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, is_symmetrical, (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Real-Time Symmetry Detection", frame)

    # ✅ Exit on 'q' Key Press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
