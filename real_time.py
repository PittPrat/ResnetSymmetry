import cv2
import numpy as np
import onnxruntime as ort
import threading
from queue import Queue
from scipy.fft import fft2, fftshift

# ✅ Load ONNX Model
onnx_model_path = "symmetry_classifier.onnx"
ort_session = ort.InferenceSession(onnx_model_path)

# ✅ Global Variables
frame_queue = Queue(maxsize=1)  # Buffer to store the latest frame
output_queue = Queue(maxsize=1)  # Buffer to store the latest symmetry results

# ✅ Function to Capture Video Frames in Background
def capture_frames():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if not frame_queue.full():
            frame_queue.put(frame)
    cap.release()

# ✅ Function to Process Frames & Run Symmetry Detection
def process_frames():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()

            # ✅ Compute Optimized Fourier Transform-Based Symmetry Score
            fourier_score = fast_fourier_symmetry(frame)

            # ✅ Extract CNN Features
            resized_frame = cv2.resize(frame, (224, 224))
            frame_np = np.transpose(resized_frame, (2, 0, 1)).astype(np.float32) / 255.0
            frame_np = frame_np.reshape(1, 3, 224, 224)  # Batch format

            # ✅ Run ONNX Model for Fast Inference
            cnn_features = np.random.rand(1, 256).astype(np.float32)  # Placeholder for CNN features
            input_features = np.concatenate((cnn_features, [[fourier_score]]), axis=1)

            result = ort_session.run(None, {"input": input_features})[0][0]
            is_symmetrical = "Symmetrical" if result > 0.5 else "Non-Symmetrical"

            # ✅ Store Result for Display
            if not output_queue.full():
                output_queue.put((frame, fourier_score, is_symmetrical))

# ✅ Function to Display Results
def display_results():
    while True:
        if not output_queue.empty():
            frame, score, label = output_queue.get()

            # ✅ Draw Results on Frame
            cv2.putText(frame, f"Fourier Score: {score:.2f}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, label, (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            cv2.imshow("Real-Time Symmetry Detection", frame)

            # ✅ Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

# ✅ Start Multi-Threaded Execution
threading.Thread(target=capture_frames, daemon=True).start()
threading.Thread(target=process_frames, daemon=True).start()
display_results()
cv2.destroyAllWindows()
