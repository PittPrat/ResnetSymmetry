import numpy as np
import joblib

# ✅ Load Trained Model
clf = joblib.load("symmetry_classifier.pkl")

# ✅ Generate CNN Features (Placeholder, should come from actual feature extraction)
cnn_features = np.random.rand(1, 256)  # Simulate CNN features

# ✅ Generate GPT Features (Placeholder, should be extracted from GPT model)
gpt_features = np.random.rand(1, 1)  # Simulate GPT feature

# ✅ Stack All Features Correctly
input_features = np.hstack((cnn_features, [[fourier_score]], gpt_features))  # Shape (1, 258)

# ✅ Predict Symmetry
is_symmetrical = clf.predict(input_features)[0]

print("✅ Prediction:", "Symmetrical" if is_symmetrical == 1 else "Non-Symmetrical")
