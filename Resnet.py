import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

import torch
import torchvision.models as models

# ✅ Choose device (Use Mac GPU if available)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ✅ Recreate the ResNet model architecture
model = models.resnet18(pretrained=False)  # Ensure this matches the original architecture
model.fc = torch.nn.Linear(model.fc.in_features, 1)  # Modify output layer for symmetry score prediction
model = model.to(device)

# ✅ Load saved model weights correctly
model.load_state_dict(torch.load("best_symmetry_model.pth", map_location=device))
model.eval()  # Set to evaluation mode

print("✅ ResNet model loaded successfully!")

# ✅ Define Image Preprocessing (Must Match Training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet expects 224x224 images
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
])

# ✅ Classification Function
def classify_symmetry(score, threshold=75):
    return "Symmetrical" if score >= threshold else "Non-Symmetrical"

# ✅ Start MacBook Camera Stream
cap = cv2.VideoCapture(0)  # 0 = Default MacBook Camera

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert OpenCV BGR frame to PIL Image
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    # Preprocess Image
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Predict Symmetry Score
    with torch.no_grad():
        output = model(image_tensor)

    symmetry_score = output.item() * 100
    classification = classify_symmetry(symmetry_score)

    # ✅ Display Results on Camera Feed
    cv2.putText(frame, f"Symmetry: {symmetry_score:.2f}%", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, classification, (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Real-Time Symmetry Detection (ResNet)", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ✅ Cleanup
cap.release()
cv2.destroyAllWindows()
