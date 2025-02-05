import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import os
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
from transformers import LlavaProcessor, LlavaForConditionalGeneration, AutoModel, AutoTokenizer

# ==============================
# 🔹 CONFIGURATION
# ==============================
device = "cuda" if torch.cuda.is_available() else "mps"  # Use MacBook MPS or CUDA
llava_model_name = "liuhaotian/llava-v1.5-7b-lora"  # LLaVA model
resnet_model_name = "resnet50"  # Pretrained CNN
transformer_model_name = "bert-base-uncased"  # Text transformer

# ==============================
# 🔹 IMAGE PREPROCESSING
# ==============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ==============================
# 🔹 CUSTOM DATASET LOADER
# ==============================
class SymmetryDataset(Dataset):
    def __init__(self, image_folder, transform):
        self.image_folder = image_folder
        self.image_files = os.listdir(image_folder)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Use Fourier Transform for Symmetry Score
        gray = cv2.cvtColor(np.array(image.permute(1, 2, 0)), cv2.COLOR_RGB2GRAY)
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        symmetry_score = np.abs(f_shift).mean()  # Use magnitude mean as proxy for symmetry

        return image, torch.tensor(symmetry_score, dtype=torch.float32)

# ==============================
# 🔹 LOAD PRE-TRAINED MODELS
# ==============================
# Load ResNet-50 for Image Embeddings
resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)  # Correct way
resnet.eval()
resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove FC layer
resnet.eval().to(device)
hf_token= "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"
# Load LLaVA for Text Annotations
llava_processor = LlavaProcessor.from_pretrained(llava_model_name, hf_token)
llava_model = LlavaForConditionalGeneration.from_pretrained(llava_model_name, token=hf_token).to("mps")

# Load Transformer for Text Embeddings
text_model = AutoModel.from_pretrained(transformer_model_name).to(device)
text_tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)

# ==============================
# 🔹 FEATURE EXTRACTION FUNCTION
# ==============================
def extract_features(image):
    """Extract image embeddings using ResNet-50"""
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        img_embedding = resnet(image).squeeze()
    return img_embedding.cpu().numpy()

def generate_text_annotation(image):
    """Generate textual description of an image using LLaVA"""
    inputs = llava_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_text = llava_model.generate(**inputs)
    return generated_text[0]

def extract_text_embedding(text):
    """Extract text embeddings from transformer"""
    inputs = text_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_embedding = text_model(**inputs).last_hidden_state.mean(dim=1).squeeze()
    return text_embedding.cpu().numpy()

# ==============================
# 🔹 CLASSIFIER MODEL
# ==============================
class SymmetryClassifier(nn.Module):
    def __init__(self, img_embed_dim=2048, text_embed_dim=768, hidden_dim=512):
        super(SymmetryClassifier, self).__init__()
        self.fc1 = nn.Linear(img_embed_dim + text_embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, img_features, text_features):
        x = torch.cat((img_features, text_features), dim=1)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Initialize model
classifier = SymmetryClassifier().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001)

# ==============================
# 🔹 TRAINING LOOP
# ==============================
def train_classifier(dataset, epochs=5):
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    classifier.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for images, symmetry_scores in dataloader:
            img_features = torch.tensor([extract_features(img) for img in images]).to(device)
            text_annotations = [generate_text_annotation(img) for img in images]
            text_features = torch.tensor([extract_text_embedding(txt) for txt in text_annotations]).to(device)

            optimizer.zero_grad()
            predictions = classifier(img_features, text_features).squeeze()
            loss = criterion(predictions, symmetry_scores.to(device))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")

# ==============================
# 🔹 REAL-TIME MACBOOK CAMERA TEST
# ==============================
def real_time_symmetry():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_features = extract_features(transform(image))
        text = generate_text_annotation(image)
        text_features = extract_text_embedding(text)

        symmetry_score = classifier(torch.tensor(img_features).to(device), torch.tensor(text_features).to(device)).item()
        print(f"Symmetry Score: {symmetry_score:.2f}")

        cv2.imshow("Live Symmetry Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# ==============================
# 🔹 RUN
# ==============================
dataset = SymmetryDataset("/Users/prathikpittala/Documents/RESNETSYMMETRY/symmetry_dataset", transform)
train_classifier(dataset)
real_time_symmetry()
