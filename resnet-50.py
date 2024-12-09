import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
from PIL import Image
import os
import random
from tqdm import tqdm
import json

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = models.resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model = model.to(device)

epochs = 250
img_size = (540, 960)
batch_size = 8
learning_rate = 0.00001
final_lr = 0.0000001
confidence_threshold = 0.1

# Augmentations
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, folders, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []

        for folder in folders:
            for label, subfolder in enumerate(['GOOD', 'BAD']):
                label_folder = os.path.join(folder, subfolder)
                if os.path.exists(label_folder):
                    image_files = os.listdir(label_folder)
                    for image_name in image_files:
                        image_path = os.path.join(label_folder, image_name)
                        self.images.append(image_path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

dataset_dir = './Individual-Frames'
all_folders = [os.path.join(dataset_dir, folder) for folder in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, folder))]
random.shuffle(all_folders)
split_index = int(0.8 * len(all_folders))

train_folders = all_folders[:split_index]
val_folders = all_folders[split_index:]

train_dataset = CustomDataset(train_folders, transform=transform)
val_dataset = CustomDataset(val_folders, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Training
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch", dynamic_ncols=True) as pbar:
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{running_loss / (pbar.n + 1):.4f}"})
            pbar.update(1)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

# Validation
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

from sklearn.metrics import precision_score, recall_score, accuracy_score
precision = precision_score(all_labels, all_preds, average='binary')
recall = recall_score(all_labels, all_preds, average='binary')
accuracy = accuracy_score(all_labels, all_preds)
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}")

# Save model and metrics
file_prefix = f"ResNet50-{epochs}-epoch-{batch_size}-batch-{learning_rate}-lr"
model_file = f"{file_prefix}_model.pth"
torch.save(model.state_dict(), model_file)

output_file = f"{file_prefix}_results.json"
output_data = {
    "precision": precision,
    "recall": recall,
    "accuracy": accuracy
}
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=4)

print(f"Model saved to {model_file} and metrics to {output_file}.")
