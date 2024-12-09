import torch
import torch.nn as nn
import torch.optim as optim
import json
from torchvision import transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, accuracy_score
from tqdm import tqdm
import os
from PIL import Image
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = models.efficientnet_b0(pretrained=True)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 2)
model = model.to(device)

# Hyperparameters
epochs = 100
batch_size = 8
learning_rate = 0.00001
img_size = (960, 540)

transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


dataset_dir = './Individual-Frames'

class CustomBalancedDataset(torch.utils.data.Dataset):
    def __init__(self, folders, transform=None, balance_ratio=20):
        self.transform = transform
        self.images = []
        self.labels = []

        for folder in folders:
            for label, subfolder in enumerate(['GOOD', 'BAD']):
                label_folder = os.path.join(folder, subfolder)
                if os.path.exists(label_folder):
                    image_files = sorted(os.listdir(label_folder))

                    if subfolder == 'GOOD':
                        image_files = [image_files[i] for i in range(0, len(image_files), balance_ratio)]
                        print(f"Folder {folder}/{subfolder}: Selected {len(image_files)} images out of {len(os.listdir(label_folder))}")
                    
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

all_folders = [os.path.join(dataset_dir, folder) for folder in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, folder))]
random.shuffle(all_folders)
split_index = int(0.8 * len(all_folders))  # 80% for training, 20% for validation

train_folders = all_folders[:split_index]
val_folders = all_folders[split_index:]

# Load datasets
train_dataset = CustomBalancedDataset(train_folders, transform=transform)
val_dataset = CustomBalancedDataset(val_folders, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"Total training samples: {len(train_dataset)}")
print(f"Total validation samples: {len(val_dataset)}")

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
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

precision = precision_score(all_labels, all_preds, average='binary')
recall = recall_score(all_labels, all_preds, average='binary')
accuracy = accuracy_score(all_labels, all_preds)

# Print out metrics
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Accuracy: {accuracy:.4f}")

# Save the model weights
torch.save(model.state_dict(), 'best_efficientnetb0_model.pth')

# Save metrics to JSON
output_data = {
    "precision": precision,
    "recall": recall,
    "accuracy": accuracy
}
output_file = 'efficientnetb0_training_results.json'

with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=4)

print(f"Training complete. Model saved to 'best_efficientnetb0_model.pth' and metrics saved to {output_file}")
