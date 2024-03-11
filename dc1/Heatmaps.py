import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from net import Net
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

# Loading the data
xray_data = np.load(r"C:\Users\20223661\OneDrive - TU Eindhoven\Documents\Git\DBL-1-Group-3\data\X_test.npy")
image_index = 0  # This is just the first image
img_array = xray_data[image_index]

# Preprocess the image (For the data we have it's already done)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Adjust normalization as needed for grayscale images
])

img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).float()

# Load the trained model
model = Net(n_classes=6)
state_dict = torch.load(r"C:\Users\20223661\OneDrive - TU Eindhoven\Documents\Git\DBL-1-Group-3\dc1\model_weights\model_03_11_9_01.pt")
filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
# ignore the keys that we don't need for the visualization
model.load_state_dict(filtered_state_dict, strict=False)
model.eval()

# Get the feature map from the GAP layer
with torch.no_grad():
    feature_map = model.cnn_layers(img_tensor.squeeze(0))  # Squeeze the extra batch dimension

# Get the weights from the last convolutional layer
weights = model.cnn_layers[5].weight[0].view(-1)

# Calculate the heatmap
heatmap = F.relu(feature_map.squeeze(2).squeeze(2))  # Squeeze spatial dimensions

print("Weights Shape:", weights.shape)
print("Heatmap Shape:", heatmap.shape)

# The shapes do in fact not match up
heatmap = heatmap.view(-1, 1)
weights = weights.view(1, -1)

# Matrix multiplication
heatmap = heatmap @ weights

# Normalizing and reshaping the heatmap
heatmap_normal = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
heatmap_resized = cv2.resize(heatmap_normal.detach().numpy(), (img_array.shape[2], img_array.shape[1]))

print("Feature Map Shape after GAP:", feature_map.shape)
print("img_array dtype:", img_array.dtype)
print("heatmap dtype:", heatmap.dtype)
print("Image Array Shape:", img_array.shape)
print(img_tensor)

# Showing the image with the heatmap overlay
scale = 1 / 1  # I don't know the "right"scale
plt.figure(figsize=(12, 12))
plt.imshow(img_array[0], cmap='gray', aspect='auto')
plt.imshow(zoom(heatmap_resized, zoom=(scale, scale)), cmap='jet', alpha=0.5, aspect='auto')
plt.show()
