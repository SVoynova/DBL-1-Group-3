import torch
import torch.nn.functional as F
from torchvision import transforms
from net import Net
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from torchsummary import summary
import cv2


class GradCAM:
    def __init__(self, model, class_idx):
        self.model = model
        self.class_idx = class_idx
        self.target_layer = self.findLastConvLayer(self.model)

    def find_target_layer(self):
        if self.layer_name is None:
            for name, module in reversed(self.model.named_modules()):
                if isinstance(module, torch.nn.Conv2d):
                    return module  # Return the first Conv2d layer found in reverse order
            raise ValueError("cant find Conv2D layer")
        else:
            if self.layer_name in dict(self.model.named_modules()):
                return dict(self.model.named_modules())[self.layer_name]
            else:
                raise ValueError(f"Layer {self.layer_name} not found in the model.")

    def forward_hook(self, module, input, output):
        self.conv_output = output

    def backward_hook(self, module, grad_in, grad_out):
        self.grad_output = grad_out[0]

    def generate_heatmap(self, input_tensor):
        # Register hooks
        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_backward_hook(self.backward_hook)

        # Forward pass
        output = self.model(input_tensor)
        if isinstance(output, tuple):
            output = output[0]
        self.model.zero_grad()

        # Backward pass
        one_hot_output = torch.FloatTensor(1, output.size()[-1]).zero_()
        one_hot_output[0][self.class_idx] = 1
        output.backward(gradient=one_hot_output, retain_graph=True)

        guided_gradients = self.grad_output[0]
        target = self.conv_output

        print("convOutputs shape:", self.conv_output.shape)
        print("guidedGrads shape:", guided_gradients.shape)

        # Weighted feature map
        weights = torch.mean(guided_gradients, dim=[1, 2], keepdim=True)
        cam = torch.mul(weights, target).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, input_tensor.shape[2:], mode='bilinear', align_corners=False)

        # Normalize the heatmap
        cam = cam - cam.min()
        cam = cam / cam.max()

        return cam.data.squeeze().cpu().numpy()

    def findLastConvLayer(self, model_m):
        last_conv_layer = None
        for name, child in model_m.named_children():
            if isinstance(child, torch.nn.Conv2d):
                last_conv_layer = child
            elif list(child.children()):
                potential_last_conv_layer = self.findLastConvLayer(child)
                if potential_last_conv_layer is not None:
                    last_conv_layer = potential_last_conv_layer
        return last_conv_layer

## ___________ start

xray_data = np.load(r"../data/X_test.npy")
yray_data = np.load(r"../data/Y_test.npy")

image_index = 0
img_array = xray_data[image_index]
yray_index = yray_data[image_index]


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

img_tensor = torch.from_numpy(img_array).unsqueeze(0).float()

model = Net(n_classes=6)
state_dict = torch.load("model_weights/model_03_17_15_32.pt")
filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
model.load_state_dict(filtered_state_dict, strict=False)
model.eval()
outputs = model(img_tensor)

probabilities = F.softmax(outputs, dim=1)
predicted_class = torch.argmax(probabilities, dim=1)
input_size = (1, img_array.shape[0], img_array.shape[1])

_, predicted_class = torch.max(probabilities, 1)
print(f'Predicted class: {predicted_class.item()}')

## ______________________________________________________________

def normalizeHeatmap(heatmap):
    heatmap_normalized = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
    return heatmap_colored


def convertToRgb(image, squeeze=False):
    if squeeze:
        image = image.squeeze()
    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def overlay(background, overlay, alpha=0.8):
    overlay_resized = cv2.resize(overlay, (background.shape[1], background.shape[0]))
    return np.clip(alpha * overlay_resized + (1 - alpha) * background, 0, 255).astype("uint8")


def prepare_and_visualize_heatmap(model, img_tensor, img_array, predicted_class, actual_label):
    gradcam = GradCAM(model, class_idx=predicted_class.item())
    heatmap = gradcam.generate_heatmap(img_tensor)
    heatmap_colored = normalizeHeatmap(heatmap)
    original_image = np.uint8(255 * img_array).squeeze()
    original_image_rgb = convertToRgb(original_image)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlayed_image = overlay(original_image_rgb, heatmap_rgb, 0.2)

    # Preparing the figure with a single title
    plt.figure(figsize=(12, 4))

    # Subplot 1: Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(original_image_rgb)
    plt.axis('off')

    # Subplot 2: Heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap_rgb)
    plt.axis('off')

    # Subplot 3: Overlayed Image
    plt.subplot(1, 3, 3)
    plt.imshow(overlayed_image)
    plt.axis('off')

    # Set a single title for the whole figure
    class_title = f"Predicted class: {predicted_class.item()}"
    plt.suptitle(f'Original Image | Heatmap | Overlayed Image - {class_title} | Actual: {actual_label}')

    plt.show()


prepare_and_visualize_heatmap(model, img_tensor, img_array, predicted_class, yray_index)


