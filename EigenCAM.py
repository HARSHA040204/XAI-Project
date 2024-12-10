#pip install grad-cam
#Yolo11
from ultralytics import YOLO
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import torch
import numpy as np
import cv2

# Define the YOLOWrapper class
class YOLOWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[0]  # Return only the first element of the tuple

# Function to apply EigenCAM
def apply_eigencam_to_yolo(model, img_tensor, target_layers):
    wrapped_model = YOLOWrapper(model.model)
    cam = EigenCAM(wrapped_model, target_layers)
    img_tensor = img_tensor.to(next(model.parameters()).device)
    grayscale_cam = cam(img_tensor, targets=None)

    if isinstance(grayscale_cam, list) and len(grayscale_cam) == 1:
        grayscale_cam = grayscale_cam[0]

    if len(grayscale_cam.shape) == 3:
        grayscale_cam = np.mean(grayscale_cam, axis=0)

    return grayscale_cam


# Load your model
model = YOLO('/Users/karthikrajanichenametla/X_AI_cassava/yolo_cam_env/best_yolov11_cotton.pt')
target_layers = [model.model.model[-2]]  # Adjust based on your model architecture

# Load and preprocess the image
image_path = '/Users/karthikrajanichenametla/X_AI_cassava/cotton.jpeg'
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img, (640, 640))
img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0

# Apply EigenCAM
grayscale_cam = apply_eigencam_to_yolo(model, img_tensor, target_layers)

# Overlay the CAM on the image
cam_image = show_cam_on_image(img_resized / 255.0, grayscale_cam, use_rgb=True)

# Display the result
Image.fromarray(cam_image).show()  # Or save it if you want


