import torch
from facenet_pytorch import MTCNN
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')

# Load an image
image_path = "jake.jpg" # Replace with your image path
img = Image.open(image_path).convert('RGB')

# Detect faces and get bounding boxes
boxes, _ = mtcnn.detect(img)

# Draw bounding boxes on the image
if boxes is not None:
    img_tensor = transforms.ToTensor()(img)
    drawn_img = vutils.draw_bounding_boxes((img_tensor * 255).to(torch.uint8), torch.from_numpy(boxes.astype(float)), colors="red", width=2)
    # Save or display the image with bounding boxes
    vutils.save_image(drawn_img.float() / 255, "output_image_with_faces.png")
    print(f"Detected {len(boxes)} faces.")
else:
    print("No faces detected.")