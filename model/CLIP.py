from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image_paths = ['data/toys/images/1.jpg', 'data/toys/images/2.jpg', 'data/toys/images/3.jpg']

images = [Image.open(p).convert("RGB") for p in image_paths]
inputs = processor(images=images, return_tensors="pt").to(device)
with torch.no_grad():
    embeds = model.get_image_features(**inputs)
    embeds = embeds / embeds.norm(dim=-1, keepdim=True)
    torch.save(embeds, "image_embeds.pt")