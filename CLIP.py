from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

choice = input("Enter dataset name: ")
photo_dir = Path("photos") / choice
image_paths = sorted(photo_dir.glob("*"))

all_embeds = []
with torch.no_grad():
    batch_size = 32
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt").to(device)
        embeds = model.get_image_features(**inputs)
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        all_embeds.append(embeds.cpu())
        print(f"Processed {i + len(batch_paths)}/{len(image_paths)}")

embeds = torch.cat(all_embeds, dim=0)
out_dir = Path("photos/embeddings")
out_dir.mkdir(parents=True, exist_ok=True)
torch.save(embeds, out_dir / f"{choice}_image_embeds.pt")
print(f"\n Saved {len(embeds)} embeddings to {out_dir/f'{choice}_image_embeds.pt'}")