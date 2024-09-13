from PIL import Image
from transformers import CLIPProcessor, CLIPModel

COAD_image_fp="/data/liz0f/Python/mutation_images/Figure 6c transcriptomic manipulations/COAD_original.png"
OV_image_fp="/data/liz0f/Python/mutation_images/Figure 6c transcriptomic manipulations/OV_original.png"
UCEC_image_fp="/data/liz0f/Python/mutation_images/Figure 6c transcriptomic manipulations/UCEC_original.png"

model = CLIPModel.from_pretrained("downloaded_data/plip")
processor = CLIPProcessor.from_pretrained("downloaded_data/plip")

image_COAD = Image.open(COAD_image_fp)
image_OV = Image.open(OV_image_fp)
image_UCEC = Image.open(UCEC_image_fp)

inputs = processor(text=["a photo of colon tumor", "a photo of ovarian tumor", "a photo of endometrial tumor"],images=[image_COAD,image_OV,image_UCEC], return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)