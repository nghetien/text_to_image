import torch
from diffusers import DiffusionPipeline

from models.text_to_image import TextToImage
import time


class StableDiffusionXLBase1(TextToImage):
    pipe: DiffusionPipeline

    def __init__(self):
        super().__init__()
        self.pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                      torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        self.pipe = self.pipe.to(device)

    def text_to_image(self, prompt: str, number_images: int) -> list[str]:
        images = self.pipe(prompt=prompt).images
        images_list = []
        for i, image in enumerate(images):
            image_path = f"models/stable_diffusion_xl_base_1/images_gen/image_{time.time()}.png"
            images_list.append(image_path)
            image.save(image_path)
        return images_list
