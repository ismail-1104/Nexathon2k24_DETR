from diffusers import StableDiffusionPipeline
import tensorflow as tf
from tensorflow import keras
import torch


class CFG:
    device = "cpu"
    seed = 42
    generator = torch.Generator().manual_seed(seed)
    image_gen_steps = 35
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (400,400)
    image_gen_guidance_scale = 9
    prompt_gen_model_id = "gpt2"
    prompt_dataset_size = 6
    prompt_max_length = 12

def generate_image(prompt, model):
    image = model(
        prompt, num_inference_steps=CFG.image_gen_steps,
        generator=CFG.generator,
        guidance_scale=CFG.image_gen_guidance_scale
    ).images[0]

    image = image.resize(CFG.image_gen_size)
    return image


# model = keras.models.load_model("my_model.h5")
pipe = StableDiffusionPipeline.from_pretrained("my_model.h5")
# pipe.load_model(model, device="cpu")
pipe = pipe.to(CFG.device)
generate_image("cat with goggles", pipe)