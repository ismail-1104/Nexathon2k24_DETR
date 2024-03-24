from flask import Flask, render_template, request
from product import generate_image  # Assuming poster function is defined in product.py
# from diffusion import generate
from diffusers import StableDiffusionPipeline
import tensorflow as tf
from tensorflow import keras
import torch
import os


import subprocess



class CFG:
    device = "cuda"
    seed = 42
    generator = torch.Generator().manual_seed(seed)
    image_gen_steps = 35
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (400,400)
    image_gen_guidance_scale = 9
    prompt_gen_model_id = "gpt2"
    prompt_dataset_size = 6
    prompt_max_length = 12

def generate(prompt, model):
    print("Generating.........")
    print(prompt)
    image = model(
        prompt, num_inference_steps=CFG.image_gen_steps,
        generator=CFG.generator,
        guidance_scale=CFG.image_gen_guidance_scale
    ).images[0]

    image = image.resize(CFG.image_gen_size)
    output_dir = 'static/img'
    filename_prefix = 'generated_image-stable'

    img_path = os.path.join(output_dir, f'{filename_prefix}.png')
    image.save(img_path, 'PNG')
    print("GENERATED")
    return img_path

pipe = StableDiffusionPipeline.from_pretrained("my_model.h5")
# pipe.load_model(model, device="cpu")
pipe = pipe.to(CFG.device)

app = Flask(__name__)
image_location = f'static/img/generated_image-stable.png'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/product2', methods=['GET', 'POST'])
def product2():
    if request.method == 'POST':
        prompt = request.form['prompt']
        # subprocess.run(["python", "diffusion.py"])
        image = generate(prompt, pipe)
        return render_template('product2.html', image_location=image_location)
        # image_location = f'static/img/generated_image-stable.png'  # Assuming the image is saved with this name
        # return render_template('product2.html', image_location=image_location)
    return render_template('product2.html')

@app.route('/product', methods=['GET', 'POST'])
def product():
    if request.method == 'POST':
        text_input = request.form['text_input']
        image_location = generate_image(text_input)  # Call the poster function with text input
        return render_template('product.html', image_location=image_location)
    return render_template('product.html')

if __name__ == '__main__':
    app.run(debug=True)