# import os
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import array_to_img

# def generate_image(text):
#     # Load the pre-trained generator model
#     generator_model = load_model('generator.h5')
    
#     # Define other parameters
#     output_dir = 'static/img'
#     filename_prefix = 'generated_image'
#     latent_dim = 100

#     # Generate random noise
#     noise = np.random.normal(0, 1, (1, latent_dim))
    
#     # Generate image
#     generated_image = generator_model.predict(noise)[0]
    
#     # Save the generated image
#     # os.makedirs(output_dir, exist_ok=True)
#     img = array_to_img(generated_image)
#     img_path = os.path.join(output_dir, f'{filename_prefix}.png')
#     img.save(img_path)
#     return img_path


import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import array_to_img
from PIL import Image, ImageDraw, ImageFont

def generate_image(text):
    # Load the pre-trained generator model
    generator_model = load_model('generator.h5')

    # Define other parameters
    output_dir = 'static/img'
    filename_prefix = 'generated_image'
    latent_dim = 100

    # Generate random noise
    noise = np.random.normal(0, 1, (1, latent_dim))

    # Generate image
    generated_image = generator_model.predict(noise)[0]

    # Convert array to image
    img = array_to_img(generated_image)

    # Add text overlay
    img = add_text_overlay(img, text)

    # Save the generated image with text overlay
    # os.makedirs(output_dir, exist_ok=True)
    img_path = os.path.join(output_dir, f'{filename_prefix}.png')
    img.save(img_path, 'PNG')

    return img_path

def add_text_overlay(img, text):
    # Convert image to RGBA mode
    img = img.convert("RGBA")
    draw = ImageDraw.Draw(img)

    # Get image dimensions
    width, height = img.size

    # Calculate font size
    fontsize = 0
    font = ImageFont.truetype("Arial.ttf", fontsize)
    while font.getlength(text) < 0.75 * width:
        fontsize += 1
        font = ImageFont.truetype("Arial.ttf", fontsize)

    fontsize -= 1
    font = ImageFont.truetype("Arial.ttf", fontsize)

    # Calculate text position
    val = draw.textbbox((0, 0), text, font=font)
    w, h = val[2], val[3]
    text_position = ((width / 2 - w / 2), (height / 2) - h / 2 - 1)

    # Add text to the image
    draw.text(text_position, text, font=font, fill=(255, 255, 255))

    return img

# # Generate image with text overlay and return final path
# final_img_path = generate_image("Chips")
# print(f"Final image saved at: {final_img_path}")
