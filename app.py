from flask import Flask, request, send_file
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import io

app = Flask(__name__)

# Configurações do modelo
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

def image_to_bytes(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

@app.route('/generate', methods=['GET'])
def generate_image():
    prompt = request.args.get('prompt', default="a photo of an astronaut riding a horse on mars", type=str)
    image = pipe(prompt).images[0]
    img_bytes = image_to_bytes(image)
    return send_file(img_bytes, mimetype='image/png', as_attachment=True, attachment_filename='image.png')

if __name__ == '__main__':
    app.run(debug=True)
