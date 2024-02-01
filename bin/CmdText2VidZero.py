""" This is the comparison used in our paper
"""

import torch
import imageio
import os
from diffusers import TextToVideoZeroPipeline

model_root = "your model root"
model_id = "runwayml/stable-diffusion-v1-5"
model_path = os.path.join(model_root, model_id)
pipe = TextToVideoZeroPipeline.from_pretrained(
    model_path, torch_dtype=torch.float16
).to("cuda")

prompt = "A cat walking on the grass field"
prompt = "A clown fish swimming in a coral reef"
prompt = "A macro video of a bee pollinating a flower."
result = pipe(
    prompt=prompt, motion_field_strength_x=8, motion_field_strength_y=0, video_length=24,
).images
result = [(r * 255).astype("uint8") for r in result]
imageio.mimsave("video.mp4", result, fps=4)
