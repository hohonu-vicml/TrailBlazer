import torch
import os

DEVICE = "cuda"
GUIDANCE_SCALE = 7.5
WIDTH = 512
HEIGHT = 512
NUM_BACKWARD_STEPS = 50
STEPS = 50
DTYPE = torch.float16

MODEL_HOME = f"{os.path.expanduser('~')}/Workspace/Project/Models"

NEGATIVE_PROMPT = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer difits, cropped, worst quality, low quality, deformed body, bloated, ugly, unrealistic"
POSITIVE_PROMPT = "best quality, extremely detailed, HD, ultra-realistic, 8K, HQ, masterpiece, trending on artstation, art, smooth"


SD_V1_5_ID = "runwayml/stable-diffusion-v1-5"
SD_V1_5_PATH = f"{MODEL_HOME}/{SD_V1_5_ID}"
CNET_CANNY_ID = "lllyasviel/sd-controlnet-canny"
CNET_CANNY_PATH = f"{MODEL_HOME}/{CNET_CANNY_ID}"
CNET_OPENPOSE_ID = "lllyasviel/sd-controlnet-openpose"
CNET_OPENPOSE_PATH = f"{MODEL_HOME}/{CNET_OPENPOSE_ID}"
