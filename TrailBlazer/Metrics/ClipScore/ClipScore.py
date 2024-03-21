import requests
import torch
from PIL import Image
from io import BytesIO
import os
import glob
import imageio
import tqdm
import numpy as np
from diffusers import StableDiffusionImg2ImgPipeline
from torchmetrics.functional.multimodal import clip_score
from functools import partial

from TrailBlazer.Misc import Logger as log

os.environ["TOKENIZERS_PARALLELISM"] = "0"


def calculate_clip_score(path, frame_skips=2):

    model_root = "/home/kma/Workspace/Project/Models"
    model_id = "openai/clip-vit-base-patch16"
    model_path = os.path.join(model_root, model_id)
    clip_score_fn = partial(clip_score, model_name_or_path=model_path)

    video_paths = sorted(glob.glob(os.path.join(path, "*.mp4")))
    mean_clipscore = 0
    total_number = 0
    for f, video_path in enumerate(video_paths):
        vid = imageio.get_reader(video_path)
        metadata_path = os.path.splitext(video_path)[0] + ".pt"
        metadata = torch.load(metadata_path)
        if metadata["bundle"].get("prompt"):
            prompt = metadata["bundle"].get("prompt")
        else:
            prompt = metadata["bundle"]["keyframe"][0]["prompt"]
        clipscore_per_video = []
        for i, frame in tqdm.tqdm(
            enumerate(vid),
            total=vid.count_frames(),
            leave=False,
            bar_format="{l_bar}{bar:15}{r_bar}{bar:-15b}",
        ):
            if i % frame_skips != 0:
                continue
            image_int = np.array(Image.fromarray(frame))[np.newaxis, ...].astype(
                "uint8"
            )
            # images_int = (image * 255).astype("uint8")
            clipscore = clip_score_fn(
                torch.from_numpy(image_int).permute(0, 3, 1, 2), [prompt]
            ).detach()
            clipscore_per_video.append(clipscore)
        mean_clipscore_video = np.array(clipscore_per_video).mean()

        msg = f"{f:02d}/{len(video_paths)} |"
        msg += f"CS: {mean_clipscore_video:.2f} |"
        msg += f"Prompt: {prompt} |"
        msg += f"Config: {os.path.basename(video_path)}"
        log.info(msg)
        mean_clipscore += np.array(clipscore_per_video).mean()
    return round(float(mean_clipscore) / len(video_paths), 4)
