"""This code is edited based on the following implementation at latest commit
5fc80c4 of the file:

https://github.com/microsoft/Peekaboo/blob/main/src/generate.py

The main changes are listed as follows:

1) We incorporate TrailBlazer configuration settings into the Peekaboo project
instead of hard-coding.

2) We introduce additional parser flags, such as model root, to enhance
flexibility.

3) This configuration can now be utilized for both TrailBlazer and Peekaboo,
enabling a more effective comparison.

How to use this command:

python bin/CmdPeekaboo.py -mr /path/to/diffusion/model/folder --config path/to/valid/config.yaml

To reproduce the command used in Peekaboo README.md

python bin/CmdPeekaboo.py -mr /home/kma/Workspace/Project/Models --config config/Peekaboo-Reproduce.yaml
"""


import os
import glob
import sys
import copy
from pprint import pprint
sys.path.insert(1, os.path.join(sys.path[0], ".."))

import warnings

import cv2
import numpy as np
import tqdm
import torch
import torch.nn.functional as F
import torchvision.io as vision_io

from TrailBlazer.Baseline.Peekaboo.src.models.pipelines import (
    TextToVideoSDPipelineSpatialAware,
)
from TrailBlazer.Pipeline.Utils import keyframed_bbox
from TrailBlazer.Misc import ConfigIO
from TrailBlazer.Misc import Logger as log
from TrailBlazer.Setting import Keyframe

from diffusers.utils import export_to_video
from PIL import Image
import torchvision

import argparse
import warnings

warnings.filterwarnings("ignore")

DTYPE = torch.float32


def get_parser():
    parser = argparse.ArgumentParser(
        description="Generate videos with different prompts and fg objects"
    )
    parser.add_argument(
        "-c", "--config", help="Input config file", required=True, type=str
    )
    parser.add_argument(
        "-mr", "--model-root", help="Model root directory", default="./", type=str
    )
    parser.add_argument(
        "-s",
        "--search",
        type=str,
        default="",
        help="Search parameter based on the number of trailing attention",
    )
    parser.add_argument(
        "--seed", type=int, default=2, help="Seed for random number generation"
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default="/tmp",
        help="Path to save the generated videos",
    )
    return parser


def generate_video(
    pipe,
    overall_prompt,
    latents,
    get_latents=False,
    num_frames=24,
    num_inference_steps=50,
    fg_masks=None,
    fg_masked_latents=None,
    frozen_steps=0,
    custom_attention_mask=None,
    fg_prompt=None,
    height=320,
    width=576,
):

    video_frames = pipe(
        overall_prompt,
        num_frames=num_frames,
        latents=latents,
        num_inference_steps=num_inference_steps,
        frozen_mask=fg_masks,
        frozen_steps=frozen_steps,
        latents_all_input=fg_masked_latents,
        custom_attention_mask=custom_attention_mask,
        fg_prompt=fg_prompt,
        make_attention_mask_2d=True,
        attention_mask_block_diagonal=True,
        height=height,
        width=width,
    ).frames
    if get_latents:
        video_latents = pipe(
            overall_prompt,
            num_frames=num_frames,
            latents=latents,
            num_inference_steps=num_inference_steps,
            output_type="latent",
        ).frames
        return video_frames, video_latents

    return video_frames


def save_frames(path):
    video, audio, video_info = vision_io.read_video(f"{path}.mp4", pts_unit="sec")

    # Number of frames
    num_frames = video.size(0)

    # Save each frame
    os.makedirs(f"{path}", exist_ok=True)
    for i in range(num_frames):
        frame = video[i, :, :, :].numpy()
        # Convert from C x H x W to H x W x C and from torch tensor to PIL Image
        # frame = frame.permute(1, 2, 0).numpy()
        img = Image.fromarray(frame.astype("uint8"))
        img.save(f"{path}/frame_{i:04d}.png")


class ClearCache:
    def __enter__(self):
        torch.cuda.empty_cache()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()


    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_root = os.environ.get("ZEROSCOPE_MODEL_ROOT")
    if not model_root:
        model_root = args.model_root
    model_id = "cerspense/zeroscope_v2_576w"
    model_path = os.path.join(model_root, model_id)

    pipe = TextToVideoSDPipelineSpatialAware.from_pretrained(
        model_path, torch_dtype=DTYPE, variant="fp32"
    ).to(torch_device)

    if DTYPE == torch.float16:
        pipe.unet = pipe.unet.half()

    experiment_bundles = []
    log.info("Loading config..")
    if os.path.isdir(args.config):
        configs = glob.glob(f"{args.config}/*.yaml")
        for cfg in configs:
            log.info(cfg)
            bundle = ConfigIO.config_loader(cfg)
            experiment_bundles.append([bundle, cfg])
    else:
        log.info(args.config)
        bundle = ConfigIO.config_loader(args.config)
        experiment_bundles.append([bundle, args.config])
        if args.search:
            for i in range(-5,5):
                bundle_new = copy.deepcopy(bundle)
                bundle_new[args.search] = bundle[args.search] + i
                print(bundle_new[args.search])
                experiment_bundles.append([bundle_new, args.config])

    for bundle, config in experiment_bundles:

        if not bundle.get("keyframe"):
            bundle["keyframe"] = Keyframe.get_stt_keyframe(bundle["prompt"])

        num_frames = 24
        height = int(bundle["height"] // 8)
        width = int(bundle["width"] // 8)

        bbox_mask = torch.zeros(
            [num_frames, 1, height, width],
            device=torch_device,
        )
        bbox_mask.fill_(0.1)
        fg_masks = torch.zeros(
            [num_frames, 1, height, width],
            device=torch_device,
        )

        if not bundle.get("peekaboo"):
            log.warn("No [peekaboo] field found in the config file. Abort.")
            continue

        bbox = keyframed_bbox(bundle)
        seed = bundle["seed"]
        random_latents = torch.randn(
            [1, 4, num_frames, height, width],
            generator=torch.Generator().manual_seed(seed),
            dtype=DTYPE,
        ).to(torch_device)

        y_start = [int(b[0] * width) for b in bbox]
        y_end = [int(b[2] * width) for b in bbox]
        x_start = [int(b[1] * height) for b in bbox]
        x_end = [int(b[3] * height) for b in bbox]

        # Populate the bbox_mask tensor with ones where the bounding box is located
        for i in range(num_frames):
            bbox_mask[i, :, x_start[i] : x_end[i], y_start[i] : y_end[i]] = 1
            fg_masks[i, :, x_start[i] : x_end[i], y_start[i] : y_end[i]] = 1

        fg_masked_latents = None

        overall_prompt = bundle["keyframe"][0]["prompt"]
        fg_object = " ".join(
            [overall_prompt.split(" ")[i - 1] for i in bundle["token_inds"]]
        )
        save_path = "Peekaboo"
        log.info(f"Generating video for prompt: [{overall_prompt}]")
        log.info(f"FG object: [{fg_object}]")

        frozen_steps = bundle["peekaboo"]["frozen_steps"]
        num_inference_steps = (
            50
            if not bundle.get("num_inference_steps")
            else bundle.get("num_inference_steps")
        )
        log.info(f"Frozen steps: [{frozen_steps}]")
        assert (
            frozen_steps <= num_inference_steps
        ), "Frozen steps should be less than or equal to the number of inference steps"
        pprint("=================================")
        pprint(bundle)
        pprint("=================================")

        # NOTE: The Peekaboo entry
        video_frames = generate_video(
            pipe,
            overall_prompt,
            random_latents,
            get_latents=False,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            fg_masks=fg_masks,
            fg_masked_latents=fg_masked_latents,
            frozen_steps=frozen_steps,
            fg_prompt=fg_object,
            height=bundle["height"],
            width=bundle["width"],
        )
        # Save video frames
        output_folder = os.path.join(args.output_path, save_path)
        os.makedirs(output_folder, exist_ok=True)
        task_name = os.path.splitext(os.path.basename(config))[0]
        output_video_path = os.path.join(output_folder, f"{task_name}.0000.mp4")
        if os.path.exists(output_video_path):
            import glob

            repeated = os.path.join(output_folder, task_name + "*mp4")
            num_reapts = len(glob.glob(repeated))
            output_video_path = os.path.join(
                output_folder, task_name + ".{:04d}.mp4".format(num_reapts)
            )

        video_path = export_to_video(video_frames, output_video_path)
        mask_folder = os.path.join(
            output_folder, os.path.splitext(output_video_path)[0] + "-mask"
        )
        os.makedirs(mask_folder, exist_ok=True)
        for i in range(num_frames):
            filepath = os.path.join(mask_folder, f"frame.{i:04d}.png")
            torchvision.utils.save_image(bbox_mask[i], filepath)

        config_path = os.path.splitext(output_video_path)[0] + ".yaml"
        ConfigIO.config_saver(bundle, config_path)

        log.info(f"Video saved at {output_video_path}")
