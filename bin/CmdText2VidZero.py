""" This is the comparison used in our paper
"""
import argparse
import torch
import imageio
import glob
import os
import numpy as np
from diffusers import TextToVideoZeroPipeline
from diffusers.utils import export_to_video

from TrailBlazer.Misc import Logger as log
from TrailBlazer.Misc import ConfigIO


def get_args():
    """args parsing
    Args:
    Returns:
    """
    parser = argparse.ArgumentParser(description="Directed Video Diffusion")
    # parser.add_argument('--foobar', action='store_true')
    parser.add_argument(
        "-c", "--config", help="Input config file", required=True, type=str
    )
    parser.add_argument(
        "-mr", "--model-root", help="Model root directory", default="./", type=str
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="/tmp",
        help="Path to save the generated videos",
    )
    parser.add_argument(
        "-xl", "--zeroscope-xl", help="Search parameter", action="store_true"
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()

    experiment_bundles = []
    log.info("Loading config..")
    if os.path.isdir(args.config):
        configs = sorted(glob.glob(f"{args.config}/*.yaml"))
        for cfg in configs:
            log.info(cfg)
            bundle = ConfigIO.config_loader(cfg)
            experiment_bundles.append([bundle, cfg])
    else:
        log.info(args.config)
        bundle = ConfigIO.config_loader(args.config)
        experiment_bundles.append([bundle, args.config])

    model_id = "runwayml/stable-diffusion-v1-5"
    model_path = os.path.join(args.model_root, model_id)
    pipe = TextToVideoZeroPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16
    ).to("cuda")

    for bundle, config in experiment_bundles:

        if not bundle.get("text2vidzero"):
            log.warn("No [text2vidzero] field found in the config file. Abort.")
            continue
        if not bundle.get("keyframe"):
            prompt = bundle["prompt"]
        else:
            prompt = bundle["keyframe"][0]["prompt"]
        motion_field_strength_x = bundle["text2vidzero"]["motion_field_strength_x"]
        motion_field_strength_y = bundle["text2vidzero"]["motion_field_strength_y"]
        motion_field_strength_x = 0.
        motion_field_strength_y = 0.
        height = bundle["height"]
        width = bundle["width"]
        num_inference_steps = bundle.get("num_inference_steps")
        kwargs = {}
        if num_inference_steps:
            kwargs.update({"num_inference_steps": num_inference_steps})
        result = pipe(
            prompt=prompt,
            motion_field_strength_x=motion_field_strength_x,
            motion_field_strength_y=motion_field_strength_y,
            video_length=24,
            height=height,
            width=width,
            **kwargs,
        ).images

        # Save video frames
        save_path = "Text2VideoZero"
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
        result = [(r * 255).astype("uint8") for r in result]
        imageio.mimsave(output_video_path, result, fps=4)

        data = {
            "latents": {},
            "bundle": bundle,
            "bbox": {},
        }
        latent_path = os.path.splitext(output_video_path)[0] + ".pt"
        torch.save(data, latent_path)
        config_path = os.path.splitext(output_video_path)[0] + ".yaml"
        ConfigIO.config_saver(bundle, config_path)

        log.info(f"Video saved at {output_video_path}")
