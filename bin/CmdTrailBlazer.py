"""

"""
#!/usr/bin/env pyton
import argparse
import copy
import os
import glob
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.pipelines import TextToVideoSDPipeline
from diffusers.utils import export_to_video
from PIL import Image

from TrailBlazer.Misc import ConfigIO
from TrailBlazer.Setting import Keyframe
from TrailBlazer.Misc import Logger as log
from TrailBlazer.Misc import Const
from TrailBlazer.Pipeline.TextToVideoSDPipelineCall import (
    text_to_video_sd_pipeline_call,
)
from TrailBlazer.Pipeline.UNet3DConditionModelCall import (
    unet3d_condition_model_forward,
)

TextToVideoSDPipeline.__call__ = text_to_video_sd_pipeline_call
from diffusers.models.unet_3d_condition import UNet3DConditionModel

unet3d_condition_model_forward_copy = UNet3DConditionModel.forward
UNet3DConditionModel.forward = unet3d_condition_model_forward


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
        "-mr", "--model-root", help="Model root directory", default="", type=str
    )
    parser.add_argument(
        "-cr", "--config-recover", help="Input saved data path", type=str
    )
    parser.add_argument(
        "-s",
        "--search",
        help="Search parameter based on the number of trailing attention",
        action="store_true",
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


def main():
    """The entry point to execute this program
    Args:
    Returns:
    """
    args = get_args()
    video_frames = None

    output_folder = os.path.join(args.output_path, "TrailBlazer")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if args.config_recover:
        if not os.path.exists(args.config_recover):
            log.info("Path [{}] is invalid.".format(args.config_recover))

        data = torch.load(args.config_recover)["bundle"]

        filepath = os.path.splitext(os.path.basename(args.config_recover))[0] + ".yaml"
        ConfigIO.config_saver(data, filepath=filepath)
        log.info("Config recovered in [{}]".format(filepath))

    elif args.config:

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


        model_root = os.environ.get("ZEROSCOPE_MODEL_ROOT")
        if not model_root:
            model_root = args.model_root

        model_id = "cerspense/zeroscope_v2_576w"
        model_path = os.path.join(model_root, model_id)
        pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()

        def run(bundle, config):
            # Note: We use Const module in attention processor as well and that's why here
            Const.DEFAULT_HEIGHT = bundle["height"]
            Const.DEFAULT_WIDTH = bundle["width"]
            print(Const.DEFAULT_HEIGHT, Const.DEFAULT_WIDTH)
            num_inference_steps = (
                40
                if not bundle.get("num_inference_steps")
                else bundle.get("num_inference_steps")
            )
            generator = torch.Generator().manual_seed(bundle["seed"])
            result = pipe(
                bundle=bundle,
                height=Const.DEFAULT_HEIGHT,
                width=Const.DEFAULT_WIDTH,
                generator=generator,
                num_inference_steps=num_inference_steps,
            )
            video_frames = result.frames
            video_latent = result.latents
            task_name = os.path.splitext(os.path.basename(config))[0]
            output_video_path = os.path.join(output_folder, task_name + ".0000.mp4")
            if os.path.exists(output_video_path):
                import glob
                repeated = os.path.join(output_folder, task_name + "*mp4")
                num_reapts = len(glob.glob(repeated))
                output_video_path = os.path.join(
                    output_folder, task_name + ".{:04d}.mp4".format(num_reapts)
                )
            export_to_video(video_frames, output_video_path=output_video_path)
            data = {
                "latents": result.latents,
                "bundle": bundle,
                "bbox": result.bbox_per_frame,
            }
            latent_path = os.path.splitext(output_video_path)[0] + ".pt"
            torch.save(data, latent_path)
            config_path = os.path.splitext(output_video_path)[0] + ".yaml"
            ConfigIO.config_saver(bundle, config_path)
            log.info(latent_path)
            log.info(output_video_path)
            log.info(config_path)
            log.info("Done")
            return video_frames

        if args.search:
            log.info(
                "Searching trailing length by range (-3, 4) of given {}".format(
                    bundle_copy["trailing_length"]
                )
            )
            for i in range(-3, 4):
                bundle = copy.deepcopy(bundle_copy)
                bundle["trailblazer"]["trailing_length"] += i
                run(bundle)
        else:
            for bundle, config in experiment_bundles:

                if not bundle.get("keyframe"):
                    bundle["keyframe"] = Keyframe.get_dyn_keyframe(bundle["prompt"])
                    # TODO:
                    bundle["trailblazer"]["spatial_strengthen_scale"] = 0.125
                    bundle["trailblazer"]["temp_strengthen_scale"] = 0.125
                    bundle["trailblazer"]["trailing_length"] = 15

                if not bundle.get("trailblazer"):
                    log.warn("No [trailblazer] field found in the config file. Abort.")
                    continue
                video_frames = run(bundle, config)

    if args.zeroscope_xl:

        UNet3DConditionModel.forward = unet3d_condition_model_forward_copy

        if not video_frames:
            log.error(
                "Cannot find the cache of video_frames. Did you run the base zeroscope?"
            )
            return
        model_id = "cerspense/zeroscope_v2_XL"
        model_root = "/home/kma/Workspace/Project/Models"
        model_path = os.path.join(model_root, model_id)
        pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        # memory optimization
        pipe.unet.enable_forward_chunking(chunk_size=1, dim=1)
        pipe.enable_vae_slicing()
        video = [Image.fromarray(frame).resize((1024, 576)) for frame in video_frames]
        video_frames = pipe(bundle["prompt_base"], video=video, strength=0.8).frames
        video_path = export_to_video(video_frames)
        log.info(video_path)


if __name__ == "__main__":
    main()
