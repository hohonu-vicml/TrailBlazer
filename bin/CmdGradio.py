# import spaces
import sys
import time
import os
import torch
import gradio as gr
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageColor
from urllib.request import urlopen

# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

root = os.path.dirname(os.path.abspath(__file__))
static = os.path.join(root, "static")

from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.pipelines import TextToVideoSDPipeline
from diffusers.utils import export_to_video

from TrailBlazer.Misc import ConfigIO
from TrailBlazer.Misc import Logger as log
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


model_id = "cerspense/zeroscope_v2_576w"
model_path = sys.argv[-1] + model_id
pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

MAX_KEYS = 10

# @spaces.GPU(duration=120)
def core(bundle):
    generator = torch.Generator().manual_seed(int(bundle["seed"]))
    result = pipe(
        bundle=bundle,
        height=512,
        width=512,
        generator=generator,
        num_inference_steps=40,
        progress=gr.Progress(track_tqdm=True),
    )
    return result.frames


def clear_btn_fn():

    return 0, *[""] * MAX_KEYS, *[""] * MAX_KEYS, *[""] * MAX_KEYS, ""


def gen_btn_fn(
    *args,
    # *prompts,
    # *bboxes,
    # *frame_indices,
    # word_prompt_indices_tb,
    # trailing_length,
    # n_spatial_steps,
    # n_temporal_steps,
    # spatial_strengthen_scale,
    # spatial_weaken_scale,
    # temporal_strengthen_scale,
    # temporal_weaken_scale,
    # rand_seed,
    progress=gr.Progress(),
):

    # no prompt at all
    if not args[0]:
        return

    rand_seed = args[-1]
    temporal_weaken_scale = args[-2]
    temporal_strengthen_scale = args[-3]
    spatial_weaken_scale = args[-4]
    spatial_strengthen_scale = args[-5]
    n_temporal_steps = args[-6]
    n_spatial_steps = args[-7]
    trailing_length = args[-8]
    word_prompt_indices = args[-9]

    bundle = {}
    bundle["trailing_length"] = trailing_length
    bundle["num_dd_spatial_steps"] = n_spatial_steps
    bundle["num_dd_temporal_steps"] = n_temporal_steps
    bundle["num_frames"] = 24
    bundle["seed"] = rand_seed
    bundle["spatial_strengthen_scale"] = spatial_strengthen_scale
    bundle["spatial_weaken_scale"] = spatial_weaken_scale
    bundle["temp_strengthen_scale"] = temporal_strengthen_scale
    bundle["temp_weaken_scale"] = temporal_weaken_scale
    bundle["token_inds"] = [int(v) for v in word_prompt_indices.split(",")]

    bundle["keyframe"] = []

    for i in range(MAX_KEYS):
        keyframe = {}
        if not args[i]:
            break
        keyframe["prompt"] = args[i]
        keyframe["bbox_ratios"] = [float(v) for v in args[i + MAX_KEYS].split(",")]
        keyframe["frame"] = int(args[i + 2 * MAX_KEYS])
        bundle["keyframe"].append(keyframe)
    print(bundle)
    result = core(bundle)
    path = export_to_video(result)
    return path


def keyframe_update(num):
    keyframes = []
    for i in range(num):
        keyframes.append(gr.Row(visible=True))
    for i in range(MAX_KEYS - num):
        keyframes.append(gr.Row(visible=False))
    return keyframes


def save_mask(inputs):
    layers = inputs["layers"]
    if not layers:
        return inputs["background"]
    mask = layers[0]
    new_image = Image.new("RGBA", mask.size, color="white")
    new_image.paste(mask, mask=mask)
    new_image = new_image.convert("RGB")
    print("SAve")
    return ImageOps.invert(new_image)


def out_label_cb(im):
    layers = im["layers"]
    if not isinstance(layers, list):
        layers = [layers]

    img = None
    text = "Bboxes: "
    for idx, layer in enumerate(layers):
        mask = np.array(layer).sum(axis=-1)
        ys, xs = np.where(mask != 0)
        h, w = mask.shape
        if not list(xs) or not list(ys):
            continue
        x_min = np.min(xs)
        x_max = np.max(xs)
        y_min = np.min(ys)
        y_max = np.max(ys)

        text += "{:.2f},{:.2f},{:.2f},{:.2f}".format(
            x_min * 1.0 / w, y_min * 1.0 / h, x_max * 1.0 / w, y_max * 1.0 / h
        )
        text += ";\n"
    return text


def out_board_cb(im):

    layers = im["layers"]
    if not isinstance(layers, list):
        layers = [layers]

    img = None
    for idx, layer in enumerate(layers):
        mask = np.array(layer).sum(axis=-1)
        ys, xs = np.where(mask != 0)

        if not list(xs) or not list(ys):
            continue

        h, w = mask.shape
        if not img:
            img = Image.new("RGBA", (w, h))
        x_min = np.min(xs)
        x_max = np.max(xs)
        y_min = np.min(ys)
        y_max = np.max(ys)

        # output
        shape = [(x_min, y_min), (x_max, y_max)]
        colors = list(ImageColor.colormap.keys())
        draw = ImageDraw.Draw(img)
        draw.rectangle(shape, outline=colors[idx], width=5)
        text = "Bbox#{}".format(idx)
        font = ImageFont.load_default()
        draw.text((x_max - 0.5 * (x_max - x_min), y_max), text, font=font, align="left")

    return img


with gr.Blocks(
    analytics_enabled=False,
    title="TrailBlazer Demo",
) as main:

    description = """
    <h1 align="center" style="font-size: 48px">TrailBlazer: Trajectory Control for Diffusion-Based Video Generation (v0.0.3)</h1>
    <h4 align="center" style="margin: 0;">If you like our project, please give us a star ✨ at our Huggingface space, and our Github repository.</h4>
        <br>
        <span align="center" style="font-size: 18px">
            [<a href="https://hohonu-vicml.github.io/Trailblazer.Page/" target="_blank">Project Page</a>]
            [<a href="http://arxiv.org/abs/2401.00896" target="_blank">Paper</a>]
            [<a href="https://github.com/hohonu-vicml/Trailblazer" target="_blank">GitHub</a>]
            [<a href="https://www.youtube.com/watch?v=kEN-32wN-xQ" target="_blank">Project Video</a>]
            [<a href="https://www.youtube.com/watch?v=P-PSkS7sNco" target="_blank">Result Video</a>]
        </span>
    </p>
    <p>
        <strong>Usage:</strong> Our Gradio app is implemented based on our executable script CmdTrailBlazer in our github repository. Please see our general information below for a quick guidance, as well as the hints within the app widgets.
    <ul>
    <li><strong>Basic</strong>: Our app workflow is straightforward. First, select the Number of keyframes, then fill all the values in the appeared prompt/frame indice/bounding box(bbox) for each keyframe, as well as the word prompt indices. Finally, hit the Generate button to run the TrailBlazer. It is roughly 90secs to get the result. We also provide the <strong>SketchPadHelper</strong> to visually design our bbox format</li>

    <li><strong>Advanced Options</strong>: We also offer some key parameters to adjust the synthesis result. Please see our paper for more information about the ablations.</li>
    </ul>

    For your initial use, it is advisable to select one of the examples provided below and attempt to swap the subject first (e.g., cat -> lion). Subsequently, define the keyframe with the associated bbox/frame/prompt. Please note that our current work is based on the ZeroScope (cerspense/zeroscope_v2_576w) model. Using prompts that are commonly recognized in the ZeroScope model context is recommended.
    </p>
    """

    gr.HTML(description)
    dummy_note = gr.Textbox(interactive=True, label="Note", visible=False)

    keyframes = []
    prompts = []
    bboxes = []
    frame_indices = []

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Row():
                with gr.Tab("Main"):

                    # text_prompt_tb = gr.Textbox(
                    #     interactive=True, label="Keyframe: Prompt"
                    # )
                    # bboxes_tb = gr.Textbox(interactive=True, label="Keyframe: Bboxes")
                    # frame_tb = gr.Textbox(
                    #     interactive=True, label="Keyframe: frame indices"
                    # )

                    dropdown = gr.Dropdown(
                        label="Number of keyframes",
                        choices=range(2, MAX_KEYS),
                    )
                    for i in range(MAX_KEYS):
                        with gr.Row(visible=False) as row:
                            text = f"Keyframe #{i}"
                            text = gr.HTML(text, visible=True)
                            prompt = gr.Textbox(
                                None,
                                label=f"Prompt #{i}",
                                visible=True,
                                interactive=True,
                                scale=4,
                            )
                            frame_ids = gr.Textbox(
                                None,
                                label=f"Frame indice #{i}",
                                visible=True,
                                interactive=True,
                                scale=1,
                            )
                            bbox = gr.Textbox(
                                None,
                                label=f"BBox #{i}",
                                visible=True,
                                interactive=True,
                                scale=3,
                            )
                            prompts.append(prompt)
                            bboxes.append(bbox)
                            frame_indices.append(frame_ids)
                        keyframes.append(row)
                    dropdown.change(keyframe_update, dropdown, keyframes)

                    with gr.Row():
                        word_prompt_indices_tb = gr.Textbox(
                            interactive=True, label="Word prompt indices:"
                        )
                        text = "<strong>Hint</strong>: Each keyframe is associated with a prompt, frame indice, and the corresponding bbox. The bbox is the tuple of the four floats determining the four bbox corners (left, top, right, bottom) in normalized ratio. The word prompt indices is 1-indexed value to indicate the word in prompt. Note that we use <strong>COMMA</strong> to separate the multiple values."
                        gr.HTML(text)
                    with gr.Row():
                        clear_btn = gr.Button(value="Clear")
                        gen_btn = gr.Button(value="Generate")

                    with gr.Accordion("Advanced Options", open=False):
                        text = "<strong>Hint</strong>: This default value should be sufficient for most tasks. However, it's important to note that our approach is currently implemented on ZeroScope, and its performance may be influenced by the model's characteristics. We plan to conduct experiments on different models in the future."
                        gr.HTML(text)
                        text = "<strong>Hint</strong>: When the #Spatial edits and #Temporal edits sliders are 0, it means the experiment will run without TrailBlazer but just simply a T2V generation through ZeroScope."
                        gr.HTML(text)
                        with gr.Row():
                            trailing_length = gr.Slider(
                                minimum=0,
                                maximum=30,
                                step=1,
                                value=13,
                                interactive=True,
                                label="#Trailing",
                            )
                            n_spatial_steps = gr.Slider(
                                minimum=0,
                                maximum=30,
                                step=1,
                                value=5,
                                interactive=True,
                                label="#Spatial edits",
                            )
                            n_temporal_steps = gr.Slider(
                                minimum=0,
                                maximum=30,
                                step=1,
                                value=5,
                                interactive=True,
                                label="#Temporal edits",
                            )
                        with gr.Row():
                            spatial_strengthen_scale = gr.Slider(
                                minimum=0,
                                maximum=2,
                                step=0.01,
                                value=0.15,
                                interactive=True,
                                label="Spatial Strengthen Scale",
                            )
                            spatial_weaken_scale = gr.Slider(
                                minimum=0,
                                maximum=1,
                                step=0.01,
                                value=0.001,
                                interactive=True,
                                label="Spatial Weaken Scale",
                            )
                            temporal_strengthen_scale = gr.Slider(
                                minimum=0,
                                maximum=2,
                                step=0.01,
                                value=0.15,
                                interactive=True,
                                label="Temporal Strengthen Scale",
                            )
                            temporal_weaken_scale = gr.Slider(
                                minimum=0,
                                maximum=1,
                                step=0.01,
                                value=0.001,
                                interactive=True,
                                label="Temporal Weaken Scale",
                            )

                        with gr.Row():
                            guidance_scale = gr.Slider(
                                minimum=0,
                                maximum=50,
                                step=0.5,
                                value=7.5,
                                interactive=True,
                                label="Guidance Scale",
                            )
                            rand_seed = gr.Slider(
                                minimum=0,
                                maximum=523451232531,
                                step=1,
                                value=0,
                                interactive=True,
                                label="Seed",
                            )

                with gr.Tab("SketchPadHelper"):
                    with gr.Row():
                        user_board = gr.ImageMask(type="pil", label="Draw me")
                        out_board = gr.Image(type="pil", label="Processed bbox")
                        user_board.change(
                            out_board_cb, inputs=[user_board], outputs=[out_board]
                        )
                    with gr.Row():
                        text = "<strong>Hint</strong>: Utilize a black pen with the Draw Button to create a ``rough'' bbox. When you press the green ``Save Changes'' Button, the app calculates the minimum and maximum boundaries. Each ``Layer'', located at the bottom left of the pad, corresponds to one bounding box. Copy the returned value to the bbox textfield in the main tab."
                        gr.HTML(text)
                    with gr.Row():
                        out_label = gr.Label(label="Converted bboxes string")
                        user_board.change(
                            out_label_cb, inputs=[user_board], outputs=[out_label]
                        )

        with gr.Column(scale=1):
            gr.HTML(
                '<span style="font-size: 20px; font-weight: bold">Generated Video</span>'
            )
            with gr.Row():
                out_gen_1 = gr.Video(visible=True, show_label=False)

    with gr.Row():
        gr.Markdown("## Two keyframes example")
    with gr.Row():
        gr.Examples(
            examples=[
                [
                    "assets/gradio/fish-RL.mp4",
                    "It generates clownfish at right, then move to left",
                    2,
                    "A clownfish swimming in a coral reef",
                    "A clownfish swimming in a coral reef",
                    "0",
                    "24",
                    "0.5,0.35,1.0,0.65",
                    "0.0,0.35,0.5,0.65",
                    "1,2",
                    "123451232531",
                ],
                [
                    "assets/gradio/fish-TL2BR.mp4",
                    "The fish moves from top left to bottom right, from far to near.",
                    2,
                    "A fish swimming in the ocean",
                    "A fish swimming in the ocean",
                    "0",
                    "24",
                    "0.0,0.0,0.1,0.1",
                    "0.5,0.5,1.0,1.0",
                    "1, 2",
                    "0",
                ],
                [
                    "assets/gradio/tiger-TL2BR.mp4",
                    "Same with the above but now the prompt associates with tiger",
                    2,
                    "A tiger walking alone down the street",
                    "A tiger walking alone down the street",
                    "0",
                    "24",
                    "0.0,0.0,0.1,0.1",
                    "0.5,0.5,1.0,1.0",
                    "1, 2",
                    "0",
                ],
                [
                    "assets/gradio/Cat2Dog.mp4",
                    "The subject will deformed from cat to dog.",
                    2,
                    "A white cat walking on the grass",
                    "A yellow dog walking on the grass",
                    "0",
                    "24",
                    "0.7,0.4,1.0,0.65",
                    "0.0,0.4,0.3,0.65",
                    "1,2,3",
                    "123451232531",
                ],
            ],
            inputs=[
                out_gen_1,
                dummy_note,
                dropdown,
                prompts[0],
                prompts[1],
                frame_indices[0],
                frame_indices[1],
                bboxes[0],
                bboxes[1],
                word_prompt_indices_tb,
                rand_seed,
            ],
            outputs=None,
            fn=None,
            cache_examples=False,
        )

    with gr.Row():
        gr.Markdown("## Five keyframes example")
    with gr.Row():
        gr.Examples(
            examples=[
                [
                    "assets/gradio/cat-LRLR.mp4",
                    "The poor cat will run Left/Right/Left/Right :(",
                    5,
                    "A cat is running on the grass",
                    "A cat is running on the grass",
                    "A cat is running on the grass",
                    "A cat is running on the grass",
                    "A cat is running on the grass",
                    "0",
                    "6",
                    "12",
                    "18",
                    "24",
                    "0.0,0.35,0.4,0.65",
                    "0.6,0.35,1.0,0.65",
                    "0.0,0.35,0.4,0.65",
                    "0.6,0.35,1.0,0.65",
                    "0.0,0.35,0.4,0.65",
                    "1, 2",
                    "123451232530",
                ],
            ],
            inputs=[
                out_gen_1,
                dummy_note,
                dropdown,
                prompts[0],
                prompts[1],
                prompts[2],
                prompts[3],
                prompts[4],
                frame_indices[0],
                frame_indices[1],
                frame_indices[2],
                frame_indices[3],
                frame_indices[4],
                bboxes[0],
                bboxes[1],
                bboxes[2],
                bboxes[3],
                bboxes[4],
                word_prompt_indices_tb,
                rand_seed,
            ],
            outputs=None,
            fn=None,
            cache_examples=False,
        )

    clear_btn.click(
        clear_btn_fn,
        inputs=[],
        outputs=[dropdown, *prompts, *bboxes, *frame_indices, word_prompt_indices_tb],
        queue=False,
    )

    gen_btn.click(
        gen_btn_fn,
        inputs=[
            *prompts,
            *bboxes,
            *frame_indices,
            word_prompt_indices_tb,
            trailing_length,
            n_spatial_steps,
            n_temporal_steps,
            spatial_strengthen_scale,
            spatial_weaken_scale,
            temporal_strengthen_scale,
            temporal_weaken_scale,
            rand_seed,
        ],
        outputs=[out_gen_1],
    )

main.launch()
# main.launch(max_threads=400)
