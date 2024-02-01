"""
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as nnf
import torchvision
import einops
import matplotlib.pyplot as plt
import scipy.stats as st
from PIL import Image, ImageFont, ImageDraw

plt.rcParams["figure.figsize"] = [
    float(v) * 1.5 for v in plt.rcParams["figure.figsize"]
]


class CrossAttnPainter:

    def __init__(self, bundle, pipe, root="/tmp"):
        self.dim = 64
        self.folder =

    def plot_frames(self):
        folder = "/tmp"
        from PIL import Image
        for i, f in enumerate(video_frames):
            img = Image.fromarray(f)
            filepath = os.path.join(folder, "recons.{:04d}.jpg".format(i))
            img.save(filepath)


    def plot_spatial_attn(self):

        arr = (
            pipe.unet.up_blocks[1]
            .attentions[0]
            .transformer_blocks[0]
            .attn2.processor.cross_attention_map
        )
        heads = pipe.unet.up_blocks[1].attentions[0].transformer_blocks[0].attn2.heads
        arr = torch.transpose(arr, 1, 3)
        arr = nnf.interpolate(arr, size=(64, 64), mode='bicubic', align_corners=False)
        arr = torch.transpose(arr, 1, 3)
        arr = arr.cpu().numpy()
        arr = arr.reshape(24, heads, 64, 64, 77)
        arr = arr.mean(axis=1)
        n = arr.shape[0]
        for i in range(n):
            filename = "/tmp/spatialca.{:04d}.jpg".format(i)
            plt.clf()
            plt.imshow(arr[i, :, :, 2], cmap="jet")
            plt.gca().set_axis_off()
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                                hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.savefig(filename, bbox_inches = 'tight',pad_inches = 0)
            print(filename)

    def plot_temporal_attn(self):

        # arr = pipe.unet.mid_block.temp_attentions[0].transformer_blocks[0].attn2.processor.cross_attention_map
        import matplotlib.pyplot as plt
        import torch.nn.functional as nnf
        arr = (
            pipe.unet.up_blocks[2]
            .temp_attentions[1]
            .transformer_blocks[0]
            .attn2.processor.cross_attention_map
        )
        #arr = pipe.unet.transformer_in.transformer_blocks[0].attn2.processor.cross_attention_map
        arr = torch.transpose(arr, 0, 2).transpose(1, 3)
        arr = nnf.interpolate(arr, size=(64, 64), mode="bicubic", align_corners=False)
        arr = torch.transpose(arr, 0, 2).transpose(1, 3)
        arr = arr.cpu().numpy()
        n = arr.shape[-1]
        for i in range(n-2):
            filename = "/tmp/tempcaiip2.{:04d}.jpg".format(i)
            plt.clf()
            plt.imshow(arr[..., i+2, i], cmap="jet")
            plt.gca().set_axis_off()
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.savefig(filename, bbox_inches="tight", pad_inches=0)
            print(filename)










def plot_latent_noise(latents, mode):

    for i in range(latents.shape[0]):
        tensor = latents[i].cpu()
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        scale = 255 * (max_val - min_val)
        tensor = scale * (tensor - min_val)
        tensor = tensor.type(torch.int8)
        tensor = einops.rearrange(tensor, "c w h -> w h c")
        if mode == "RGB":
            tensor = tensor[...,:3]
            mode_ = "RGB"
        elif mode == "RGBA":
            mode_ = "RGBA"
            pass
        elif mode == "GRAY":
            tensor = tensor[...,0]
            mode_ = "L"

        x = tensor.numpy()

        img = Image.fromarray(x, mode_)
        img = img.resize((256, 256), resample=Image.NEAREST )
        filepath = f"/tmp/out.{i:04d}.jpg"
        img.save(filepath)

        tensor = latents[i].cpu()
        x = tensor.flatten().numpy()
        x /= x.max()
        plt.hist(x, density=True, bins=20, range=[-1, 1])
        mn, mx = plt.xlim()
        plt.xlim(mn, mx)
        kde_xs = np.linspace(mn, mx, 300)
        kde = st.gaussian_kde(x)
        plt.plot(kde_xs, kde.pdf(kde_xs), label="PDF")
        filepath = f"/tmp/hist.{i:04d}.jpg"
        plt.savefig(filepath)
        plt.clf()

        print(i)


def plot_activation(cross_attn, prompt, filepath="", plot_with_trailings=False, n_trailing=2):
    splitted_prompt = prompt.split(" ")
    n = len(splitted_prompt)
    start = 0
    arrs = []
    if plot_with_trailings:
        for j in range(n_trailing):
            arr = []
            for i in range(start, start + n):
                cross_attn_sliced = cross_attn[..., i + 1]
                arr.append(cross_attn_sliced.T)
            start += n
            arr = np.hstack(arr)
            arrs.append(arr)
        arrs = np.vstack(arrs).T
    else:
        arr = []
        for i in range(start, start + n):
            cross_attn_sliced = cross_attn[..., i + 1]
            arr.append(cross_attn_sliced)
        arrs = np.vstack(arr)
    plt.imshow(arrs, cmap="jet", vmin=0.0, vmax=.5)
    plt.title(prompt)
    if filepath:
        plt.savefig(filepath)
    else:
        plt.show()


def draw_dd_metadata(img, bbox, text="", target_res=1024):
    img = img.resize((target_res, target_res))
    image_editable = ImageDraw.Draw(img)

    for region in [bbox]:
        x0 = region[0] * target_res
        y0 = region[2] * target_res
        x1 = region[1] * target_res
        y1 = region[3] * target_res
        image_editable.rectangle(xy=[x0, y0, x1, y1], outline=(255, 0, 0, 255), width=5)
        if text:
            font = ImageFont.truetype("./assets/JetBrainsMono-Bold.ttf", size=13)
            image_editable.multiline_text(
                (15, 15),
                text,
                (255, 255, 255, 0),
                font=font,
                stroke_width=2,
                stroke_fill=(0, 0, 0, 255),
                spacing=0,
            )
    return img




























if __name__ == "__main__":
    latents = torch.load("assets/experiments/a-cat-sitting-on-a-car_230615-144611/latents.pt")
    plot_latent_noise(latents, "GRAY")
