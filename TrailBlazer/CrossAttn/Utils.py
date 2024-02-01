import enum
import torch
import torchvision
import numpy as np

from ..Misc import Logger as log
from ..Setting import Config

import matplotlib.pyplot as plt
import matplotlib

# To avoid plt.imshow crash
matplotlib.use("Agg")


class CAttnProcChoice(enum.Enum):
    INVALID = -1
    BASIC = 0


def plot_activations(cross_attn, prompt, plot_with_trailings=False):
    num_frames = cross_attn.shape[0]
    cross_attn = cross_attn.cpu()
    for i in range(num_frames):
        filename = "/tmp/out.{:04d}.jpg".format(i)
        plot_activation(cross_attn[i], prompt, filename, plot_with_trailings)


def plot_activation(cross_attn, prompt, filepath="", plot_with_trailings=False):

    splitted_prompt = prompt.split(" ")
    n = len(splitted_prompt)
    start = 0
    arrs = []
    if plot_with_trailings:
        for j in range(5):
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
            print(i)
            cross_attn_sliced = cross_attn[..., i + 1]
            arr.append(cross_attn_sliced)
        arrs = np.hstack(arr).astype(np.float32)
    plt.clf()

    v_min = arrs.min()
    v_max = arrs.max()
    n_min = 0.0
    n_max = 1

    arrs = (arrs - v_min) / (v_max - v_min)
    arrs = (arrs * (n_max - n_min)) + n_min

    plt.imshow(arrs, cmap="jet")
    plt.title(prompt)
    plt.colorbar(orientation="horizontal", pad=0.2)
    if filepath:
        plt.savefig(filepath)
        log.info(f"Saved [{filepath}]")
    else:
        plt.show()


def get_cross_attn(
    unet,
    resolution=32,
    target_size=64,
):
    """To get the cross attention map softmax(QK^T) from Unet.
    Args:
        unet (UNet2DConditionModel): unet
        resolution (int): the cross attention map with specific resolution. It only supports 64, 32, 16, and 8
        target_size (int): the target resolution for resizing the cross attention map
    Returns:
        (torch.tensor): a tensor with shape (target_size, target_size, 77)
    """
    attns = []
    check = [8, 16, 32, 64]
    if resolution not in check:
        raise ValueError(
            "The cross attention resolution only support 8x8, 16x16, 32x32, and 64x64. "
            "The given resolution {}x{} is not in the list. Abort.".format(
                resolution, resolution
            )
        )
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        # NOTE: attn2 is for cross-attention while attn1 is self-attention
        dim = resolution * resolution
        if not hasattr(module, "processor"):
            continue
        if hasattr(module.processor, "cross_attention_map"):
            attn = module.processor.cross_attention_map[None, ...]
            attns.append(attn)

    if not attns:
        print("Err: Quried attns size [{}]".format(len(attns)))
        return
    attns = torch.cat(attns, dim=0)
    attns = torch.sum(attns, dim=0)
    # resized = torch.zeros([target_size, target_size, 77])
    # f = torchvision.transforms.Resize(size=(64, 64))
    # dim = attns.shape[1]
    # print(attns.shape)
    # for i in range(77):
    #     attn_slice = attns[..., i].view(1, dim, dim)
    #     resized[..., i] = f(attn_slice)[0]
    return attns


def get_avg_cross_attn(unet, resolutions, resize):
    """To get the average cross attention map across its resolutions.
    Args:
        unet (UNet2DConditionModel): unet
        resolution (list): a list of specific resolution. It only supports 64, 32, 16, and 8
        target_size (int): the target resolution for resizing the cross attention map
    Returns:
        (torch.tensor): a tensor with shape (target_size, target_size, 77)
    """
    cross_attns = []
    for resolution in resolutions:
        try:
            cross_attns.append(get_cross_attn(unet, resolution, resize))
        except:
            log.warn(f"No cross-attention map with resolution [{resolution}]")
    if cross_attns:
        cross_attns = torch.stack(cross_attns).mean(0)
    return cross_attns


def save_cross_attn(unet):
    """TODO: to save cross attn"""
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn2" in name:
            folder = "/tmp"
            filepath = os.path.join(folder, name + ".pt")
            torch.save(module.attn, filepath)
            print(filepath)


def use_dd(unet, use=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn2" in name:
            module.processor.use_dd = use


def use_dd_temporal(unet, use=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn2" in name:
            module.processor.use_dd_temporal = use


def get_loss(unet):
    loss = 0
    total = 0
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn2" in name:
            loss += module.processor.loss
            total += 1
    return loss / total


def get_params(unet):
    parameters = []
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn2" in name:
            parameters.append(module.processor.parameters)
    return parameters
