import torch, os, sys, glob
from TrailBlazer.Misc import ConfigIO
from TrailBlazer.Pipeline.Utils import keyframed_bbox

path = "/home/kma/Workspace/Project/Trailblazer/ECCV/Supp/Peekaboo"
configs = sorted(glob.glob(f"{path}/*.yaml"))
for cfg in configs:
    bundle = ConfigIO.config_loader(cfg)
    keyframe_bboxes = keyframed_bbox(bundle)

    latent_path = os.path.splitext(cfg)[0] + ".pt"
    data = {
        "latents": {},
        "bundle": bundle,
        "bbox": keyframe_bboxes,
    }
    torch.save(data, latent_path)
    print(latent_path)
