import os, sys
import tqdm
import glob
import imageio
import numpy as np
import torch
from PIL import Image

from CommonMetricsOnVideoQuality.calculate_fvd import calculate_fvd as calc_fvd
from InceptionScorePytorch.inception_score import inception_score
from OwlVit.main import compute_miou
from PytorchFid.src.pytorch_fid.fid_score import calculate_fid_given_paths
from ClipScore.ClipScore import calculate_clip_score
from CleanFid.cleanfid import fid

from TrailBlazer.Misc import Logger as log


def to_torch(path):
    arr = np.load(path)
    arr = np.transpose(arr, [0, 3, 1, 2])
    return torch.from_numpy(arr)


def calculate_kid(paths):
    paths = [os.path.join(p, "images") for p in paths]
    score = fid.compute_kid(paths[0], paths[1])
    print("\n -----> KID:", score)
    return score

def calculate_clipscore(path):
    clip_score = calculate_clip_score(path, frame_skips=1)
    print("\n -----> ClipScore:", clip_score)
    return clip_score


def calculate_fvd(paths):

    paths = [os.path.join(p, "out.npy") for p in paths]
    videos1 = torch.unsqueeze(to_torch(paths[0]), 0)
    videos2 = torch.unsqueeze(to_torch(paths[1]), 0)

    # NUMBER_OF_VIDEOS = 8
    # VIDEO_LENGTH = 50
    # CHANNEL = 3
    # SIZE = 64
    # videos1 = torch.zeros(
    #     NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False
    # )
    # videos2 = torch.ones(
    #     NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False
    # )
    device = torch.device("cuda")
    # device = torch.device("cpu")

    import json

    result = calc_fvd(videos1, videos2, device, method="videogpt")
    fvd_value = sum(result["value"].values()) / len(result["value"])
    print("\n -----> FVD:", fvd_value)
    return fvd_value

def calculate_is(path):

    import torchvision.datasets as dset
    import torchvision.transforms as transforms
    from torch.utils.data import Dataset, DataLoader

    class MyDataset(Dataset):
        def __init__(self, data, transform=None):
            self.data = data
            # self.targets = torch.LongTensor(targets)
            self.transform = transform

        def __getitem__(self, index):
            x = self.data[index]
            # y = self.targets[index]

            # if self.transform:
            #     x = Image.fromarray(self.data[index].astype(np.uint8).transpose(1,2,0))
            #     x = self.transform(x)

            return x

        def __len__(self):
            return len(self.data)

    path = os.path.join(path, "out.npy")
    data = to_torch(path)

    dataset = MyDataset(data)
    result = inception_score(dataset, cuda=True, batch_size=32, resize=True, splits=10)
    print(
        "\n  -----> IS(Mean/Std): ",
        result
    )
    result = [round(v, 4) for v in result]
    return result


def calculate_miou(path):
    miou = compute_miou(path)
    print("\n  -----> mIOU: ", miou)
    return miou

def calculate_fid(paths, batch_size=64, device="cuda", dims=2048, num_workers=8):
    paths = [os.path.join(p, "images") for p in paths]
    fid_value = calculate_fid_given_paths(paths, batch_size, device, dims, num_workers)
    print("\n  -----> FID: ", fid_value)
    return fid_value

def prepare_assets(
    video_folder, frame_skips=1, res_skips=1, out_filename="out.npy", max_n_videos=-1
):

    video_paths = glob.glob(os.path.join(video_folder, "*.mp4"))
    image_folder = os.path.join(video_folder, "images")
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    videos = []
    total_frame = 0
    target_shape = ()
    msg = "Pre-analyzing the video folders by "
    msg += f"frame skips every {frame_skips}, "
    msg += f"res skips {res_skips}, "
    msg += f"and maximum videos {max_n_videos}"
    log.info(msg)
    n = max_n_videos if max_n_videos > 0 else len(video_paths)
    for f, video_path in tqdm.tqdm(enumerate(video_paths), total=n, leave=False):
        if max_n_videos > 0 and f > max_n_videos:
            break
        vid = imageio.get_reader(video_path)
        shape = vid.get_data(0).shape
        frames = np.zeros((vid.count_frames(), *shape), dtype=np.float16)
        if frame_skips > 1:
            frames = frames[::frame_skips]
        if res_skips > 1:
            frames = frames[:, ::res_skips, ::res_skips, :]
        total_frame += frames.shape[0]
        if not target_shape:
            target_shape = frames.shape[1:]
    log.info(f"Analyzed, total frame: {total_frame}, target res: {target_shape}")

    videos_arr = np.zeros((total_frame, *target_shape), dtype=np.float16)
    total_frame = 0
    current_total_frame_ids = 0
    log.info("Start feeding the video data...")
    for f, video_path in tqdm.tqdm(enumerate(video_paths), total=n, leave=False):
        if max_n_videos > 0 and f > max_n_videos:
            break
        vid = imageio.get_reader(video_path)
        shape = vid.get_data(0).shape
        frames = np.zeros((vid.count_frames(), *shape), dtype=np.float16)
        for i, frame in enumerate(vid):
            frames[i] = frame
            if i % frame_skips == 0:
                filename = (
                    os.path.splitext(os.path.basename(video_path))[0] + f".{i:04d}.jpg"
                )
                image = Image.fromarray(frame)
                image_filepath = os.path.join(image_folder, filename)
                image.save(image_filepath)

        if frame_skips > 1:
            frames = frames[::frame_skips]
        if res_skips > 1:
            frames = frames[:, ::res_skips, ::res_skips, :]

        total_frame += frames.shape[0]

        videos_arr[
            current_total_frame_ids : current_total_frame_ids + frames.shape[0]
        ] = frames
        current_total_frame_ids += frames.shape[0]
        # msg = ""
        # msg += f"Prog: {f:05d}/{len(video_paths)} "
        # msg += f"File: {filename} "
        # msg += f"#Fr(U/O): {frames.shape[0]:03d}/{vid.count_frames():03d} "
        # msg += (
        #     f"Res(U/O): ({frames.shape[1]},{frames.shape[2]})/({shape[0]},{shape[1]}) "
        # )
        # msg += f"Current total frames: {total_frame}"
        # print(msg)

    # videos = np.concatenate(videos, dtype=np.float16)
    out_filepath = os.path.join(video_folder, out_filename)
    np.save(out_filepath, videos_arr)
    log.info(f"Saved [{out_filepath}]")
    log.info(
        f"Array size: {videos_arr.shape} File size: {os.path.getsize(out_filepath)/(1024**2):.2f}MB"
    )
    return videos


def batch_evaluate(real_folder, fake_folder):
    paths = [fake_folder, real_folder]
    scores = {}
    scores["FID"] = calculate_fid(paths)
    scores["FVD"] = calculate_fvd(paths)
    scores["IS"] = calculate_is(paths[0])
    scores["KID"] = calculate_kid(paths)
    scores["CLIPSim"] = calculate_clipscore(paths[0])
    if "Text2VideoZero" not in fake_folder:
        scores["mIOU"] = calculate_miou(paths[0])
    return scores


def make_dataset(path):
    prepare_assets(
        # f"/home/kma/Workspace/Project/Trailblazer/ECCV/Metrics-Dyna/{tester}",
        path,
        frame_skips=1,
        res_skips=2,
        max_n_videos=400,
    )


if __name__ == "__main__":
    pass
    # paths = ["/home/kma/Workspace/Project/Data/Test2", "/home/kma/Workspace/Project/Data/Test1"]
    # paths = ["/home/kma/Workspace/Project/Data/Test3"]
    # prepare_assets("/home/kma/Workspace/Project/Trailblazer/ECCV/Supp/TrailBlazer")
    # prepare_assets("/home/kma/Workspace/Project/Trailblazer/ECCV/Supp/Peekaboo")
    # quit()
    # calculate_clipscore("/home/kma/Workspace/Project/Trailblazer/ECCV/Metrics/TrailBlazer")
    testers = ["Text2VideoZero", "Peekaboo", "TrailBlazer"]
    testers = ["Peekaboo", "TrailBlazer"]
    # calculate_clipscore(f"/home/kma/Workspace/Project/Trailblazer/ECCV/Metrics/{tester}")
    real_folder = "/home/kma/Workspace/Project/Data/AnimalKingdom/video/"

    all_scores = {}
    for tester in testers:
        print("\n\n\n")
        log.info(f"===== {tester} =====")
        # fake_folder = (
        #     f"/home/kma/Workspace/Project/Trailblazer/ECCV/Metrics-Static/{tester}"
        # )
        fake_folder = f"/home/kma/Workspace/Project/Trailblazer/ECCV/Supp/{tester}"
        all_scores[tester] = batch_evaluate(real_folder, fake_folder)

    log.info("===== Done =====")
    log.info("")
    log.info("")
    log.info("")
    log.info("==== Summary ====")
    log.info("")
    for tester in all_scores.keys():
        log.info(f"[[{tester}]]")
        for s in all_scores[tester].keys():
            log.info(f"    {s}:{all_scores[tester][s]}")
        log.info("------------------------")
