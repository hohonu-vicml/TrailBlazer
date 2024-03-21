import requests
from PIL import Image
import torch
import tqdm
import os
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import glob
import imageio


def compute_miou(path):

    # model
    model_root = "/home/kma/Workspace/Project/Models"
    model_id = "google/owlvit-base-patch32"
    model_path = os.path.join(model_root, model_id)
    processor = OwlViTProcessor.from_pretrained(model_path)
    model = OwlViTForObjectDetection.from_pretrained(model_path)

    video_paths = glob.glob(os.path.join(path, "*.mp4"))
    mean_iou = 0
    for f, video_path in enumerate(video_paths):
        vid = imageio.get_reader(video_path)
        shape = vid.get_data(0).shape

        metadata_path = os.path.splitext(video_path)[0] + ".pt"
        metadata = torch.load(metadata_path)

        prompt = metadata["bundle"]["keyframe"][0]["prompt"]
        token_ids = metadata["bundle"]["token_inds"]
        gt_bboxes = metadata["bbox"]
        texts = [
            [
                " ".join(
                    [p for i, p in enumerate(prompt.split(" ")) if i + 1 in token_ids]
                )
            ]
        ]

        total_frames = 0
        mean_iou_video = 0.0
        print(video_path)
        for i, frame in tqdm.tqdm(
            enumerate(vid),
            total=vid.count_frames(),
            leave=False,
            bar_format="{l_bar}{bar:15}{r_bar}{bar:-15b}",
        ):

            image = Image.fromarray(frame)
            inputs = processor(text=texts, images=image, return_tensors="pt")
            outputs = model(**inputs)
            # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
            target_sizes = torch.Tensor([image.size[::-1]])
            # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
            results = processor.post_process_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=0.1
            )

            text = texts[0]
            boxes, scores, labels = (
                results[0]["boxes"],
                results[0]["scores"],
                results[0]["labels"],
            )

            try:
                box = [v.item() for v in boxes[torch.argmax(scores)]]
            except:
                continue

            gt_box = gt_bboxes[i]
            gt_box[0] = gt_box[0] * metadata["bundle"]["width"]
            gt_box[1] = gt_box[1] * metadata["bundle"]["height"]
            gt_box[2] = gt_box[2] * metadata["bundle"]["width"]
            gt_box[3] = gt_box[3] * metadata["bundle"]["height"]
            score = scores[torch.argmax(scores)].item()

            area_overlap = (min(box[2], gt_box[2]) - max(box[0], gt_box[0])) * (
                min(box[3], gt_box[3]) - max(box[1], gt_box[1])
            )

            area_union = (
                (box[2] - box[0]) * (box[3] - box[1])
                + (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                - area_overlap
            )

            iou = area_overlap / area_union
            # print(iou)
            if iou < 0:
                iou = 0
            # print(
            #     f"Score {score:.2f} Box(L/T/R/B) {box[0]:.2f} {box[1]:.2f} {box[2]:.2f} {box[3]:.2f}"
            # )
            # print(box)
            # print(gt_box)
            # print("iou", iou)
            mean_iou_video += iou
            total_frames += 1

            # quit()
            # for box, score, label in zip(boxes, scores, labels):
            #     box = [round(i, 2) for i in box.tolist()]
            #     print(
            #         f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}"
            # )
        if total_frames:
            mean_iou_video = mean_iou_video/total_frames
            mean_iou += mean_iou_video

    return round(mean_iou / len(video_paths), 4)
