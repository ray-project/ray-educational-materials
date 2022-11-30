import json
from typing import Union

import numpy as np
import torch
from PIL.JpegImagePlugin import JpegImageFile
from datasets.arrow_dataset import Dataset
from huggingface_hub import hf_hub_download
from matplotlib import pyplot as plt

# A colormap for visualizing segmentation results.
# https://github.com/tensorflow/models/blob/3f1ca33afe3c1631b733ea7e40c294273b9e406d/research/deeplab/utils/get_dataset_colormap.py#L51
# (date accessed: Nov 9th, 2022)
ade_palette = np.array(
    [
        [0, 0, 0],
        [120, 120, 120],
        [180, 120, 120],
        [6, 230, 230],
        [80, 50, 50],
        [4, 200, 3],
        [120, 120, 80],
        [140, 140, 140],
        [204, 5, 255],
        [230, 230, 230],
        [4, 250, 7],
        [224, 5, 255],
        [235, 255, 7],
        [150, 5, 61],
        [120, 120, 70],
        [8, 255, 51],
        [255, 6, 82],
        [143, 255, 140],
        [204, 255, 4],
        [255, 51, 7],
        [204, 70, 3],
        [0, 102, 200],
        [61, 230, 250],
        [255, 6, 51],
        [11, 102, 255],
        [255, 7, 71],
        [255, 9, 224],
        [9, 7, 230],
        [220, 220, 220],
        [255, 9, 92],
        [112, 9, 255],
        [8, 255, 214],
        [7, 255, 224],
        [255, 184, 6],
        [10, 255, 71],
        [255, 41, 10],
        [7, 255, 255],
        [224, 255, 8],
        [102, 8, 255],
        [255, 61, 6],
        [255, 194, 7],
        [255, 122, 8],
        [0, 255, 20],
        [255, 8, 41],
        [255, 5, 153],
        [6, 51, 255],
        [235, 12, 255],
        [160, 150, 20],
        [0, 163, 255],
        [140, 140, 140],
        [250, 10, 15],
        [20, 255, 0],
        [31, 255, 0],
        [255, 31, 0],
        [255, 224, 0],
        [153, 255, 0],
        [0, 0, 255],
        [255, 71, 0],
        [0, 235, 255],
        [0, 173, 255],
        [31, 0, 255],
        [11, 200, 200],
        [255, 82, 0],
        [0, 255, 245],
        [0, 61, 255],
        [0, 255, 112],
        [0, 255, 133],
        [255, 0, 0],
        [255, 163, 0],
        [255, 102, 0],
        [194, 255, 0],
        [0, 143, 255],
        [51, 255, 0],
        [0, 82, 255],
        [0, 255, 41],
        [0, 255, 173],
        [10, 0, 255],
        [173, 255, 0],
        [0, 255, 153],
        [255, 92, 0],
        [255, 0, 255],
        [255, 0, 245],
        [255, 0, 102],
        [255, 173, 0],
        [255, 0, 20],
        [255, 184, 184],
        [0, 31, 255],
        [0, 255, 61],
        [0, 71, 255],
        [255, 0, 204],
        [0, 255, 194],
        [0, 255, 82],
        [0, 10, 255],
        [0, 112, 255],
        [51, 0, 255],
        [0, 194, 255],
        [0, 122, 255],
        [0, 255, 163],
        [255, 153, 0],
        [0, 255, 10],
        [255, 112, 0],
        [143, 255, 0],
        [82, 0, 255],
        [163, 255, 0],
        [255, 235, 0],
        [8, 184, 170],
        [133, 0, 255],
        [0, 255, 92],
        [184, 0, 255],
        [255, 0, 31],
        [0, 184, 255],
        [0, 214, 255],
        [255, 0, 112],
        [92, 255, 0],
        [0, 224, 255],
        [112, 224, 255],
        [70, 184, 160],
        [163, 0, 255],
        [153, 0, 255],
        [71, 255, 0],
        [255, 0, 163],
        [255, 204, 0],
        [255, 0, 143],
        [0, 255, 235],
        [133, 255, 0],
        [255, 0, 235],
        [245, 0, 255],
        [255, 0, 122],
        [255, 245, 0],
        [10, 190, 212],
        [214, 255, 0],
        [0, 204, 255],
        [20, 0, 255],
        [255, 255, 0],
        [0, 153, 255],
        [0, 41, 255],
        [0, 255, 204],
        [41, 0, 255],
        [41, 255, 0],
        [173, 0, 255],
        [0, 245, 255],
        [71, 0, 255],
        [122, 0, 255],
        [0, 255, 184],
        [0, 92, 255],
        [184, 255, 0],
        [0, 133, 255],
        [255, 214, 0],
        [25, 194, 194],
        [102, 255, 0],
        [92, 0, 255],
    ]
)


def get_image_indices(dataset: Dataset, n: int):
    image_indices = np.random.choice(dataset.num_rows, size=n, replace=False)
    return [int(i) for i in image_indices]


# https://huggingface.co/datasets/huggingface/label-files/blob/main/ade20k-id2label.json
def get_labels():
    repo_id = "huggingface/label-files"
    filename = "ade20k-id2label.json"
    id2label = json.load(
        open(
            hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset"),
            "r",
        )
    )

    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}
    return id2label, label2id


def prepare_pixels_with_segmentation(
    image: JpegImageFile, segmentation_maps: Union[torch.Tensor, np.array]
):
    segmentation_maps = np.array(segmentation_maps)
    color_segments = np.zeros(
        (segmentation_maps.shape[0], segmentation_maps.shape[1], 3), dtype=np.uint8
    )
    for label, color in enumerate(ade_palette):
        color_segments[segmentation_maps == label, :] = color
    color_segments = color_segments[..., ::-1]  # convert to BGR
    pixels_with_segmentation = np.array(image) * 0.5 + color_segments * 0.5
    return pixels_with_segmentation.astype(np.uint8)


def visualize_predictions(image: JpegImageFile, segmentation_maps: torch.Tensor):
    pxs = prepare_pixels_with_segmentation(
        image=image, segmentation_maps=segmentation_maps
    )
    plt.imshow(pxs)
    plt.axis("off")


def display_example_images(dataset: Dataset, n: int = 2):
    fig, axes = plt.subplots(nrows=n, ncols=n, figsize=(10, 10))
    fig.set_tight_layout(True)
    for i, j in enumerate(
        np.random.choice(dataset.num_rows, size=(n * n), replace=False)
    ):
        image_with_pixels = prepare_pixels_with_segmentation(
            image=dataset[int(j)]["image"],
            segmentation_maps=np.array(dataset[int(j)]["annotation"]),
        )
        axes[int(i / n), i % n].imshow(image_with_pixels)
        axes[int(i / n), i % n].axis("off")


def convert_image_to_rgb(data_item):
    if data_item["image"].mode != "RGB":
        data_item["image"] = data_item["image"].convert(mode="RGB")

    return data_item
