# palette was copied from:
# https://github.com/tensorflow/models/blob/3f1ca33afe3c1631b733ea7e40c294273b9e406d/research/deeplab/utils/get_dataset_colormap.py#L51
# (date accessed: Nov 9th, 2022)
from typing import Union

import numpy as np
import torch
from PIL.JpegImagePlugin import JpegImageFile
from datasets.arrow_dataset import Dataset
from matplotlib import pyplot as plt

from palette import create_ade20k_label_colormap as ade_palette


def prepare_pixels_with_segmentation(image: JpegImageFile, predictions: Union[torch.Tensor, np.array]):
    predictions = np.array(predictions)
    color_segments = np.zeros(
        (predictions.shape[0], predictions.shape[1], 3), dtype=np.uint8
    )
    palette = np.array(ade_palette())
    for label, color in enumerate(palette):
        color_segments[predictions == label, :] = color
    color_segments = color_segments[..., ::-1]  # convert to BGR

    pixels_with_segmentation = (
        np.array(image) * 0.5 + color_segments * 0.5
    )  # plot the image with the segmentation map
    return pixels_with_segmentation.astype(np.uint8)


def visualize_predictions(
    image: JpegImageFile, predictions: torch.Tensor, loss: np.array
):
    pxs = prepare_pixels_with_segmentation(image=image, predictions=predictions)
    plt.imshow(pxs)
    plt.axis("off")
    plt.title(f"loss: {loss:.4f}")


def display_example_images(dataset: Dataset, n: int = 2):
    fig, axes = plt.subplots(nrows=n, ncols=n, figsize=(10, 10))
    fig.set_tight_layout(True)
    print(Dataset.num_rows)
    for i, j in enumerate(
        np.random.choice(dataset.num_rows, size=(n * n), replace=False)
    ):
        image_with_pixels = prepare_pixels_with_segmentation(
            image=dataset["image"][j], predictions=np.array(dataset["annotation"][j])
        )
        axes[int(i / n), i % n].imshow(image_with_pixels)
        axes[int(i / n), i % n].axis("off")
