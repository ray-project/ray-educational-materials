import os
import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image, ImageFilter
from torchvision import transforms as T

#
# borrowed URLs ideas and heavily modified from https://analyticsindiamag.com/how-to-run-python-code-concurrently-using-multithreading/
#

URLS = [
    "https://images.pexels.com/photos/305821/pexels-photo-305821.jpeg",
    "https://images.pexels.com/photos/509922/pexels-photo-509922.jpeg",
    "https://images.pexels.com/photos/325812/pexels-photo-325812.jpeg",
    "https://images.pexels.com/photos/1252814/pexels-photo-1252814.jpeg",
    "https://images.pexels.com/photos/1420709/pexels-photo-1420709.jpeg",
    "https://images.pexels.com/photos/963486/pexels-photo-963486.jpeg",
    "https://images.pexels.com/photos/1557183/pexels-photo-1557183.jpeg",
    "https://images.pexels.com/photos/3023211/pexels-photo-3023211.jpeg",
    "https://images.pexels.com/photos/1031641/pexels-photo-1031641.jpeg",
    "https://images.pexels.com/photos/439227/pexels-photo-439227.jpeg",
    "https://images.pexels.com/photos/696644/pexels-photo-696644.jpeg",
    "https://images.pexels.com/photos/911254/pexels-photo-911254.jpeg",
    "https://images.pexels.com/photos/1001990/pexels-photo-1001990.jpeg",
    "https://images.pexels.com/photos/3518623/pexels-photo-3518623.jpeg",
    "https://images.pexels.com/photos/916044/pexels-photo-916044.jpeg",
    "https://images.pexels.com/photos/2253879/pexels-photo-2253879.jpeg",
    "https://images.pexels.com/photos/3316918/pexels-photo-3316918.jpeg",
    "https://images.pexels.com/photos/942317/pexels-photo-942317.jpeg",
    "https://images.pexels.com/photos/1090638/pexels-photo-1090638.jpeg",
    "https://images.pexels.com/photos/1279813/pexels-photo-1279813.jpeg",
    "https://images.pexels.com/photos/434645/pexels-photo-434645.jpeg",
    "https://images.pexels.com/photos/1571460/pexels-photo-1571460.jpeg",
    "https://images.pexels.com/photos/1080696/pexels-photo-1080696.jpeg",
    "https://images.pexels.com/photos/271816/pexels-photo-271816.jpeg",
    "https://images.pexels.com/photos/421927/pexels-photo-421927.jpeg",
    "https://images.pexels.com/photos/302428/pexels-photo-302428.jpeg",
    "https://images.pexels.com/photos/443383/pexels-photo-443383.jpeg",
    "https://images.pexels.com/photos/3685175/pexels-photo-3685175.jpeg",
    "https://images.pexels.com/photos/2885578/pexels-photo-2885578.jpeg",
    "https://images.pexels.com/photos/3530116/pexels-photo-3530116.jpeg",
    "https://images.pexels.com/photos/9668911/pexels-photo-9668911.jpeg",
    "https://images.pexels.com/photos/14704971/pexels-photo-14704971.jpeg",
    "https://images.pexels.com/photos/13865510/pexels-photo-13865510.jpeg",
    "https://images.pexels.com/photos/6607387/pexels-photo-6607387.jpeg",
    "https://images.pexels.com/photos/13716813/pexels-photo-13716813.jpeg",
    "https://images.pexels.com/photos/14690500/pexels-photo-14690500.jpeg",
    "https://images.pexels.com/photos/14690501/pexels-photo-14690501.jpeg",
    "https://images.pexels.com/photos/14615366/pexels-photo-14615366.jpeg",
    "https://images.pexels.com/photos/14344696/pexels-photo-14344696.jpeg",
    "https://images.pexels.com/photos/14661919/pexels-photo-14661919.jpeg",
    "https://images.pexels.com/photos/5977791/pexels-photo-5977791.jpeg",
    "https://images.pexels.com/photos/5211747/pexels-photo-5211747.jpeg",
    "https://images.pexels.com/photos/5995657/pexels-photo-5995657.jpeg",
    "https://images.pexels.com/photos/8574183/pexels-photo-8574183.jpeg",
    "https://images.pexels.com/photos/14690503/pexels-photo-14690503.jpeg",
    "https://images.pexels.com/photos/2100941/pexels-photo-2100941.jpeg",
    "https://images.pexels.com/photos/210019/pexels-photo-210019.jpeg",
    "https://images.pexels.com/photos/112460/pexels-photo-112460.jpeg",
    "https://images.pexels.com/photos/116675/pexels-photo-116675.jpeg",
    "https://images.pexels.com/photos/3586966/pexels-photo-3586966.jpeg",
    "https://images.pexels.com/photos/313782/pexels-photo-313782.jpeg",
    "https://www.nasa.gov/centers/stennis/images/content/702979main_SSC-2012-01487.jpg",
    "https://live.staticflickr.com/2443/3984080835_71b0426844_b.jpg",
    "https://www.aero.jaxa.jp/eng/facilities/aeroengine/images/th_aeroengine05.jpg",
    "https://images.pexels.com/photos/370717/pexels-photo-370717.jpeg",
    "https://images.pexels.com/photos/1323550/pexels-photo-1323550.jpeg",
    "https://images.pexels.com/photos/11374974/pexels-photo-11374974.jpeg",
    "https://images.pexels.com/photos/408951/pexels-photo-408951.jpeg",
    "https://images.pexels.com/photos/3889870/pexels-photo-3889870.jpeg",
    "https://images.pexels.com/photos/1774389/pexels-photo-1774389.jpeg",
    "https://images.pexels.com/photos/3889854/pexels-photo-3889854.jpeg",
    "https://images.pexels.com/photos/2196578/pexels-photo-2196578.jpeg",
    "https://images.pexels.com/photos/2885320/pexels-photo-2885320.jpeg",
    "https://images.pexels.com/photos/7189303/pexels-photo-7189303.jpeg",
    "https://images.pexels.com/photos/9697598/pexels-photo-9697598.jpeg",
    "https://images.pexels.com/photos/6431298/pexels-photo-6431298.jpeg",
    "https://images.pexels.com/photos/7131157/pexels-photo-7131157.jpeg",
    "https://images.pexels.com/photos/4840134/pexels-photo-4840134.jpeg",
    "https://images.pexels.com/photos/5359974/pexels-photo-5359974.jpeg",
    "https://images.pexels.com/photos/3889854/pexels-photo-3889854.jpeg",
    "https://images.pexels.com/photos/1753272/pexels-photo-1753272.jpeg",
    "https://images.pexels.com/photos/2328863/pexels-photo-2328863.jpeg",
    "https://images.pexels.com/photos/6102161/pexels-photo-6102161.jpeg",
    "https://images.pexels.com/photos/6101986/pexels-photo-6101986.jpeg",
    "https://images.pexels.com/photos/3334492/pexels-photo-3334492.jpeg",
    "https://images.pexels.com/photos/5708915/pexels-photo-5708915.jpeg",
    "https://images.pexels.com/photos/5708913/pexels-photo-5708913.jpeg",
    "https://images.pexels.com/photos/6102436/pexels-photo-6102436.jpeg",
    "https://images.pexels.com/photos/6102144/pexels-photo-6102144.jpeg",
    "https://images.pexels.com/photos/6102003/pexels-photo-6102003.jpeg",
    "https://images.pexels.com/photos/6194087/pexels-photo-6194087.jpeg",
    "https://images.pexels.com/photos/5847900/pexels-photo-5847900.jpeg",
    "https://images.pexels.com/photos/1671479/pexels-photo-1671479.jpeg",
    "https://images.pexels.com/photos/3335507/pexels-photo-3335507.jpeg",
    "https://images.pexels.com/photos/6102522/pexels-photo-6102522.jpeg",
    "https://images.pexels.com/photos/6211095/pexels-photo-6211095.jpeg",
    "https://images.pexels.com/photos/720347/pexels-photo-720347.jpeg",
    "https://images.pexels.com/photos/3516015/pexels-photo-3516015.jpeg",
    "https://images.pexels.com/photos/3325717/pexels-photo-3325717.jpeg",
    "https://images.pexels.com/photos/849835/pexels-photo-849835.jpeg",
    "https://images.pexels.com/photos/302743/pexels-photo-302743.jpeg",
    "https://images.pexels.com/photos/167699/pexels-photo-167699.jpeg",
    "https://images.pexels.com/photos/259620/pexels-photo-259620.jpeg",
    "https://images.pexels.com/photos/300857/pexels-photo-300857.jpeg",
    "https://images.pexels.com/photos/789380/pexels-photo-789380.jpeg",
    "https://images.pexels.com/photos/735987/pexels-photo-735987.jpeg",
    "https://images.pexels.com/photos/572897/pexels-photo-572897.jpeg",
    "https://images.pexels.com/photos/300857/pexels-photo-300857.jpeg",
    "https://images.pexels.com/photos/760971/pexels-photo-760971.jpeg",
    "https://images.pexels.com/photos/789382/pexels-photo-789382.jpeg",
    "https://images.pexels.com/photos/33041/antelope-canyon-lower-canyon-arizona.jpg",
    "https://images.pexels.com/photos/1004665/pexels-photo-1004665.jpeg",
]
THUMB_SIZE = (64, 64)


def extract_times(lst: Tuple[int, float]) -> List[float]:
    """
    Given a list of Tuples[batch_size, execution_time] extract the latter
    """
    times = [t[1] for t in lst]
    return times


def plot_times(batches: List[int], s_lst: List[float], d_lst: List[float]) -> None:
    """
    Plot the execution times for serail vs distributed for each respective batch size of images
    """
    s_times = extract_times(s_lst)
    d_times = extract_times(d_lst)
    data = {"batches": batches, "serial": s_times, "distributed": d_times}

    df = pd.DataFrame(data)
    df.plot(x="batches", y=["serial", "distributed"], kind="bar")
    plt.ylabel("Times in sec", fontsize=12)
    plt.xlabel("Number of Batches of Images", fontsize=12)
    plt.grid(False)
    plt.show()


def display_random_images(image_list: List[str], n: int = 3) -> None:
    """
    Display a grid of images, default 3 of images we want to process
    """
    random_samples_idx = random.sample(range(len(image_list)), k=n)
    plt.figure(figsize=(16, 8))
    for i, targ_sample in enumerate(random_samples_idx):
        plt.subplot(1, n, i + 1)
        img = Image.open(image_list[targ_sample])
        img_as_array = np.asarray(img)
        plt.imshow(img_as_array)
        title = f"\nshape: {img.size}"
        plt.axis("off")
        plt.title(title)
    plt.show()


def download_images(url: str, data_dir: str) -> None:
    """
    Given a URL and the image data directory, fetch the URL and save it in the data directory
    """
    img_data = requests.get(url).content
    img_name = url.split("/")[4]
    img_name = f"{data_dir}/{img_name}.jpg"
    with open(img_name, "wb+") as f:
        f.write(img_data)


def transform_image(img_name: str, verbose=False):
    """
    This is a deliberate compute intensive image transfromation and tensor operation
    to simulate a compute intensive image processing
    """
    img = Image.open(img_name)
    before_shape = img.size

    # Make the image blur with specified intensify
    # Use torchvision transformation to augment the image
    img = img.filter(ImageFilter.GaussianBlur(radius=20))
    augmentor = T.TrivialAugmentWide(num_magnitude_bins=31)
    img = augmentor(img)

    # Convert image to tensor and transpose
    tensor = torch.tensor(np.asarray(img))
    t_tensor = torch.transpose(tensor, 0, 1)

    # compute intensive operations on tensors
    random.seed(42)
    for _ in range(3):
        tensor.pow(3).sum()
        t_tensor.pow(3).sum()
        torch.mul(tensor, random.randint(2, 10))
        torch.mul(t_tensor, random.randint(2, 10))
        torch.mul(tensor, tensor)
        torch.mul(t_tensor, t_tensor)

    # Resize to a thumbnail
    img.thumbnail(THUMB_SIZE)
    after_shape = img.size
    if verbose:
        print(
            f"{os.path.basename(img_name)} augmented: shape:{img.size}| image tensor shape:{tensor.size()} transpose shape:{t_tensor.size()}"
        )

    return before_shape, after_shape
