import random
from typing import List

import pandas as pd


def get_random_elements(dataset: List, num_examples: int = 2) -> pd.DataFrame:
    """
    Picks a random subset of elements from the given dataset and displays them
    as a Pandas DataFrame.

    Args:
        dataset: A list of elements to choose from.
        num_examples: The number of elements to choose. Defaults to 2.

    Raises:
        ValueError: If `num_examples` is greater than the length of `dataset`.

    Returns:
        None
    """

    if num_examples > len(dataset):
        raise ValueError("Can't pick more elements than there are in the dataset.")

    picks = random.sample(range(len(dataset)), k=num_examples)
    return pd.DataFrame(dataset[picks])
