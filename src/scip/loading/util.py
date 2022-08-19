import logging
from typing import List, Tuple, Mapping, Any

import dask.bag

from pathlib import Path
from functools import partial


def _load_image_partition(partition, channels, load):
    return [load(event, channels) for event in partition]


def get_images_bag(
    *,
    paths: List[str],
    output: Path,
    channels: List[int],
    config: Mapping[str, Any],
    partition_size: int,
    gpu_accelerated: bool,
    loader_module
) -> Tuple[dask.bag.Bag, Mapping[str, type]]:

    loader = partial(
        loader_module.bag_from_directory,
        channels=channels,
        partition_size=partition_size,
        gpu_accelerated=gpu_accelerated,
        output=output,
        **(config["load"]["kwargs"] or dict())
    )

    images = []

    loader_meta = loader_module.get_loader_meta(**(config["load"]["kwargs"] or dict()))

    for path in paths:
        assert Path(path).exists(), f"{path} does not exist."

        logging.info(f"Bagging {path}")
        bag = loader(path=path)
        images.append(bag)

    images = dask.bag.concat(images)
    return images, loader_meta
