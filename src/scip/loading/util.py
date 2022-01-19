from typing import List, Tuple
from functools import partial
import logging

import dask.bag



def get_images_bag(
    *,
    paths: List[str],
    channels: List[int],
    config: dict,
    partition_size: int,
    gpu_accelerated: bool,
    limit: int,
    loader_module
) -> Tuple[dask.bag.Bag, int, dict]:

    loader = partial(
        loader_module.bag_from_directory,
        channels=channels,
        partition_size=partition_size,
        gpu_accelerated=gpu_accelerated,
        **(config["loading"]["loader_kwargs"] or dict()))

    images = []
    maximum_pixel_value = 0

    for path in paths:

        if limit == 0:
            break

        logging.info(f"Bagging {path}")
        bag, loader_meta, maximum_pixel_value, length = loader(path=path, limit=limit)
        images.append(bag)

        limit -= length

    images = dask.bag.concat(images)
    return images, maximum_pixel_value, loader_meta
