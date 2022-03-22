from typing import List, Tuple, Mapping, Any
import logging

import dask.bag

from pathlib import Path
from functools import partial


def get_images_bag(
    *,
    paths: List[str],
    output: Path,
    channels: List[int],
    config: Mapping[str, Any],
    partition_size: int,
    gpu_accelerated: bool,
    limit: int = -1,
    reach_limit: bool = False,
    loader_module
) -> Tuple[dask.bag.Bag, Mapping[str, type]]:

    loader = partial(
        loader_module.bag_from_directory,
        channels=channels,
        partition_size=partition_size,
        gpu_accelerated=gpu_accelerated,
        output=output,
        **(config["loading"]["loader_kwargs"] or dict()))

    images = []

    while (limit > 0) or (limit == -1):
        for path in paths:

            if limit == 0:
                break

            logging.info(f"Bagging {path}")
            bag, futures, loader_meta, length = loader(path=path, limit=limit)
            images.append(bag)

            limit -= length

        if not reach_limit:
            break

    images = dask.bag.concat(images)
    return images, futures, loader_meta
