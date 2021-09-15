
from pathlib import Path


def load_image()


def bag_from_directory(path, idx, channels, partition_size):

    events = []
    for i, p in enumerate(Path(path).glob("**/*.tif")):
        events.append(dict(path=str(p), idx=idx + i))