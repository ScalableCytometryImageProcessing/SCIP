from dask.distributed import WorkerPlugin, get_worker

import torch.nn
import torch

import numpy
import operator

from sip.data_features import models


def croppadND(img, bounding):
    padding = tuple(map(lambda a, b: abs(min(0, b - a)), bounding, img.shape))
    if sum(padding) > 0:
        before_after = tuple(map(lambda a: (a // 2, (a // 2) + (a % 2)), padding))
        img = numpy.pad(
            array=img,
            pad_width=before_after,
            mode='edge'
        )

    start = tuple(map(lambda a, da: a // 2 - da // 2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]


class ModelRemoverPlugin(WorkerPlugin):
    def teardown(self, worker):
        if hasattr(worker, "model"):
            del worker.model


def map_to_layer(partition, *, model, input_shape, batch_size):

    # check if model is already instantiated on worker
    worker = get_worker()
    if not hasattr(worker, "model"):
        # if the model is not yet present on the worker, instantiate it
        worker.model = models.get_untrained_model(input_shape)
        worker.model.eval()
        worker.output_size = worker.model(
            torch.randn(1, *input_shape, dtype=torch.float32)).shape[-1]

    if len(partition) < batch_size:
        batch_size = len(partition)

    full_out = numpy.zeros(
        shape=(len(partition), worker.output_size), dtype=numpy.float32)

    with torch.no_grad():
        n_batches = len(partition) // batch_size
        for i in range(n_batches + 1):

            if i == n_batches:
                end = len(partition)
            else:
                end = (i + 1) * batch_size

            batch = list(map(
                lambda j: croppadND(partition[j]["mask_img"], input_shape),
                range(i * batch_size, end)
            ))

            full_out[i * batch_size:end] = worker.model(
                torch.as_tensor(batch, dtype=torch.float32))

    return list(map(tuple, full_out))


def extract_features(*, images, model, input_shape, batch_size):
    """Extract intermediate representations from pretrained PyTorch models

    model: Path to PyTorch model
    intermediate_layer: Layer index from which to extract representations
    """

    # extract intermediate representations
    features = images.map_partitions(
        map_to_layer,
        model=model,
        input_shape=input_shape,
        batch_size=batch_size).to_dataframe()

    return features
