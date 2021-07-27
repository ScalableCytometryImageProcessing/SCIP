from dask.distributed import WorkerPlugin, get_worker, get_client
import dask.dataframe

import torch.nn
import torch
import torchvision

import numpy

from sip.data_features import models

from collections import OrderedDict


class GPUPlugin(WorkerPlugin):

    def __init__(self, *, model, submodule_id, input_shape, batch_size):
        self.model = model
        self.submodule_id = submodule_id
        self.input_shape = input_shape
        self.batch_size = batch_size

    def setup(self, worker):
        worker.model_name = self.model
        worker.submodule_id = self.submodule_id
        worker.input_shape = self.input_shape
        worker.batch_size = self.batch_size

    def teardown(self, worker):
        del worker.model_name
        del worker.submodule_id
        del worker.input_shape
        del worker.batch_size


def get_activation(out):
    def hook(model, input, output):
        output = output.detach()
    return hook


def map_to_layer(partition):

    # check if model is already instantiated on worker
    worker = get_worker()
    if not hasattr(worker, "model"):
        # if the model is not yet present on the worker, instantiate it
        worker.model = models.get_untrained_model(worker.input_shape)

        # find index of named submodule so that the model can be clipped to this layer
        for i, (name, _) in enumerate(worker.model.named_modules()):
            if name == worker.submodule_id:
                idx = i
                break

        # create a new 'clipped' model
        worker.model = torch.nn.Sequential(
            OrderedDict(list(worker.model.children())[:idx + 1]))
        worker.intermediate_shape = worker.model(torch.randn(1, *worker.input_shape)).shape

    batch_size = worker.batch_size
    if len(partition) < worker.batch_size:
        batch_size = len(partition)

    dtype = partition[0]["single_blob_mask_img"].dtype
    full_out = numpy.zeros(
        shape=(len(partition),) + worker.intermediate_shape, dtype=dtype)
    batch_out = numpy.empty(
        shape=(batch_size,) + worker.intermediate_shape, dtype=dtype)
    (
        worker.model
        .get_submodule(worker.submodule_id)
        .register_forward_hook(get_activation(batch_out))
    )

    transform = torchvision.transforms.Compose(
        torchvision.transforms.CenterCrop(worker.input_shape[-1]),
        torchvision.transforms.ToTensor()
    )

    for i in range((len(partition) % batch_size)-1):

        batch = map(
            transform, 
            (partition[j]["single_blob_mask_img"] 
                for j in range(i * batch_size, (i + 1) * batch_size))
            )

        worker.model(batch)
        full_out[i * batch_size:(i + 1) * batch_size] = batch_out.numpy()

    return dask.dataframe.from_array(
        x=full_out, 
        columns=["model_%d" % i for i in range(worker.intermediate_shape[1])]
    )


def extract_features(*, images):
    """Extract intermediate representations from pretrained PyTorch models

    model: Path to PyTorch model
    intermediate_layer: Layer index from which to extract representations
    """

    # extract intermediate representations
    features = images.map_partitions(map_to_layer)

    return features
