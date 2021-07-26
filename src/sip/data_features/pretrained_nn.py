from dask.distributed import WorkerPlugin, get_worker, get_client

import torch.nn
import torch

import numpy

from sip.data_features import models


class GPUPlugin(WorkerPlugin):

    def __init__(self, model, submodule_id, input_shape, batch_size):
        self.model = model
        self.submodule_id = submodule_id
        self.input_shape = input_shape
        self.batch_size = batch_size

    def setup(self, worker):
        worker.model = self.model
        worker.submodule_id = self.submodule_id
        worker.input_shape = self.input_shape
        worker.batch_size = self.batch_size

    def teardown(self, worker):
        del worker.model
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
        worker.model = models.get_random_model(worker.input_shape)

        # find index of named submodule so that the model can be clipped to this layer
        for i, (name, _) in enumerate(worker.model.named_modules()):
            if name == worker.submodule_id:
                idx = i
                break

        # create a new 'clipped' model
        worker.model = torch.nn.Sequential(*list(worker.model.children())[:idx + 1])
        worker.intermediate_shape = torch.randn(*worker.input_shape)

    if len(partition) < worker.batch_size:
        batch_size = len(partition)

    full_out = numpy.empty(
        shape=(len(partition),) + worker.intermediate_shape, dtype=partition.dtype)
    batch_out = numpy.empty(
        shape=(len(batch_size),) + worker.intermediate_shape, dtype=partition.dtype)
    (
        worker.model
        .get_submodule(worker.intermediate_layer)
        .register_forward_hook(get_activation(batch_out))
    )
    for i in range(partition % batch_size):
        worker.model([d["masked"] for d in partition[i * batch_size:(i + 1) * batch_size]])
        full_out[i * batch_size:(i + 1) * batch_size] = batch_out.numpy()


def extract_features(*, images, model, input_shape, intermediate_layer):
    """Extract intermediate representations from pretrained PyTorch models

    model: Path to PyTorch model
    intermediate_layer: Layer index from which to extract representations
    """

    model_plugin = GPUPlugin(model, intermediate_layer, input_shape, 32)

    # register plugin to instantiate models on the workers
    get_client().register_worker_plugin(model_plugin, name="model")

    # extract intermediate representations
    images.map_partitions(map_to_layer)

    # unregister plugin to remove models from the workers
    get_client().unregister_worker_plugin(name="model")
