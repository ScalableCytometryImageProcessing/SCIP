from distributed.worker import get_client
from scip.utils import util
import dask
from dask.distributed import get_worker, get_client
import time


def main():
    with util.ClientClusterContext(
            mode="local",
            port=9002,
            cores=6,
            memory=1,
            threads_per_process=1
    ) as context:

        def func(part, gpu):
            print(get_client().cluster)
            time.sleep(1)
            return part

        df = dask.datasets.timeseries()

        # with dask.annotate(resources={"GPU": 1}):
        #     df = df.map_partitions(func, gpu=True, meta=df)
        df = df.map_partitions(func, gpu=False, meta=df)

        df.compute()

if __name__ == "__main__":
    main()