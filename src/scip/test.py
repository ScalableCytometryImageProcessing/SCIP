from distributed.worker import get_client
from scip.utils import util
import dask
from dask.distributed import get_worker, get_client
import time


def main():
    with util.ClientClusterContext(
            mode="mpi",
            port=9002,
            cores=6,
            memory=1,
            threads_per_process=1
    ) as context:

        @dask.delayed
        def func(gpu):
            print(get_worker().name, gpu)
            time.sleep(1)
            return True

        d = []
        with dask.annotate(resources={"cellpose": 1}):
            for _ in range(5):
                d.append(func(gpu=True))
            
        for _ in range(5):
            d.append(func(gpu=False))

        dask.compute(d)

if __name__ == "__main__":
    main()