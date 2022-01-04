# Copyright (C) 2022 Maxim Lippeveld
#
# This file is part of SCIP.
#
# SCIP is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SCIP is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with SCIP.  If not, see <http://www.gnu.org/licenses/>.

from scip.utils import util
import dask
from dask.distributed import get_worker
import time


def main():
    with util.ClientClusterContext(
            mode="mpi",
            port=9002,
            cores=6,
            memory=1,
            threads_per_process=1
    ):

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
