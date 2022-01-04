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

from pathlib import Path
from datetime import datetime
import uuid
import pandas
import os


def main():

    output = Path(os.environ["VSC_DATA_VO_USER"]) / "results/scip_benchmark"
    output = output / Path("benchmark_%s" % datetime.now().strftime("%Y%m%d%H%M%S"))
    output.mkdir()
    (output / "results").mkdir()

    iterations = 5
    total_mem = 120

    commands = []
    for partition_size in [100, 200, 400, 800, 1600]:
        for n_workers in [1, 2, 4, 8, 16, 26]:
            for _ in range(iterations):
                ident = uuid.uuid4()

                o = str(output / "results" / str(ident))

                commands.append(dict(
                    n_workers=n_workers,
                    memory=total_mem // n_workers,
                    partition_size=partition_size,
                    output=o,
                    np=n_workers + 2,
                    prefix=str(output)
                ))

    pandas.DataFrame(commands).to_csv(str(output / "data.csv"), index=False)


if __name__ == "__main__":
    main()
