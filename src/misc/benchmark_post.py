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

import pandas
from pathlib import Path
import json


def main(d):

    d = Path(d)

    timing_data = []
    for timing in d.glob("*.json"):
        with open(timing) as fp:
            timing_data.append(json.load(fp))
    pandas.DataFrame.from_records(timing_data).to_csv(
        str(d / 'timing_results.csv'), index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", type=str)

    args = parser.parse_args()

    main(args.dir)
