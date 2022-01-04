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

import numpy
from typing import Callable
from functools import partial


_OPS: dict[str, Callable[[numpy.ndarray], numpy.ndarray]] = {
    "max": partial(numpy.max, axis=2),
    "mean": partial(numpy.mean, axis=2)
}


def project_block(
    block: numpy.ndarray,
    op: str
) -> numpy.ndarray:
    return _OPS[op](block)
