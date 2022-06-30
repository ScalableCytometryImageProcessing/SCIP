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
from typing import Any, Callable, Mapping
from functools import partial

from scip.utils.util import copy_without


_OPS: Mapping[str, Callable[[numpy.ndarray], numpy.ndarray]] = {
    "max": partial(numpy.max, axis=1),
    "mean": partial(numpy.mean, axis=1)
}


def project_block(
    event: Mapping[str, Any],
    op: str
) -> numpy.ndarray:
    newevent = copy_without(event, without=["pixels"])
    newevent["pixels"] = _OPS[op](event["pixels"])

    return newevent
