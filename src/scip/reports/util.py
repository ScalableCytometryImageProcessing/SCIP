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

from jinja2 import Environment, FileSystemLoader, select_autoescape


def get_jinja_template(template_dir, template):
    jinja_env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape()
    )
    jinja_template = jinja_env.get_template(template)

    return jinja_template
