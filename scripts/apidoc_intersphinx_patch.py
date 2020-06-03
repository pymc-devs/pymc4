#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A simple helper script that is used to replace the intersphinx mapping in
automatically generated sphinx apidoc.
"""

import sys
import os

INTERSPHINX_MAPPING = """intersphinx_mapping = {
    "python": ("https://user:password@docs.python.org/3", None),
    "tensorflow": ("https://www.tensorflow.org/api_docs/python", "https://github.com/mr-ubik/tensorflow-intersphinx/raw/master/tf2_py_objects.inv"),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "arviz": ("https://arviz-devs.github.io/arviz/", None),
}
"""

if len(sys.argv) > 1:
    conf_path = sys.argv[1]
else:
    conf_path = os.curdir()
if not conf_path.endswith("conf.py"):
    conf_path = os.path.join(conf_path, "conf.py")

content = []
with open(os.path.abspath(conf_path), "r") as f:
    for line in f.read().splitlines():
        if line.startswith("intersphinx_mapping"):
            line = INTERSPHINX_MAPPING
        content.append(line)
with open(os.path.abspath(conf_path), "w") as f:
    f.write(os.linesep.join(content))