# -*- coding: utf-8 -*-

import pathlib

__version__ = "0.2"


_root_path = pathlib.Path(__file__).parent.parent
_data_path = _root_path / "data"
_homag_path = _data_path / "homag"


class _DIR:

    ROOT: pathlib.Path = _root_path
    CONF: pathlib.Path = _root_path / "conf"

    DATA: pathlib.Path = _data_path
    CACHE: pathlib.Path = _data_path / "cache"

    HOMAG: pathlib.Path = _homag_path
    HOMAG_DIST: pathlib.Path = _homag_path / "dist"
    HOMAG_GRAPH: pathlib.Path = _homag_path / "graph"
    HOMAG_MATCHES: pathlib.Path = _homag_path / "matches"
    HOMAG_MODELS: pathlib.Path = _homag_path / "models"
    HOMAG_SOURCE: pathlib.Path = _homag_path / "source"
    HOMAG_TEXTS: pathlib.Path = _homag_path / "texts"
    HOMAG_CROSSVAL: pathlib.Path = _homag_path / "crossval"


class ENV:
    DIR = _DIR


class DraugError(Exception):
    pass
