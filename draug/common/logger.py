# -*- coding: utf-8 -*-

import draug

import os
import pathlib
import logging
import configparser

from logging.config import fileConfig


ENV_DRAUG_LOG_CONF = "DRAUG_LOG_CONF"
ENV_DRAUG_LOG_OUT = "DRAUG_LOG_OUT"


def init_logging():

    # probe environment for logging configuration:
    #   1. if conf/logging.conf exists use this
    #   2. if DRAUG_LOG_CONF is set as environment variable use its value
    #      as path to logging configuration

    fconf = None

    if ENV_DRAUG_LOG_CONF in os.environ:
        fconf = str(os.environ[ENV_DRAUG_LOG_CONF])

    else:
        path = pathlib.Path(draug.ENV.DIR.CONF / "logging.conf")
        if path.is_file():
            cp = configparser.ConfigParser()
            cp.read(path)

            opt = cp["handler_fileHandler"]
            (fname,) = eval(opt["args"])

            if ENV_DRAUG_LOG_OUT in os.environ:
                fname = pathlib.Path(os.environ[ENV_DRAUG_LOG_OUT])
            else:
                fname = draug.ENV.DIR.ROOT / fname

            fname.parent.mkdir(exist_ok=True, parents=True)
            fname.touch(exist_ok=True)
            opt["args"] = repr((str(fname),))

            fconf = cp

    if fconf is not None:
        fileConfig(cp)

    log = logging.getLogger(__name__)
    log.info("-" * 80)
    log.info("initialized logging")


# ---


# be nice if used as a library - do not log to stderr as default
log = logging.getLogger("draug")
log.addHandler(logging.NullHandler())
