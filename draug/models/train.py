# -*- coding: utf-8 -*-

import draug
from draug.models import data
from draug.models import models
from draug.common import ryaml
from draug.homag import crossval
from draug.homag.graph import Graph

import yaml
from ktz.filesystem import path as kpath

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from tqdm import tqdm as _tqdm

import random
import logging
from pathlib import Path

from functools import partial
from datetime import datetime

from typing import Optional


log = logging.getLogger(__name__)
tqdm = partial(_tqdm, ncols=80)


def join(defaults: dict, **kwargs):
    return kwargs | defaults


def _init_datamodule(config: dict):
    ds_dir = kpath(config["dataset_dir"], is_dir=True)
    tokenizer = ds_dir

    if "crossvalidation" in config:
        cvc = config["crossvalidation"]
        name = crossval.create_name(cvc)
        log.info(f"! using crossvalidation setting: {name}")
        ds_dir = ds_dir / "crossval" / name / f"split.{cvc['split']}"

    graph = Graph.from_dir(path=config["graph_dir"])
    datamodule = data.DataModule.create(
        path=str(ds_dir),
        graph=graph,
        config=config,
        tokenizer=tokenizer,
    )

    return datamodule


def train(config: dict):
    debug: bool = config["trainer"]["fast_dev_run"]
    if debug:
        log.warning("running in debug mode")

    logger = WandbLogger(project="draug", log_model=False, name=config["name"])
    logger.experiment.config.update(config, allow_val_change=False)

    datamodule = _init_datamodule(config=config)

    model = models.get_cls(name=config["model"])(
        config=config,
        bert=config["transformer"],
        datamodule=datamodule,
        **config["model_kwargs"],
    )

    out_dir = Path(config["out_dir"])

    checkpoint_args = join(
        config["checkpoint"],
        dirpath=out_dir / "checkpoints",
    )

    callbacks = [
        pl.callbacks.ModelCheckpoint(**checkpoint_args),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
    ]

    if "early_stopping" in config:
        log.info("add early stopping callback")
        EarlyStopping = pl.callbacks.early_stopping.EarlyStopping
        callbacks.append(EarlyStopping(**config["early_stopping"]))

    # training

    if not debug:
        with (out_dir / "config.yml").open(mode="w") as fd:
            yaml.dump(config, fd)

    trainer_args = join(
        config["trainer"],
        logger=logger,
        callbacks=callbacks,
        weights_save_path=out_dir / "weights",
    )

    trainer = pl.Trainer(**trainer_args)

    log.error("pre-training validation disabled")
    # trainer.validate(model, datamodule=datamodule)

    trainer.fit(model, datamodule=datamodule)


def add_logger(config: dict):
    loghandler = logging.FileHandler(
        str(kpath(config["out_dir"]) / "training.log"),
        mode="w",
    )

    loghandler.setLevel(log.getEffectiveLevel())
    loghandler.setFormatter(log.root.handlers[0].formatter)

    logging.getLogger("draug").addHandler(loghandler)


def prepare_environment(config):
    data_path = kpath(
        draug.ENV.DIR.HOMAG_MODELS / config["source"] / config["transformer"],
        is_dir=True,
        message="using data directory {path_abbrv}",
    )

    # construct graph and dataset paths
    config["dataset_dir"] = str(data_path / "dataset")
    graph_path = draug.ENV.DIR.HOMAG_GRAPH / config["graph"]
    config["graph_dir"] = str(graph_path)

    # create output folder path
    timestamp = datetime.now().strftime("%y.%m.%d-%H.%M.%S")
    out_name = config["name"].replace(" ", "_")
    out_dir = data_path / "models" / f"{timestamp}-{out_name}"
    config["out_dir"] = str(out_dir)

    # only create folder for non-debug runs
    if not config["trainer"]["fast_dev_run"]:
        out_dir.mkdir(exist_ok=True, parents=True)
        add_logger(config)


def set_seed(config, seed: Optional[int]):
    if seed:
        log.info("overwriting seed with cl arg")
        config["seed"] = seed

    if "seed" not in config:
        log.info("generating random seed")
        config["seed"] = random.randint(10 ** 5, 10 ** 7)

    log.info(f"! setting seed: {config['seed']}")
    pl.utilities.seed.seed_everything(config["seed"])


def check_config(
    config: dict,
    debug: bool,
    mask_prob: Optional[float],
    crossval_split: Optional[int],
    name: Optional[str],
):

    #  overwrites

    if mask_prob is not None:
        assert 0 < mask_prob < 1
        log.info(f"! overwriting mask probability with {mask_prob}")
        config["train_dataset_kwargs"]["mask_prob"] = mask_prob

    if crossval_split is not None:
        assert "crossvalidation" in config
        log.info(f"! overwriting crossvalidation split with {crossval_split}")
        config["crossvalidation"]["split"] = crossval_split

    if name:
        log.info(f"! overwriting the experiment name with {name}")
        config["name"] = name

    #  additional settings

    config["trainer"]["fast_dev_run"] = debug


def main(
    config: list[str],
    seed: Optional[int],
    **kwargs,
):
    config = ryaml.load(configs=config)
    check_config(config, **kwargs)
    set_seed(config=config, seed=seed)
    prepare_environment(config=config)
    train(config=config)
