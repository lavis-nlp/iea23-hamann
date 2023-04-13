# -*- coding: utf-8 -*-

# new approach to hold query ranks
# replaces rankings in predict.py


import draug
from draug.homag import text
from draug.models import data
from draug.models import models
from draug.models import predict
from draug.homag.graph import NID
from draug.homag.graph import Graph

from ktz.string import args_hash
from ktz.string import encode_line
from ktz.string import decode_line
from ktz.filesystem import path as kpath

import yaml
import h5py
import numpy as np
from tqdm import tqdm as _tqdm

import torch
import torch.nn.functional as F
from torch.utils import data as td

import gzip
import logging
import textwrap
import contextlib
from pathlib import Path
from itertools import count
from itertools import islice
from functools import partial
from dataclasses import dataclass

from typing import Union
from typing import Literal
from typing import BinaryIO
from typing import Optional
from collections.abc import Iterator


tqdm = partial(_tqdm, ncols=80)
log = logging.getLogger(__name__)


PID = int


class Context:

    fd_h5: h5py.File
    fd_preds: BinaryIO

    queries: td.DataLoader
    model: models.DraugModule
    datamodule: data.DataModule


class Manager(contextlib.AbstractContextManager):

    ckpt_file: Path
    report_dir: Path
    queries_file: Path

    model_config: dict

    def _enter(self, fn, *args, **kwargs):
        return self._stack.enter_context(
            fn(*args, **kwargs),
        )

    def __init__(
        self,
        report_dir: Path,
        queries_file: Path,
        ckpt_file: Path,
        model_config: dict,
    ):
        log.info("setting up ranking manager")

        self.report_dir = report_dir
        self.model_config = model_config
        self.queries_file = queries_file
        self.ckpt_file = ckpt_file

    # --

    def _enter_fds(self, ctx: Context):

        ctx.fd_h5 = self._enter(
            h5py.File,
            self.report_dir / "rankings.h5",
            mode="w",
        )

        ctx.fd_preds = self._enter(
            gzip.open,
            self.report_dir / "predictions.txt.gz",
            mode="w",
        )

    # --

    def _load_model(self, ctx: Context):

        ds_queries = data.QueryDataset(path=self.queries_file)
        ctx.queries = td.DataLoader(
            ds_queries,
            collate_fn=ds_queries.collate_fn,
            batch_size=self.model_config["valid_loader"]["batch_size"],
            shuffle=False,
        )

        ctx.datamodule = predict.load_data(
            config=self.model_config,
        )

        ctx.model = predict.load_model(
            config=self.model_config,
            ckpt_file=self.ckpt_file,
            datamodule=ctx.datamodule,
        )

        ctx.model.eval()

    # --

    def __enter__(self):
        log.info("entering ranking context")
        self._stack = contextlib.ExitStack().__enter__()

        ctx = Context()
        self._enter_fds(ctx)
        self._load_model(ctx)

        return ctx

    def __exit__(self, *args):
        log.info("leaving ranking context")
        return self._stack.__exit__(*args)


class Ranker:

    rank_config_file: Path
    n_batches: Optional[int]

    name: str
    ctx: Context

    def __init__(
        self,
        name: str,
        queries: Union[str, Path],
        ckpt: Union[str, Path],
        model_config: dict,
        report_dir: Union[str, Path],
        n_batches: Optional[int] = None,
    ):
        self.name = name
        self.manager = Manager(
            report_dir=kpath(report_dir, is_dir=True),
            queries_file=kpath(queries, is_file=True),
            ckpt_file=kpath(ckpt, is_file=True),
            model_config=model_config,
        )

        self.n_batches = n_batches
        log.info(f"ranking directory: {report_dir}")

    # -- create ranking if not cached

    def run(self):

        # --

        def _yield_query_results(ctx: Context):

            gen = islice(ctx.queries, self.n_batches)
            total = self.n_batches or len(ctx.queries)

            for batch in tqdm(gen, total=total):
                batch = ctx.datamodule.transfer_batch_to_device(
                    batch,
                    ctx.model.device,
                    0,  # TODO hard-coded dataloader index
                )

                yield from ctx.model.query(*batch)

        with self.manager as ctx:
            # ctx: Context

            shape = (
                len(ctx.queries.dataset),
                len(Graph.RELATIONS),
                ctx.model.entities.num_embeddings,
            )

            log.info(f"create h5/scores dataset of size {shape}")
            scores = ctx.fd_h5.create_dataset(
                "scores",
                shape,  # P x R x E
                dtype=float,
            )

            # --

            pids: Iterator[PID] = count()
            contexts: dict[str, PID] = {}
            pid_bar = tqdm(unit=" pids", position=1)

            for res in _yield_query_results(ctx=ctx):

                context = res.query.nomatch.context
                identifier = res.query.nomatch.identifier

                if context not in contexts:
                    contexts[context] = next(pids)

                    prediction = (contexts[context], identifier, context)
                    rep = encode_line(prediction, sep=text.SEP, fn=str)

                    ctx.fd_preds.write(rep)
                    pid_bar.update()

                pid = contexts[context]

                # idxs and scores are given in sorted order
                # for h5py we need to map them to index order
                idxs = np.array(res.prediction.idxs)
                scrs = np.array(res.prediction.scores)
                scores[pid, res.prediction.r_idx, :] = scrs[idxs.argsort()]


def setup(
    name: str,
    report_dir: Path,
    model_config: dict,
    hash: str,
    **kwargs,
):
    ranking_config_file_path = report_dir / "config.yml"

    # run if not yet computed

    if not ranking_config_file_path.exists():
        ranking_config = (
            dict(
                name=name,
                hash=hash,
            )
            | kwargs
        )

        with ranking_config_file_path.open(mode="w") as fd:
            yaml.dump(ranking_config, fd)

        ranker = Ranker(
            name=name,
            model_config=model_config,
            report_dir=report_dir,
            **kwargs,
        )

        ranker.run()

    # load if already computed

    else:
        with ranking_config_file_path.open(mode="r") as fd:
            ranking_config = yaml.load(fd, Loader=yaml.FullLoader)

        if ranking_config["hash"] != hash:
            msg = "hash value mismatch - same name, different parameters?"
            print(msg)
            log.error(msg)
            exit(2)

    return ranking_config


@dataclass(frozen=True)
class Prediction:

    pid: int
    nid: int
    relation: Graph.RELATIONS

    score: float
    score_norm: float

    context: str


class Predictions:

    scores: np.ndarray  # PID x REL x NID
    scores_norm: np.ndarray  # PID x REL x NID

    contexts: dict[PID, str]

    def __init__(
        self,
        graph: Graph,
        preds_path: Union[str, Path],
        h5_path: Union[str, Path],
        normalization: Literal["sigmoid", "softmax"],
    ):
        log.info("copying score database to memory")
        with h5py.File(kpath(h5_path, is_file=True), mode="r") as h5:
            self.scores = h5["scores"][:]
            log.info(f"loaded scores data: {self.scores.shape}")

        log.info(f"got score database of size {self.scores.shape}")
        log.info(f"creating normalised scores ({normalization})")

        normalizer = {
            "softmax": dict(fn=F.softmax, kwargs=dict(dim=-1)),
            "sigmoid": dict(fn=torch.sigmoid, kwargs={}),
        }

        assert normalization in normalizer, "unknown normalizer"

        self.scores_norm = normalizer[normalization]["fn"](
            torch.Tensor(self.scores),
            **normalizer[normalization]["kwargs"],
        ).numpy()

        log.info("loading contexts")
        self.contexts = {}
        with gzip.open(kpath(preds_path, is_file=True), mode="r") as fd:
            for line in fd:
                pid, _, context = decode_line(line, sep=text.SEP, fns=(int, str, str))
                self.contexts[pid] = context

        log.info(f"got {len(self.contexts)} contexts")

    def top_k(
        self,
        nid: NID,
        k: int = 10,
        norm: bool = True,
        noskip: bool = False,  # TODO
    ) -> dict[Graph.RELATIONS, tuple[Prediction]]:

        agg = {rel: [] for rel in Graph.RELATIONS}

        def _return(agg):
            return {k: tuple(v) for k, v in agg.items()}

        # only the case when adding/removing nodes after ranking
        if nid >= self.scores.shape[2]:
            return _return(agg)

        for rel in agg:
            scores = self.scores[:, rel.value, nid]
            norms = self.scores_norm[:, rel.value, nid]

            # tie breaking
            zipped = np.array(
                list(zip(scores, norms)),
                dtype=[("scores", "float"), ("norms", "float")],
            )

            order = ("norms", "scores") if norm else ("scores", "norms")
            pids = list(np.argsort(zipped, order=order))

            # pids may have been deleted
            while pids and (k is None or len(agg[rel]) < k):
                pid = pids.pop()

                # skip deleted predictions
                if pid not in self.contexts:
                    if noskip:
                        k -= 1
                    continue

                pred = Prediction(
                    pid=pid,
                    nid=nid,
                    relation=rel,
                    context=self.contexts[pid],
                    score=float(scores[pid]),
                    score_norm=float(norms[pid]),
                )

                agg[rel].append(pred)

        return _return(agg)

    # ---
    # RUI interface
    # to be changed in the future

    def count_by_nid(self, nid: NID):
        return {rel: 1234 for rel in Graph.RELATIONS}

    def del_prediction(self, pid: PID):
        del self.contexts[pid]


def _create_banner(variables: dict[str, str]):
    variables = "\n".join(f"-  {k}: {v}" for k, v in variables.items())

    s = """
    -------------------------
     DRAUG PREDICTIONS CLIENT
    -------------------------

    variables in scope:
    {variables}
    """

    formatted = textwrap.dedent(s).format(variables=variables)
    return textwrap.indent(formatted, "  ")


# What inspired you to build a second Krusty Krab
# right next door to the original?
#
# will write results to $model/ranking/$name
def rank(name: str, config: str, **kwargs):

    model_config_file = kpath(config, is_file=True)
    model_config = predict.load_model_config(
        config_file=model_config_file,
    )

    report_dir = kpath(
        model_config_file.parent / "ranking" / name,
        create=True,
    )

    hash = args_hash(name, config, kwargs)
    setup(
        name=name,
        report_dir=report_dir,
        model_config=model_config,
        hash=hash,
        **kwargs,
    )

    print("load graph")
    graph = Graph.from_dir(draug.ENV.DIR.HOMAG_GRAPH / model_config["graph"])

    print("load predictions")

    Predictions(
        graph=graph,
        preds_path=report_dir / "predictions.txt.gz",
        h5_path=report_dir / "rankings.h5",
        normalization="softmax",
    )

    # preds = Predictions(...
    # import IPython

    # banner = _create_banner({"preds": "Predictions", "graph": "Graph"})
    # IPython.embed(banner1=banner)
