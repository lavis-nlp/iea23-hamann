#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ktz.filesystem import path as kpath

import click
import pretty_errors

import os
import logging
import textwrap

from typing import Optional


# --- import order matters here:

from draug.common import logger

logger.init_logging()

import draug  # noqa: E402

# ---

log = logging.getLogger(__name__)
os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"


pretty_errors.configure(
    filename_display=pretty_errors.FILENAME_EXTENDED,
    lines_after=2,
    line_number_first=True,
)


def _create_banner(variables: dict[str, str]):
    variables = "\n".join(f"-  {k}: {v}" for k, v in variables.items())

    s = """
    --------------------
     DRAUG CLIENT
    --------------------

    variables in scope:
    {variables}
    """

    formatted = textwrap.dedent(s).format(variables=variables)
    return textwrap.indent(formatted, "  ")


@click.group()
def main():
    """
    DRAUG - empolis data adventures
    """
    log.info(" · DRAUG CLI ·")
    log.info(f"initialized path to draug: {draug.ENV.DIR.ROOT}")


def wrapped_main():
    try:
        main()
    except Exception as exc:
        log.error(str(exc))
        raise exc


# --- HOMAG


@main.group(name="homag")
def click_homag():
    """
    Homag data processing
    """
    pass


@click_homag.command(name="graph-import")
@click.option(
    "--name",
    type=str,
    required=True,
    help='name, e.g. "symptax.v6"',
)
@click.option(
    "--csv",
    type=str,
    required=True,
    help="csv file",
)
@click.option(
    "--raw/--transformed",
    type=bool,
    default=False,
    required=False,
    help="whether to apply graph transformations",
)
def click_homag_graph_import(name: str, csv: str, raw: bool):
    """
    Create a graph from a taxonomy export file
    """

    from draug.homag import graph

    graph.import_csv(
        name=name,
        path=kpath(csv, is_file=True),
        raw=raw,
    )


@click_homag.command(name="graph-import-undesired")
@click.option(
    "--graph",
    type=str,
    required=True,
    help="graph instance",
)
@click.option(
    "--undesired",
    type=str,
    required=True,
    help="undesired phrase mapping yaml",
)
def click_homag_graph_import_undesired(graph: str, undesired: str):
    """
    Associate undesired phrases/patterns to entities
    """

    from draug.homag.graph import Graph
    from draug.homag.graph import import_undesired

    graph = Graph.from_dir(path=graph)
    import_undesired(graph=graph, undesired=undesired)


@click_homag.command(name="graph-cli")
@click.option("--path", type=str, required=True, help="path to the graph")
def click_homag_graph_cli(path: str):
    """
    Load graph and drop into ipython
    """

    import IPython
    from draug.homag.graph import Graph

    path = kpath(path, is_dir=True)
    g = graph = Graph.from_dir(path)  # noqa: F841

    banner = _create_banner({"g, graph": "Graph"})
    IPython.embed(banner1=banner)


# ----------


@click_homag.command("text-split")
@click.option(
    "--tickets",
    type=str,
    required=True,
    help="source tickets xml",
)
@click.option(
    "--out-dir",
    type=str,
    required=True,
    help="directory to write to",
)
@click.option(
    "--ratio",
    type=float,
    required=True,
    help="split ration of train and test",
)
@click.option(
    "--seed",
    type=int,
    required=True,
    help="seed for reproduction",
)
@click.option(
    "--n",
    type=int,
    required=False,
    help="only process n tickets",
)
@click.option(
    "--min-len",
    type=int,
    required=False,
    help="at least n chararcters per sentence",
)
@click.option(
    "--max-len",
    type=int,
    required=False,
    help="at most n chars per sentence",
)
def click_homag_text_split(**kwargs):
    """
    Pre-process tickets and split them into train/test
    """

    from draug.homag import text

    text.split(**kwargs)


@click_homag.command(name="text-match")
@click.option(
    "--graph",
    type=str,
    required=True,
    help="path to the graph",
)
@click.option(
    "--out-dir",
    type=str,
    required=True,
    help="folder to write to",
)
@click.option(
    "--tickets",
    type=str,
    required=True,
    help="ticket xml",
)
@click.option(
    "--limit-tickets",
    type=int,
    required=False,
    default=None,
    help="yield at most n tickets",
)
@click.option(
    "--procs",
    type=int,
    required=False,
    default=1,
    help="concurrent processes for spacy",
)
def click_homag_match(
    graph: str,
    out_dir: str,
    tickets: str,
    limit_tickets: Optional[int],
    procs: int,
):
    """
    Sample textual mentions from preprocessed tickets
    """
    from draug.homag.graph import Graph
    from draug.homag import sampling

    graph = Graph.from_dir(path=kpath(graph, is_dir=True))
    yielder = sampling.yield_tickets(path=tickets, n=limit_tickets)
    handler = sampling.ResultWriter(graph=graph, out_dir=out_dir)

    nlp = sampling.get_nlp()

    sampling.match(
        nlp,
        entities={eid: graph.get_entity(eid) for eid in graph.eids},
        yielder=yielder,
        handler=handler,
        procs=procs,
        config=dict(
            graph=graph.name,
            ticket_path=tickets,
            graph_path=str(graph.path),
        ),
    )


@click_homag.command("crossval-create")
@click.option(
    "--k",
    type=int,
    help="how many folds",
)
@click.option(
    "--n",
    type=int,
    help="how many nodes (i.e. classes) per fold",
)
@click.option(
    "--seed",
    type=int,
    help="for reproduction",
)
@click.option(
    "--graph",
    type=str,
    help="graph directory",
)
@click.option(
    "--matches",
    type=str,
    help="matches file",
)
@click.option(
    "--out",
    type=str,
    help="output directory",
)
def click_homag_crossval_create(
    k: int,
    n: int,
    seed: int,
    graph: str,
    matches: str,
    out: str,
):
    """
    Create leave-n-out crossvalidation splits
    """
    from draug.homag import crossval

    crossval.create_split(
        k=k,
        n=n,
        seed=seed,
        graph_dir=kpath(graph, is_dir=True),
        matches_file=kpath(matches, is_file=True),
        out=kpath(out),
    )


@click_homag.command("crossval-apply")
@click.option(
    "--splits",
    type=str,
    required=True,
    help="directory with split data",
)
@click.option(
    "--graph",
    type=str,
    required=True,
    help="graph directory",
)
@click.option(
    "--dataset",
    type=str,
    required=True,
    help="directory with tokenized training data",
)
def click_homag_crossval_apply(
    splits: str,
    graph: str,
    dataset: str,
):
    """
    Create crossvalidation sub-datasets
    """
    from draug.homag import crossval

    crossval.apply_splits(
        splits=kpath(splits, is_dir=True),
        graph=kpath(graph, is_dir=True),
        dataset=kpath(dataset, is_dir=True),
    )


# --- MODELS


@main.group(name="models")
def click_models():
    """
    Empolis models
    """
    pass


@click_models.command(name="create-dataset")
@click.option(
    "--source",
    type=str,
    required=True,
    help="source files (e.g. match.txt)",
)
@click.option(
    "--out",
    type=str,
    required=True,
    help="output folder",
)
@click.option(
    "--graph",
    type=str,
    required=True,
    help="taxonomy folder",
)
@click.option(
    "--model",
    type=str,
    required=True,
    help="transformers model (e.g. bert-base-cased)",
)
def click_models_create_dataset(
    source: str,
    out: str,
    graph: str,
    model: str,
):
    from draug.models import data
    from draug.homag.graph import Graph

    graph = Graph.from_dir(graph)

    source = kpath(source, exists=True)
    out = kpath(out, exists=False, create=True)

    data.create_dataset(
        graph=graph,
        source=source,
        out=out,
        model=model,
    )


@click_models.command(name="split-dataset")
@click.option(
    "--matches",
    type=str,
    required=True,
    help="file containing matches",
)
@click.option(
    "--split-val",
    type=float,
    required=False,
    nargs=2,
    help="train/validation split sizes, e.g. 0.85 0.15",
)
@click.option(
    "--split-test",
    type=float,
    required=False,
    nargs=3,
    help="train/validation/test split sizes, e.g. 0.7 0.15 0.15",
)
@click.option(
    "--model",
    type=str,
    required=True,
    help="transformers model (e.g. bert-base-cased)",
)
@click.option(
    "--seed",
    type=int,
    required=True,
    help="split seed",
)
def click_models_split_dataset(
    matches: str,
    split_val: Optional[tuple[float]],
    split_test: Optional[tuple[float]],
    seed: int,
    model: str,
):
    from draug.models import data

    split = split_val if split_val is not None else split_test
    assert len(split) in {2, 3}

    data.split_dataset(
        path=kpath(matches, is_file=True),
        model=model,
        split=split,
        seed=seed,
    )


@click_models.command(name="create-queries")
@click.option(
    "--source",
    type=str,
    required=True,
    help="source file (e.g. nomatch.txt)",
)
@click.option(
    "--out",
    type=str,
    required=True,
    help="output directory",
)
@click.option(
    "--model",
    type=str,
    required=True,
    help="transformers model (e.g. bert-base-cased)",
)
def click_models_create_queries(
    source: str,
    out: str,
    model: str,
):
    from draug.models import data

    data.create_queries(
        source=kpath(source, is_file=True),
        out=kpath(out, is_dir=True),
        model=model,
    )


@click_models.command(name="train-model")
@click.option(
    "-c",
    "--config",
    type=str,
    multiple=True,
    required=True,
    help="configuration file",
)
@click.option(
    "--debug",
    type=bool,
    default=False,
    help="debug run",
)
@click.option(
    "--mask-prob",
    type=float,
    help="for 'query sometimes masked'",
)
@click.option(
    "--crossval-split",
    type=int,
    help="crossvalidation split'",
)
@click.option(
    "--name",
    type=str,
    help="overwrites the experiment name",
)
@click.option(
    "--seed",
    type=int,
    help="overwrites the seed",
)
def click_models_train(**kwargs):
    from draug.models import train

    train.main(**kwargs)


@click_models.command(name="validate-model")
@click.option(
    "--config",
    type=str,
    required=True,
    help="path to config.yml",
)
@click.option(
    "--ckpt",
    type=str,
    default=None,
    help="path to *.ckpt (vanilla otherwise)",
)
@click.option(
    "--n-batches",
    type=int,
    default=None,
    help="validate only N batches",
)
def click_models_validate(
    config: str,
    ckpt: Optional[str],
    n_batches: Optional[int],
):
    from draug.models import predict

    predict.validate(
        config_file=kpath(config, is_file=True),
        ckpt_file=kpath(ckpt, is_file=True) if ckpt else None,
        n_batches=n_batches,
    )


@click_models.command(name="rank-queries")
@click.option(
    "--config",
    required=True,
    help="path to config.yml",
)
@click.option(
    "--ckpt",
    required=True,
    help="path to *.ckpt",
)
@click.option(
    "--queries",
    required=True,
    help="path to query samples",
)
@click.option(
    "--n-batches",
    type=int,
    default=None,
    help="only run n batches",
)
@click.option(
    "--threshold",
    type=float,
    default=None,
    help="score threshold (unbounded)",
)
@click.option(
    "--threshold-norm",
    type=float,
    default=None,
    help="normalized score threshold 0 <= threshold <= 1",
)
def click_models_rank(*args, **kwargs):
    from draug.models import predict

    predict.rank(*args, **kwargs)


@click_models.command(name="rank-queries-2")
@click.option(
    "--name",
    required=True,
    help="name of the ranking: write to $model/ranking/$dir",
)
@click.option(
    "--config",
    required=True,
    help="path to config.yml",
)
@click.option(
    "--ckpt",
    required=True,
    help="path to *.ckpt",
)
@click.option(
    "--queries",
    required=True,
    help="path to query samples",
)
@click.option(
    "--n-batches",
    type=int,
    default=None,
    help="only run n batches",
)
def click_models_rank2(*args, **kwargs):
    from draug.models import ranking

    ranking.rank(*args, **kwargs)


@click_models.command(name="crossvalidate-queries")
@click.option(
    "--config",
    required=True,
    help="path to config.yml",
)
@click.option(
    "--ckpt",
    required=True,
    help="path to *.ckpt",
)
@click.option(
    "--queries",
    required=True,
    help="path to query samples",
)
def click_models_crossvalidate(*args, **kwargs):
    from draug.models import predict

    predict.crossvalidate(*args, **kwargs)
