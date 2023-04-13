# -*- coding: utf-8 -*-

from draug.homag import graph
from draug.models import data
from draug.common import helper
from draug.homag.graph import EID
from draug.homag.graph import NID
from draug.homag.graph import Graph
from draug.homag.graph import Entity
from draug.homag.text import Matches

import yaml

import random
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import field
from dataclasses import asdict
from dataclasses import dataclass
from collections import defaultdict

from typing import Any
from typing import Union
from typing import Optional


log = logging.getLogger(__name__)


def create_name(config):
    k, n, seed = config["k"], config["n"], config["seed"]
    name = f"k-{k}.n-{n}.{seed}"
    return name


@dataclass(frozen=True, order=True)
class Sample:

    eid: int
    nid: int
    rid: int
    weight: int


@dataclass
class Split:

    samples: set[Sample]
    graph: Optional[Graph] = field(default=None)

    # those removed from the graph
    entities: Optional[dict[EID, Entity]] = field(default=None)


class Samples:

    config: dict[Any, Any]
    splits: dict[int, Split]

    def _create_graph(self, graph: Graph, samples: set[Sample]) -> Graph:
        graph = Graph.from_dir(path=graph.path)
        ents = {}

        for eid in {s.eid for s in samples}:
            ents[eid] = ent = graph.get_entity(eid=eid)
            graph.del_entity(eid=ent.eid)

            if not len(graph.get_entities(nid=ent.nid)):
                deleted = graph.del_node(nid=ent.nid)
                assert (
                    len(deleted) == 1 and deleted[0] == ent.nid
                ), f"{ent.nid=} {deleted=}"

        return graph, ents

    def _load_samples(self, fname: Union[str, Path]) -> set[Sample]:
        fname = helper.path(fname, is_file=True)

        with fname.open(mode="r") as fd:
            next(fd)  # consume header
            ret = set(map(lambda s: Sample(*map(int, s.split(","))), fd.readlines()))

        return ret

    def __init__(self, path: Union[str, Path], graph: Optional[Graph] = None):
        log.info(f"loading crossvalidation samples from {path}")
        path = helper.path(path, is_dir=True)

        with (path / "config.yml").open(mode="r") as fd:
            self.config = yaml.load(fd.read(), Loader=yaml.FullLoader)

        self.splits = {}
        for i in range(self.config["split"]["k"]):

            kwargs = dict(samples=self._load_samples(path / f"split.{i}.csv"))

            if graph is not None:
                subgraph, entities = self._create_graph(
                    graph=graph,
                    samples=kwargs["samples"],
                )
                kwargs |= dict(graph=subgraph, entities=entities)

            self.splits[i] = Split(**kwargs)


def select(
    graph_dir: Path,
    matches: Matches,
    n: int,
    seed: int,
    ignore: set[EID],
) -> tuple[Graph, set[Sample]]:

    SYN = Graph.RELATIONS.synonym.value
    PAR = Graph.RELATIONS.parent.value

    # load locally to ensure an unmodified graph
    g = Graph.from_dir(path=graph_dir)

    candidates = sorted(g.eids - ignore)
    random.shuffle(candidates)
    assert n < len(candidates)

    # gathers entities to be used for validation
    agg: dict[NID, set[Sample]] = defaultdict(set)
    retain: set[NID] = set()  # keeps track of parents

    while candidates and len(agg) < n:
        eid = candidates.pop()
        assert eid in g.eids, f"{eid=}"

        ent = g.get_entity(eid=eid)
        weight = len(matches.by_eid(eid=eid))

        nid = ent.nid
        pid = g.get_parent(nid=nid)

        synonyms = g.get_entities(nid=nid)
        children = g.get_children(nid=nid)

        if (
            # _nodes_ that must not be removed if
            #   1. they are root nodes
            #   2. they are parents
            (len(synonyms) == 1 and (pid is None or nid in retain or len(children)))
            # _entities_ that must not be removed:
            or (weight < data.MIN_SAMPLES)
        ):
            continue

        # add parents for non-roots
        if pid is not None:
            agg[pid].add(Sample(eid=ent.eid, nid=pid, rid=PAR, weight=weight))
            retain.add(pid)

        agg[nid].add(Sample(eid=ent.eid, nid=nid, rid=SYN, weight=weight))
        g.del_entity(eid=ent.eid)

        # if the node is empty, it can be removed
        if len(g.get_entities(nid=nid)) == 0:
            deleted = g.del_node(nid=nid)

            assert len(children) == 0 and len(deleted) == 1
            assert all(sample.rid == SYN for sample in agg[nid])

            del agg[nid]

    for nid in g.nids:
        assert len(g.get_entities(nid=nid))

    return g, {sample for col in agg.values() for sample in col}


# TODO use Samples for persistence (from_dir etc.)


def create_samples(
    graph_dir: Path,
    matches: Matches,
    i: int,
    k: int,
    n: int,
    seed: int,
    ignore: set[EID],
) -> set[Sample]:

    g, samples = select(
        graph_dir=graph_dir,
        matches=matches,
        n=n,
        seed=seed,
        ignore=ignore,
    )

    name = create_name(dict(k=k, n=n, seed=seed))
    suffix = f"split-{i}"

    g.meta["name"] = name + "." + suffix
    out = graph_dir / "crossval" / name / suffix

    g.to_dir(path=out)
    graph.save_gml(graph=g, out=out / "graph.gml")

    return samples


def write_samples(out: Path, i: int, samples: set[Sample]):
    with (out / f"split.{i}.csv").open(mode="w") as fd:
        fd.write(",".join(asdict(list(samples)[0]).keys()) + "\n")
        for sample in samples:
            fd.write(",".join(map(str, asdict(sample).values())) + "\n")


# k-fold with n samples
def create_split(
    k: int,
    n: int,
    seed: int,
    graph_dir: Path,
    matches_file: Path,
    out: Path,
):
    name = create_name(dict(k=k, n=n, seed=seed))
    out = helper.path(out / name, create=True)

    g = Graph.from_dir(path=graph_dir)
    print(g.description)

    matches = Matches.from_file(path=matches_file, graph=g)
    print(str(matches))

    print(f"setting {seed=}")
    random.seed(seed)

    sampled: set[EID] = set()
    for i in range(k):
        print(f"\ncreate split {i} ({len(sampled)} sampled so far)")

        samples = create_samples(
            graph_dir=graph_dir,
            matches=matches,
            i=i,
            k=k,
            n=n,
            seed=seed,
            ignore=sampled,
        )

        sampled |= {sample.eid for sample in samples}
        weight = sum(sample.weight for sample in samples)

        print(f"created split {i}: total text samples: {weight}")
        write_samples(out=out, i=i, samples=samples)

    out = helper.path(out, is_dir=True)
    with (out / "config.yml").open(mode="w") as fd:
        yaml.dump(
            {
                "split": {
                    "k": k,
                    "n": n,
                    "seed": seed,
                },
                "created": datetime.now(),
                "graph": g.name,
                "graph_dir": str(g.path),
                "matches_file": str(matches_file),
            },
            fd,
        )


#
#  RE-ORDER SAMPLED MENTIONS
#
def _load_matches(path) -> dict[EID, list[data.TextSample]]:
    in_file = path / f"matches.{data.Split.ALL.value}"
    with in_file.open(mode="rb") as fd:
        gen = map(data.TextSample.from_bytes, fd)
        agg = helper.agg((sample.match.eid, sample) for sample in gen)

    return agg


def _load_nomatches(path) -> list[data.QuerySample]:
    in_file = path / f"nomatches.{data.Split.ALL.value}"
    with in_file.open(mode="rb") as fd:
        agg = list(map(data.QuerySample.from_bytes, fd))

    return agg


def apply_splits(
    splits: Path,
    graph: Path,
    dataset: Path,
):

    print("loading queries...")
    nomatches = tuple(_load_nomatches(dataset))
    print(f"got {len(nomatches)} nomatches")

    print("loading matches...")
    matches = _load_matches(dataset)
    print(f"loaded text samples for {len(matches)} eids")
    print(f"got {sum(len(m) for m in matches.values())} matches")

    print("loading graph...")
    graph = Graph.from_dir(path=graph)
    print(f"loaded graph: {graph.name}")

    samples = Samples(path=splits, graph=graph)
    print("loaded crossvalidation splits")

    name = create_name(samples.config["split"])

    for i, split in samples.splits.items():
        print(f"\ndistribute data for split {i}")

        out_dir = dataset / "crossval" / name / f"split.{i}"
        out_dir = helper.path(out_dir, create=True)

        # move text samples to query samples
        query_samples = list(nomatches)  # new instance
        for eid in split.entities:
            for text_sample in matches[eid]:
                query_samples.append(text_sample.to_query)

        # flatten text samples without split entities
        text_samples = []
        for eid in set(matches) - set(split.entities):
            text_samples += matches[eid]

        # write out
        with (out_dir / f"matches.{data.Split.ALL.value}").open(mode="wb") as fd:
            fd.writelines(map(lambda t: t.to_bytes, text_samples))

        with (out_dir / f"nomatches.{data.Split.ALL.value}").open(mode="wb") as fd:
            fd.writelines(map(lambda t: t.to_bytes, query_samples))
