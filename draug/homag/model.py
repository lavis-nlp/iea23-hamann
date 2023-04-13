# -*- coding: utf-8 -*-

from ktz.filesystem import path as kpath

from draug.homag import open_kwargs
from draug.homag.graph import Graph
from draug.homag.graph import NID

import logging
from pathlib import Path
from itertools import count
from itertools import islice
from dataclasses import fields
from dataclasses import dataclass
from collections import defaultdict

from typing import Union
from typing import Optional
from typing import Iterable


log = logging.getLogger(__name__)
SEP = "|"


@dataclass(frozen=True, eq=True)
class PredictionRow:  # v7

    # order determines csv column order
    score_norm: float
    score: float

    nid: int
    fqn: str  # e.g. 1:1:geraeusche (+1) (not interesting here)

    relation: Graph.RELATIONS

    eid: str  # unused
    ticket: str
    context: str

    def __lt__(self, other):
        return (self.score_norm, self.score) < (other.score_norm, other.score)

    # --- persistence

    @classmethod
    def from_str(Cls, rep: str):
        return Cls.from_col(rep.strip().split(SEP))

    @classmethod
    def from_col(Cls, col: Iterable[str]):
        lis = tuple(col)

        def _conv(field, item, annotation):
            if annotation is str:
                return item

            if annotation is int:
                return int(item)

            if annotation is float:
                return float(item)

            # instance/type checks don't work for GenericAlias
            if repr(annotation) == "<enum 'RELATIONS'>":
                if item == "parent":
                    return Graph.RELATIONS.parent
                elif item == "synonym":
                    return Graph.RELATIONS.synonym

            assert False, f"type conversion failed for {field}: {item}"

        # score, score_norm, ...
        fieldlis = fields(Cls)
        assert len(fieldlis) == len(lis)

        kwargs = {
            field.name: _conv(field.name, item, field.type)
            for field, item in zip(fieldlis, lis)
        }

        return Cls(**kwargs)


PID = int  # prediction identifier


@dataclass(frozen=True, eq=True)
class Prediction:

    pid: PID
    nid: NID

    score_norm: float
    score: float

    relation: Graph.RELATIONS
    context: str

    def __lt__(self, other):
        return (self.score_norm, self.score) < (other.score_norm, other.score)


class Predictions:
    def __str__(self):
        return (
            f"draug predictions: {len(self._pid2pred)} total "
            f"(nids: {len(self._by_nid)})"
        )

    def __init__(self):
        self._known = set()
        self._counter = count()  # assigned pids

        self._pid2pred: dict[PID, Prediction] = {}
        self._ctx2pids: dict[str, set[PID]] = defaultdict(set)
        self._by_nid = defaultdict(lambda: defaultdict(set))

    def count_by_nid(self, nid: NID) -> dict[Graph.RELATIONS, int]:
        ret = {}

        for rel in Graph.RELATIONS:
            ret[rel] = len(list(self._by_nid[nid][rel]))

        return ret

    def by_nid(self, nid: NID) -> dict[Graph.RELATIONS, list[Prediction]]:

        # This is a performance problem if (in the downstream web app)
        # many requests are expected. Currently negligible.
        # Later: use heapq and add caching.

        ret = {}

        for rel in Graph.RELATIONS:
            pids = list(self._by_nid[nid][rel])
            ret[rel] = sorted((self._pid2pred[pid] for pid in pids), reverse=True)

        return ret

    def by_pid(self, pid: PID) -> Prediction:
        return self._pid2pred[pid]

    def del_prediction(self, pid: PID):
        log.info(f"delete prediction {pid}")

        try:
            pred = self._pid2pred[pid]
        except KeyError:
            log.error(f"prediction {pid} vanished!")
            return

        # delete all other predictions with the same context
        pids = self._ctx2pids[pred.context]
        del self._ctx2pids[pred.context]
        for pid in pids:
            pred = self._pid2pred[pid]

            del self._pid2pred[pid]
            self._by_nid[pred.nid][pred.relation] -= {pid}

    # ---

    def _add_prediction(self, row: PredictionRow):
        if row in self._known:
            log.warning("trying to add {row} multiple times")
            return

        self._known.add(row)
        pid = next(self._counter)

        pred = Prediction(
            pid=pid,
            nid=row.nid,
            score_norm=row.score_norm,
            score=row.score,
            relation=row.relation,
            context=row.context,
        )

        self._pid2pred[pid] = pred
        self._ctx2pids[pred.context].add(pid)
        self._by_nid[pred.nid][pred.relation].add(pid)

    @classmethod
    def from_files(
        Cls,
        *paths: Union[str, Path],
        n: Optional[int] = None,
    ):
        self = Cls()

        for path in paths:
            path = kpath(path, is_file=True)

            with path.open(mode="r", **open_kwargs) as fd:
                fd.readline()  # skip header
                predictions = map(PredictionRow.from_str, islice(fd, n))

                for prediction in predictions:
                    self._add_prediction(prediction)

        return self
