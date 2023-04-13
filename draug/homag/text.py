# -*- coding: utf-8 -*-

from draug.homag import open_kwargs
from draug.homag.graph import Graph

from ktz.filesystem import path as kpath

import yaml

from deepca import dumpr
from tqdm import tqdm as _tqdm

import re
import random
import logging

from pathlib import Path
from itertools import islice
from functools import partial
from datetime import datetime
from dataclasses import field
from dataclasses import fields
from dataclasses import astuple
from dataclasses import dataclass
from collections import Counter
from collections import defaultdict

import typing
from typing import Any
from typing import Union
from typing import Optional
from typing import Iterable


log = logging.getLogger(__name__)
tqdm = partial(_tqdm, ncols=80)
SEP = "|"


@dataclass(eq=True)
class _Base:
    @property  # formerly to_str()
    def as_str(self) -> str:
        return f"{SEP}".join(self.to_tuple())

    @classmethod
    def from_str(Cls, rep: str):
        return Cls.from_col(rep.strip().split(SEP))

    def to_tuple(self) -> list[str]:
        def _conv(item):
            if type(item) is str:
                return item.strip()

            if type(item) is int:
                return str(item)

            if type(item) is tuple:
                return " ".join(map(str, item))

            if item is None:
                return ""

            assert False, f"type conversion failed for {item}"

        return tuple(map(_conv, astuple(self)))

    @classmethod
    def from_col(Cls, col: Iterable[str]):
        lis = tuple(col)

        def _conv(field, item, annotation):
            if annotation is str:
                return item

            if annotation is int:
                return int(item)

            # instance/type checks don't work for GenericAlias
            if repr(annotation) == "tuple[int]":
                return tuple(map(int, item.split()))

            # for Optiona[X] types
            # TODO instead determine default kwarg from field?
            if repr(annotation).startswith("typing.Optional"):
                if item == "":
                    return None

                t1, t2 = typing.get_args(annotation)
                assert t2 is type(None)  # noqa: E721 (isinstance does not work)
                return _conv(field, item, t1)

            assert False, f"type conversion failed for {field}: {item}"

        # ticket, nid, ...
        fieldlis = fields(Cls)
        assert len(fieldlis) == len(lis), f"{len(fieldlis)=} == {len(lis)=}"

        kwargs = {
            field.name: _conv(field.name, item, field.type)
            for field, item in zip(fieldlis, lis)
        }

        return Cls(**kwargs)


@dataclass(eq=True)
class Nomatch(_Base):
    """
    Sample of candidate context
    """

    # order determines csv layout (see _Base)
    identifier: int = field(compare=False)
    context: str
    eid: Optional[int] = field(default=None)

    def __hash__(self):
        return hash(astuple(self))

    def __lt__(self, other):
        return (self.context, self.identifier) < (other.context, other.identifier)

    @property
    def dic(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "eid": self.eid,
            "context": self.context,
        }

    @staticmethod
    def load_all(path: Union[str, Path], n: int = None) -> list["Nomatch"]:
        path = kpath(path, is_file=True)
        with path.open(mode="r") as fd:
            # TODO return generator; search callers
            return list(map(Nomatch.from_str, islice(fd, n)))


@dataclass(eq=True)
class Match(_Base):  # v7
    """

    Sample of a text context matched by homag/sampling.py

    The Match.context can be split by whitespace to obtain the
    tokens. These tokens are referenced by Match.mention_idxs in
    successive, increasing token spans. There can be no overlaps.
    Each span behaves as python ranges do: lower bound inclusive,
    upper bound exclusive:

    mention: "foo bar"
    context: ["This", "is", "some", "foo", "and", "bar"]
    -> mention_idxs: [3,4,5,6]

    """

    # order determines csv layout: see _Base
    identifier: int
    eid: int
    mention: str
    mention_idxs: tuple[int]
    context: str

    # see draug.homag.data.TextSample for an explanation why
    # frozen=True cannot be used and a manual hash implemention
    # is required
    def __hash__(self):
        return hash(astuple(self))

    @property
    def tokens(self):
        return self.context.split()

    def __str__(self):
        return f"Match [{self.eid=}] {self.mention}: >{self.context}<"

    def __lt__(self, other):
        return (self.eid, self.context) < (other.eid, other.context)


@dataclass(eq=True)
class Filtered(Match):
    """
    Filtered match
    """

    phrase: str


class Matches:  # v7

    graph: Graph

    @property
    def nids(self) -> set[int]:
        return set(self.graph.get_entity(eid=eid).nid for eid in self.eids)

    @property
    def eids(self) -> set[int]:
        return set(self._eid2match.keys())

    def by_nid(self, nid: int) -> set[Match]:
        agg = set()
        for eid in self._nid2eid:
            agg |= self.by_eid(eid)

        return agg

    def by_eid(self, eid: int) -> set[Match]:
        return self._eid2match[eid].copy()

    # no longer a property
    # -> makes it obvious that there's some cost involved here
    def flat(self) -> list[Match]:
        return [match for col in self._eid2match.values() for match in col]

    # --

    def add_match(self, match: Match):
        nid = self.graph.get_entity(eid=match.eid).nid
        self._nid2eid[nid].add(match.eid)
        self._eid2match[match.eid].add(match)

    # --

    def __str__(self) -> str:
        return (
            f"draug.matches: {len(self.flat())} matches "
            f"(nodes={len(self._nid2eid)}, "
            f"entities={len(self._eid2match)})"
        )

    def __init__(self, graph: Graph):
        self.graph = graph
        self._nid2eid = defaultdict(set)
        self._eid2match = defaultdict(set)

    @classmethod
    def from_file(
        Cls,
        path: Union[str, Path],
        graph: Graph,
        n: Optional[int] = None,
    ):
        self = Cls(graph=graph)
        # e.g. homag/matches/symptax.v6.match.txt
        path = kpath(path, is_file=True)

        with path.open(mode="r", **open_kwargs) as fd:
            gen = filter(None, map(str.strip, islice(fd, n)))
            for match in map(Match.from_str, gen):
                self.add_match(match)

        return self


# ---


def _split_aggregated(
    agg: dict[str, str],
    ratio: float,
    seed: int,
) -> tuple[dict]:
    assert 0 < ratio < 1

    q = sorted(agg.items())

    random.seed(seed)
    random.shuffle(q)

    def weight(t):
        # sentence count
        return len(t[1].split("\n"))

    t1, t2 = q.pop(), q.pop()
    parts = [
        {"tickets": [t1], "weight": weight(t1)},
        {"tickets": [t2], "weight": weight(t2)},
    ]

    while len(q):
        t = q.pop()

        a, b = parts[0]["weight"], parts[1]["weight"]
        i = 0 if (a / (a + b)) < ratio else 1

        parts[i]["tickets"].append(t)
        parts[i]["weight"] += weight(t)

    return parts


def _process_doc(
    doc: dumpr.Document,
    fd_filtered,
    counts: Counter,
    min_len: int,
    max_len: int,
) -> Optional[str]:
    # if ticket_id != '1599904':
    #    continue

    if not all((doc.content, doc.meta.get("lang_code", None) == "de")):
        return

    content = doc.content.strip() + "\n"

    # remove csv separator
    content = content.replace(SEP, "")
    # normalize/add full stop of/to every sentence
    # content = re.sub(r'( \.)?[^a-zA-Z0-9!?]*\n$', ' .\n', content)
    # ( foo , bar ) something . -> (foo, bar) something.
    # content = re.sub(r" ([.,!?)])", r"\1", content).replace("( ", "(")
    # foo - bar and foo----bar -> foo bar
    content = re.sub(r"[-_*]+", " ", content)
    # 09. 05. 03 -> 09.05.03
    # content = re.sub(r'\. ([0-9])', r'.\1', content)
    # foo    bar -> foo bar

    # remove umlauts (because the mention list looks like that)
    # also remove some special characters who confuse the transformers tokenizer
    for tar, sub in (
        ("ä", "ae"),
        ("ü", "ue"),
        ("ö", "oe"),
        ("ß", "ss"),
        ("`", ""),
        ("´", ""),
    ):
        content = content.replace(tar, sub)

    sentences = []
    for sentence in content.split("\n"):

        # content = re.sub(r' +', ' ', content)
        sentence = " ".join(sentence.strip().split())
        if not sentence:
            continue

        # foo bar -> Foo bar (per sentence)
        sentence = sentence[0].upper() + sentence[1:]

        if min_len < len(sentence) < max_len:
            sentences.append(sentence)

        else:
            ticket_id = doc.meta["ticket"]
            fd_filtered.write(f"{ticket_id}|{sentence}\n")
            counts["filtered"] += 1

    return "\n".join(sentences)


def split(
    out_dir: Union[str, Path],
    tickets: Union[str, Path],
    seed: int,
    ratio: float,
    n: Optional[int] = None,
    min_len: int = None,
    max_len: int = None,
):

    min_len = 20 if min_len is None else min_len
    max_len = 300 if max_len is None else max_len

    out_dir = kpath(out_dir, create=True)

    filtered = out_dir / "filtered.txt"
    counts = Counter()

    with dumpr.BatchReader(str(tickets)) as reader, filtered.open(mode="w") as fd:

        agg = defaultdict(str)
        for doc in tqdm(islice(reader, n), total=n or reader.count):

            ticket_id = doc.meta["ticket"]
            content = _process_doc(
                doc=doc,
                fd_filtered=fd,
                counts=counts,
                min_len=min_len,
                max_len=max_len,
            )

            if not content:
                counts["empty"] += 1
                continue

            agg[ticket_id] += content + "\n"

            # later tokenization relies on sane whitespace
            assert "  " not in agg[ticket_id] and "\n\n" not in agg[ticket_id]

    log.info(f'{counts["filtered"]} sentences filtered due to length constraints')
    log.info(f'{counts["empty"]} empty qdocuments')

    # filter empty

    assert all(len(content.strip()) for content in agg.values())
    log.info(f"selected {len(agg)} tickets")

    train, test = _split_aggregated(agg=agg, ratio=ratio, seed=seed)
    for name, part in (("train", train), ("test", test)):
        target = out_dir / f"{name}.xml"

        with dumpr.BatchWriter(target) as writer:
            for ticket_id, content in tqdm(part["tickets"]):
                doc = dumpr.Document(content="\n" + content.strip() + "\n") + dict(
                    identifier=ticket_id
                )
                writer.write(doc)

    with (out_dir / "config.yml").open(mode="w") as fd:
        yaml.dump(
            dict(
                created=datetime.now(),
                seed=seed,
                ratio=ratio,
                min_len=min_len,
                max_len=max_len,
                tickets=str(tickets),
            ),
            fd,
        )
