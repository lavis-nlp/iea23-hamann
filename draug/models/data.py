# -*- coding: utf-8 -*-

import draug
from draug.homag.text import SEP
from draug.homag.text import Match
from draug.homag.text import Matches
from draug.homag.text import Nomatch
from draug.homag.graph import Graph

from ktz.string import decode_line
from ktz.string import encode_line
from ktz.filesystem import path as kpath

import abc
import yaml
import torch

import transformers as tf
import torch.utils.data as td
import pytorch_lightning as pl
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm as _tqdm

import enum
import random
import logging
import textwrap
import contextlib
from pathlib import Path

from dataclasses import field
from dataclasses import replace
from dataclasses import dataclass
from itertools import islice
from functools import partial
from collections import deque
from collections import defaultdict
from collections import Counter

from typing import Any
from typing import Union
from typing import Deque
from typing import Optional


log = logging.getLogger(__name__)
tqdm = partial(_tqdm, ncols=80)


MIN_SAMPLES = 10
MAX_SEQ_LEN = 300


def _init_tokenizer(
    path: Path,
    model: Optional[str] = None,
    from_disk: bool = False,
):

    tok_dir = path / "tokenizer"

    if from_disk or tok_dir.is_dir():
        assert tok_dir.is_dir(), f"cannot load tokenizer from {tok_dir}"
        log.info("loading tokenizer from disk")
        return tf.BertTokenizerFast.from_pretrained(str(tok_dir))

    assert model is not None, "no tokenizer found, provide a model"

    log.info("! creating new tokenizer")
    cache_dir = str(draug.ENV.DIR.CACHE / "lib.transformers")
    tokenizer = tf.BertTokenizerFast.from_pretrained(
        model,
        cache_dir=cache_dir,
        additional_special_tokens=[
            Token.MENTION_START.value,
            Token.MENTION_END.value,
        ],
    )

    log.info("saving tokenizer to disk")
    tokenizer.save_pretrained(str(tok_dir))
    return tokenizer


class Token(enum.Enum):

    MASK = "[MASK]"
    MENTION_START = "[MENTION_START]"
    MENTION_END = "[MENTION_END]"


class Split(enum.Enum):

    ALL = "all"
    TRAIN = "training"
    VALID = "validation"
    TEST = "testing"


class Mode(enum.Enum):

    NOMASK = "not masked"
    FULLMASK = "fully masked"
    PROBMASK = "sometimes masked"

    QUERY_NOMASK = "query not masked"
    QUERY_PROBMASK = "query sometimes masked"


def _pad_idxs(idxs_lis: list[tuple[int]]) -> torch.Tensor:
    return pad_sequence(
        [torch.Tensor(idxs).to(torch.long) for idxs in idxs_lis],
        batch_first=True,
    )


@dataclass
class QuerySample:

    nomatch: Nomatch
    tok_context: tuple[int]

    # not possible to use frozen=True: see TextSample.__hash__
    def __hash__(self):
        return hash(self.nomatch)

    def __lt__(self, other):
        return self.nomatch < other.nomatch

    @property
    def to_bytes(self) -> bytes:
        toks = " ".join(map(str, self.tok_context))
        cols = self.nomatch.to_tuple() + (toks,)
        return encode_line(data=cols, sep=SEP)

    @classmethod
    def from_bytes(K: "QuerySample", encoded: bytes) -> "QuerySample":
        cols = decode_line(encoded, sep=SEP)
        nomatch_args, sample_args = cols[:3], cols[3:]

        def _tup(s: str) -> list[int]:
            return tuple(map(int, s.split()))

        nomatch = Nomatch.from_col(nomatch_args)
        tok_context = _tup(sample_args[0])
        return K(nomatch=nomatch, tok_context=tok_context)


@dataclass
class TextSample:

    match: Match

    # tokenized mention and its positions in tok_context
    tok_mention: tuple[int]
    idx_mention: tuple[int]
    tok_context: tuple[int]

    # with the newer pythorch lightning versions it is no longer
    # possible to use frozen=True dataclasses as they aggressively
    # setattr() every property for dataclasses:
    # (see pytorch_lightning.utilities.apply_func.apply_to_collection())
    # and as such, we need to implement our own __hash__
    # without the guarantee of immnutability :<
    def __hash__(self):
        # this is but a puny wrapper around Match objects
        return hash(self.match)

    def __str__(self) -> str:
        short = textwrap.shorten(self.context, width=40, placeholder="...")
        return f"Text [{self.match.nid}] {self.match.name}: {short}"

    @property
    def description(self) -> str:
        return (
            f"TextSample:\n"
            f"  {self.match.name} ({self.match.nid})\n"
            f"  Mention: {self.match.mention}\n"
            f"  Context: {self.match.context}\n"
        )

    def __lt__(self, other: "TextSample"):
        return self.match < other.match

    def mask(self, tokenizer) -> "TextSample":
        # TOOD this is slow: figure out why
        tok_context = list(self.tok_context)

        for a, b in zip(self.idx_mention[::2], self.idx_mention[1::2]):
            for i in range(a, b):
                tok_context[i] = tokenizer.vocab[tokenizer.mask_token]

        return replace(self, tok_context=tuple(tok_context))

    @property
    def to_query(self) -> QuerySample:
        return QuerySample(
            nomatch=Nomatch(
                ticket=self.match.ticket,
                context=self.match.context,
                eid=self.match.eid,
            ),
            tok_context=self.tok_context,
        )

    @property
    def to_bytes(self) -> bytes:
        assert SEP not in self.match.mention
        assert SEP not in self.match.context

        def _str(lis: list[int]) -> str:
            return " ".join(map(str, lis))

        cols = self.match.to_tuple() + (
            _str(self.tok_mention),
            _str(self.idx_mention),
            _str(self.tok_context),
        )

        return encode_line(data=cols, sep=SEP)

    @classmethod
    def from_bytes(K: "TextSample", encoded: bytes) -> "TextSample":
        cols = decode_line(encoded, sep=SEP)
        match_args, sample_args = cols[:5], cols[5:]
        match = Match.from_col(col=match_args)

        def _tup(s: str) -> list[int]:
            return tuple(map(int, s.split()))

        tok_mention, idx_mention, tok_context = map(_tup, sample_args)

        return K(
            match=match,
            tok_mention=tok_mention,
            idx_mention=idx_mention,
            tok_context=tok_context,
        )


class Dataset(td.Dataset, abc.ABC):
    """

    Base class maintaining text datasets produced by homag.sampling

    """

    mode: Mode
    split: Split
    path: Path

    modes = None

    @property
    def num_classes(self) -> int:
        return len(self._q)

    @property
    def class_counts(self) -> dict[int, int]:  # len -> #classes
        return {k: len(v) for k, v in self._q.items()}

    @abc.abstractmethod
    def init_data(self):
        # must populate self._q (mapping labels to samples)
        raise NotImplementedError()

    def __init__(
        self,
        path: Union[str, Path],
        mode: Mode,
        split: Split = Split.ALL,
        # optional, mode specific options:
        mask_prob: float = None,
        # optional overrides
        tokenizer: Optional[Path] = None,
    ):

        assert self.modes, "add .modes to dataset"
        log.info(f"initializing >[{split.value}]< samples {mode.value}")

        super().__init__()

        self.path = kpath(path, exists=True)
        self.mode = mode
        self.split = split

        # --

        self.tokenizer = _init_tokenizer(
            path=self.path if tokenizer is None else tokenizer,
            from_disk=True,
        )

        assert Token.MASK.value == self.tokenizer.mask_token
        assert Token.MENTION_START.value in self.tokenizer.additional_special_tokens
        assert Token.MENTION_END.value in self.tokenizer.additional_special_tokens

        # --

        self._q = defaultdict(deque)
        self.init_data()
        assert len(self._q)

        # --

        # [query] sometimes masked
        if self.mode is Mode.QUERY_PROBMASK or self.mode is Mode.PROBMASK:
            assert 0 < mask_prob < 1
            self.mask_prob = mask_prob
            log.info(f"setting masking probability to {self.mask_prob}")

        # find maximum contexts

        gen = enumerate(sample for q in self._q.values() for sample in q)
        gen = ((len(sample.tok_context), i) for i, sample in gen)

        counts = sorted(gen, reverse=True)
        count, index = counts[0]

        log.info(f"{self.mode.value}/{self.split.value}: max length {count} at {index}")

    # interface

    @property
    def sample_weights(self) -> list[int]:  # len -> #samples
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()


class QueryDataset:

    flat: list[QuerySample]

    def __len__(self):
        return len(self.flat)

    def __init__(self, path: Path, max_samples: int = None):
        log.info(f"loading query dataset {max_samples=}")

        with path.open(mode="rb") as fd:
            lines = islice(fd, max_samples)
            self.flat = tuple(map(QuerySample.from_bytes, lines))

        log.info(f"loaded {len(self.flat)} query samples")

    def __getitem__(self, i: int) -> QuerySample:
        return self.flat[i]

    @staticmethod
    def collate_fn(
        batch: list[QuerySample],
    ) -> tuple[tuple[QuerySample], torch.Tensor]:
        tups = ((qs, qs.tok_context) for qs in batch)
        query_samples, idxs = zip(*tups)
        return query_samples, _pad_idxs(idxs)


@dataclass
class GraphSample:

    # TextSample for positives
    # QuerySample for negatives
    h: Union[TextSample, QuerySample]

    t_idx: int
    t_str: str

    r_idx: int
    r_str: str

    # for negative sampling
    positive: bool

    def __str__(self):

        if isinstance(self.h, TextSample):
            name = f"matched: '{self.h.match.mention}' ({self.h.match.eid})"

        if isinstance(self.h, QuerySample):
            name = f"query: '{self.h.nomatch.context[:20]}'"

        return (
            f"GraphSample: {name}"
            f" {self.r_str} ({self.r_idx})"
            f" {self.t_str} ({self.t_idx})"
        )


@dataclass
class GraphCollation:

    # padded input for bert
    h_tokens: torch.LongTensor = field(default_factory=list)

    # indexes for the embedding matrix
    t_idxs: torch.LongTensor = field(default_factory=list)
    r_idxs: torch.LongTensor = field(default_factory=list)


class GraphDataset(Dataset, abc.ABC):
    """

    Graph dataset for kgc-like training

    Based on the text datasets (for efficient loading) but sampling is
    based on triples. Each sample is a triple where the heads and
    tails are associated text contexts. These text contexts are
    selected randomly.

    Each samples' weight is the reziprocal of the sum of assigned texts.
    This allows for weighted random sampling.

    Nodes for which no text contexts exist are assigned their
    phrase/name. Those are never masked.

    """

    graph: Graph

    # triples have nids (not idxs)
    triples: tuple[tuple[int, int, str]]

    idx2nid: dict[int, int]
    nid2idx: dict[int, int]
    idx2str: dict[int, str]

    modes = {
        Mode.NOMASK,
        Mode.FULLMASK,
        Mode.PROBMASK,
    }

    def should_mask(self):
        return any(
            (
                self.mode is Mode.FULLMASK,
                self.mode is Mode.PROBMASK and random.random() < self.mask_prob,
            )
        )

    # ---

    def _create_sample(
        self,
        h_sample: Union[TextSample, QuerySample],
        t: int,
        r: int,
        positive: bool = True,
    ) -> GraphSample:
        assert isinstance(h_sample, TextSample if positive else QuerySample)

        # translates nids to idxs
        return GraphSample(
            h=h_sample,
            #
            t_str=self.graph.node_name(nid=t),
            t_idx=self.nid2idx[t],
            #
            r_str=self.graph.meta["relmap"][r],
            r_idx=r,
            positive=positive,
        )

    def __init__(self, graph: Graph, *args, **kwargs):
        self.graph = graph

        super().__init__(*args, **kwargs)

        self.idx2nid = {idx: nid for idx, nid in enumerate(self.graph.nids)}
        self.nid2idx = {nid: idx for idx, nid in self.idx2nid.items()}

        self.idx2str = {
            idx: f"{idx}:{nid}:{self.graph.node_name(nid)}"
            for nid, idx in self.nid2idx.items()
        }

    # abc implementations

    def __len__(self):
        return len(self.triples)

    def init_data(self):
        log.info(f"initializing samples for {self.mode}")

        classes = set()

        in_file = self.path / f"matches.{Split.ALL.value}-{self.split.value}"
        with in_file.open(mode="rb") as fd:
            for sample in map(TextSample.from_bytes, fd):
                nid = self.graph.get_entity(eid=sample.match.eid).nid
                self._q[nid].append(sample)
                classes.add(nid)

        total = sum(self.class_counts.values())
        log.info(f"read {total} samples distributed over {self.num_classes} clusters")

    @staticmethod
    def collate_fn(
        samples: list[GraphSample],
    ) -> tuple[list[GraphSample], GraphCollation]:

        collation = GraphCollation()

        # GraphSamples use idx (not nid)
        for sample in samples:
            collation.h_tokens.append(sample.h.tok_context)
            collation.r_idxs.append(sample.r_idx)
            collation.t_idxs.append(sample.t_idx)

        return samples, replace(
            collation,
            h_tokens=_pad_idxs(collation.h_tokens),
            t_idxs=torch.tensor(collation.t_idxs, dtype=torch.long),
            r_idxs=torch.tensor(collation.r_idxs, dtype=torch.long),
        )

        # try with new pytorch version; use this if conversion does
        # not work:

        # need asdict to work with automatic conversions
        # return samples, asdict(
        #     replace(
        #         collation,
        #         h_tokens=_pad_idxs(collation.h_tokens),
        #         t_idxs=torch.tensor(collation.t_idxs, dtype=torch.long),
        #         r_idxs=torch.tensor(collation.r_idxs, dtype=torch.long),
        #     )
        # )


class GraphDatasetExhaustive(GraphDataset):
    """

    Pre-compute all possible GraphSamples based on all
    possible textual combinations. Used for evaluation.

    """

    @property
    def sample_weights(self) -> list[int]:  # len -> #samples
        raise NotImplementedError("Do not use weighted sampling!")

    def __init__(self, max_per_edge: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.triples = []
        for h, t, r in sorted(self.graph.nxg.edges):
            for h_sample in islice(self._q[h], max_per_edge):

                if self.should_mask():
                    h_sample = h_sample.mask(tokenizer=self.tokenizer)

                sample = self._create_sample(h_sample=h_sample, t=t, r=r)
                self.triples.append(sample)

        log.info(f"initialized exhaustive dataset: {len(self)} samples")

    def __getitem__(self, i: int) -> GraphSample:
        return self.triples[i]


class GraphDatasetSelective(GraphDataset):
    """

    Construct GraphSamples dynamically. Used for training.

    """

    def _init_triples(self):
        triples, edges = set(), self.graph.nxg.edges

        for h, t, r in edges:
            if len(self._q[h]):
                triples.add((h, t, r))

        self.triples = tuple(triples)
        log.info(f"using {len(triples)}/{len(edges)} triples for training")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_triples()

        log.info("! shuffling text buffers")
        for q in self._q.values():
            random.shuffle(q)

        log.info(f"initialized selective graph dataset: {len(self.triples)} triples")

    def __getitem__(self, i: int) -> GraphSample:
        h, t, r = self.triples[i]

        h_sample = self._q[h][0]
        self._q[h].rotate()

        if self.should_mask():
            h_sample = h_sample.mask(tokenizer=self.tokenizer)

        return self._create_sample(h_sample=h_sample, t=t, r=r)

    # abc implementation

    @property
    def sample_weights(self) -> list[int]:  # len -> #samples
        weights = []

        for h, t, _ in self.triples:
            # text dataset samples have nids assigned
            weight = 1 / self.class_counts[h]
            weights.append(weight)

        return weights


class GraphDatasetNegativeHeads(GraphDatasetSelective):
    """

    Sample n negative (h, r', t') for a positive (h, t, r)
    where (h, r', t') \notin T.

    Collation:
     positive(1)
     negative_1(1)
     negative_2(1)
     ...
     positive(2)
     negative_1(2)
     negative_2(2)
     ...

    """

    num_negatives: int
    query_dataset: QueryDataset
    negatives: Deque[QuerySample]

    def __init__(
        self,
        num_negatives: int,
        *args,
        max_negatives: int = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._nomatches_path = kpath(self.path / "nomatches.all", is_file=True)

        log.info(f"loading negative samples from {self._nomatches_path}")
        self.query_dataset = QueryDataset(
            path=self._nomatches_path,
            max_samples=max_negatives,
        )

        self.num_negatives = num_negatives

        negatives = sorted(self.query_dataset.flat)
        random.shuffle(negatives)
        self.negatives = deque(negatives)

        log.info(f"got {len(self.negatives)} negative sample candidates")

    def __getitem__(self, i: int) -> tuple[GraphSample]:  # len: positive + n * negative
        positive = super().__getitem__(i)

        neg_query_samples = []
        for _ in range(self.num_negatives):
            neg_query_samples.append(self.negatives[0])
            self.negatives.rotate(1)

        neg_samples = tuple(
            self._create_sample(
                h_sample=qs,
                t=self.idx2nid[positive.t_idx],  # back and forth ;)
                r=positive.r_idx,
                positive=False,
            )
            for qs in neg_query_samples
        )

        sample = (positive,) + neg_samples

        targets = torch.zeros((len(sample), len(self.idx2nid)))
        targets[0, positive.t_idx] = 1

        return sample, targets

    @staticmethod
    def collate_fn(samples: list[tuple[tuple[GraphSample, torch.Tensor]]]):
        # batchsize * (negatives + 1) many samples

        collation = GraphCollation()

        # GraphSamples use idx (not nid)
        for sample in [s for sub, _ in samples for s in sub]:
            collation.h_tokens.append(sample.h.tok_context)
            collation.r_idxs.append(sample.r_idx)
            collation.t_idxs.append(sample.t_idx)

        # padding
        collation = replace(
            collation,
            h_tokens=_pad_idxs(collation.h_tokens),
            t_idxs=torch.tensor(collation.t_idxs, dtype=torch.long),
            r_idxs=torch.tensor(collation.r_idxs, dtype=torch.long),
        )

        targets = torch.vstack([target for _, target in samples])

        return samples, collation, targets


DATASETS = {
    "graph selective": GraphDatasetSelective,
    "graph negative heads": GraphDatasetNegativeHeads,
    "graph exhaustive": GraphDatasetExhaustive,
    "query": QueryDataset,
    # archived:
    # "simple": SimpleDataset,
    # "negative sampling": NegativeSamplingDataset,
}


SAMPLER = {
    "random": td.RandomSampler,
    "weighted random": td.WeightedRandomSampler,
}


class DataModule(pl.LightningDataModule):

    path: Path
    meta: dict

    sampler_cls: str
    sampler_kwargs: dict[str, Any]

    train_ds_cls: td.Dataset
    train_ds_kwargs: dict
    train_loader_kwargs: dict

    valid_ds_cls: td.Dataset
    valid_ds_kwargs: dict
    valid_loader_kwargs: dict

    def _load_ds(self, K, split: Split, **kwargs):
        kwargs["mode"] = Mode(kwargs["mode"])
        return K(split=split, path=self.path, graph=self.graph, **kwargs)

    def _load_meta(self):
        meta_file = self.path / "meta.yml"
        if meta_file.is_file():
            log.info("loading meta file")
            with meta_file.open(mode="r") as fd:
                self.meta = yaml.load(fd, Loader=yaml.FullLoader)
                return

        log.info("generating meta information")

        ds = self._load_ds(
            self.train_ds_cls,
            split=Split.TRAIN,
            **self.train_ds_kwargs,
        )

        self.meta = dict(
            num_classes=ds.num_classes,
            nid2idx=ds.nid2idx,
            idx2str=ds.idx2str,
            class_counts=ds.class_counts,
            relations=self.graph.meta["relmap"],
        )

        with meta_file.open(mode="w") as fd:
            log.info("writing meta information to file")
            yaml.dump(self.meta, fd)

    def __init__(
        self,
        path: Path,
        graph: Graph,
        sampler: str,
        sampler_kwargs: dict[str, Any],
        train_ds: str,
        train_ds_kwargs: dict[str, Any],
        train_loader_kwargs: dict[str, Any],
        valid_ds: str,
        valid_ds_kwargs: dict[str, Any],
        valid_loader_kwargs: dict[str, Any],
    ):
        super().__init__(self)

        self.path = path
        self.graph = graph

        self.sampler_cls = sampler
        self.sampler_kwargs = sampler_kwargs

        self.train_ds_cls = DATASETS[train_ds]
        self.train_ds_kwargs = train_ds_kwargs
        self.train_loader_kwargs = train_loader_kwargs

        self.valid_ds_cls = DATASETS[valid_ds]
        self.valid_ds_kwargs = valid_ds_kwargs
        self.valid_loader_kwargs = valid_loader_kwargs

        self._load_meta()

    def _init_sampler(self):
        log.info(f"initializing {self.sampler_cls} sampler with {self.sampler_kwargs}")
        sampler_kwargs = self.sampler_kwargs.copy()

        if self.sampler_cls == "random":
            sampler_kwargs |= dict(data_source=self.train_ds)

        if self.sampler_cls == "weighted random":
            sampler_kwargs |= dict(weights=torch.tensor(self.train_ds.sample_weights))

        self.sampler = SAMPLER[self.sampler_cls](**sampler_kwargs)

    def setup(self, stage: Optional[str] = None):
        log.info("! setup datasets")

        self.train_ds = self._load_ds(
            self.train_ds_cls,
            split=Split.TRAIN,
            **self.train_ds_kwargs,
        )
        self.valid_ds = self._load_ds(
            self.valid_ds_cls,
            split=Split.VALID,
            **self.valid_ds_kwargs,
        )

        log.error("TODO disabled test dataset!")
        # self.test_ds = self._load_ds(
        #     self.valid_ds_cls,
        #     split=Split.TEST,
        #     **self.valid_ds_kwargs,
        # )

        self._init_sampler()

    def train_dataloader(self):
        ds = self.train_ds
        return td.DataLoader(
            ds,
            collate_fn=ds.collate_fn,
            **self.train_loader_kwargs,
            sampler=self.sampler,
        )

    def val_dataloader(self):
        ds = self.valid_ds
        return td.DataLoader(
            ds,
            collate_fn=ds.collate_fn,
            **self.valid_loader_kwargs,
        )

    def test_dataloader(self):
        ds = self.test_ds
        return td.DataLoader(
            ds,
            collate_fn=ds.collate_fn,
            **self.valid_loader_kwargs,
        )

    @classmethod
    def create(K, graph: Graph, path: Union[str, Path], config: dict, **ds_kwargs):
        path = kpath(path)

        train_ds_kwargs = config["train_dataset_kwargs"] | ds_kwargs
        valid_ds_kwargs = config["valid_dataset_kwargs"] | ds_kwargs

        return K(
            path=path,
            graph=graph,
            sampler=config["sampler"],
            sampler_kwargs=config["sampler_kwargs"],
            train_ds=config["train_dataset"],
            train_ds_kwargs=train_ds_kwargs,
            train_loader_kwargs=config["train_loader"],
            valid_ds=config["valid_dataset"],
            valid_ds_kwargs=valid_ds_kwargs,
            valid_loader_kwargs=config["valid_loader"],
        )


#
# ----------------------------------------
#
# MODELS


def _write_queries(
    flat: list[Nomatch],
    out_file: Path,
    out_filtered: Path,
    tokenizer,
):
    def _open(path):
        return path.open(mode="wb")

    with _open(out_file) as fd_out, _open(out_filtered) as fd_filtered:
        for nomatch in tqdm(flat):
            toks = tuple(tokenizer(nomatch.context)["input_ids"])

            if MAX_SEQ_LEN <= len(toks):
                fd_filtered.write(nomatch.as_str)
                continue

            sample = QuerySample(nomatch=nomatch, tok_context=toks)
            fd_out.write(sample.to_bytes)


def create_queries(
    source: Path,
    out: Path,
    model: str,
):

    out = out / model / "dataset"

    print("reading source file")
    flat = []
    with source.open(mode="r") as fd_in:
        for line in tqdm(fd_in):
            identifier, context, *_ = map(str.strip, line.split(SEP))
            flat.append(Nomatch(ticket=identifier, context=context))

    print("tokenizing text")
    out_file = out / f"nomatches.{Split.ALL.value}"
    out_filtered = out / f"nomatches.{Split.ALL.value}-filtered"
    tokenizer = _init_tokenizer(path=out, model=model)

    _write_queries(
        flat=flat,
        out_file=out_file,
        out_filtered=out_filtered,
        tokenizer=tokenizer,
    )


def create_dataset(
    graph: Graph,
    source: Path,
    out: Path,
    model: str,
):

    out = out / model / "dataset"
    tokenizer = _init_tokenizer(path=out, model=model)

    # ---

    print("reading source file")
    matches = Matches.from_file(path=source, graph=graph)

    # ---

    print("tokenizing text")
    out_file = out / f"matches.{Split.ALL.value}"
    out_filtered = out / f"matches.{Split.ALL.value}-filtered"

    def _open(path):
        return path.open(mode="wb")

    with _open(out_file) as fd_out, _open(out_filtered) as fd_filtered:
        for match in tqdm(matches.flat()):

            be_mention = tokenizer(match.mention)
            tok_mention = tuple(be_mention["input_ids"][1:-1])

            be_context = tokenizer(match.context)
            tok_context = tuple(be_context["input_ids"])

            constraints = (
                len(matches.by_eid(match.eid)) < MIN_SAMPLES,
                MAX_SEQ_LEN <= len(tok_context),
            )

            if any(constraints):
                fd_filtered.write(match.as_str.encode() + b"\n")
                continue

            # Create mapping between match.mention_idxs and TextSample.idx_mention.
            # be_context.char_to_token offers a mapping of input sequence character
            # to output token index
            idx_mention = []
            for a, b in zip(match.mention_idxs[::2], match.mention_idxs[1::2]):
                x = sum(len(token) + 1 for token in match.tokens[:a])
                y = sum(len(token) + 1 for token in match.tokens[:b]) - 1

                tokens = [be_context.char_to_token(c) for c in range(x, y)]
                tokens = sorted(set(filter(None, tokens)))

                # upper bound exclusive
                idx_mention += [tokens[0], tokens[-1] + 1]

            assert idx_mention
            sample = TextSample(
                match=match,
                tok_mention=tok_mention,
                idx_mention=idx_mention,
                tok_context=tok_context,
            )

            # sample.mask(tokenizer)  # just testing
            fd_out.write(sample.to_bytes)


# create training/validation/testing split
def split_dataset(
    path: Path,
    seed: int,
    split: list[int],
    model: str,
):
    assert sum(split) == 1 and len(split) in {2, 3}
    random.seed(seed)

    data = defaultdict(list)

    with path.open(mode="rb") as fd:
        for sample in map(TextSample.from_bytes, fd):
            data[sample.match.eid].append(sample)

    with (path.parent / "config.split.yml").open(mode="w") as fd:
        yaml.dump(dict(seed=seed, split=list(split)), fd)

    counts = Counter()

    # proper multiline with-statements come with python 3.10
    with contextlib.ExitStack() as stack:
        prefix = f"{path.name}-"
        paths = (
            path.parent / f"{prefix}{Split.TRAIN.value}",
            path.parent / f"{prefix}{Split.VALID.value}",
        )

        if len(split) == 3:
            paths += (path.parent / f"{prefix}{Split.TEST.value}",)

        fds = [stack.enter_context(p.open(mode="wb")) for p in paths]

        for samples in tqdm(data.values()):
            assert MIN_SAMPLES <= len(samples)

            samples = sorted(set(samples))
            random.shuffle(samples)

            # set aside minimum amount of samples per split
            l_train, l_valid, l_test = samples[:2], samples[2:4], []
            if len(split) == 2:
                remaining = samples[4:]
            if len(split) == 3:
                l_test, remaining = samples[4:6], samples[6:]

            # distribute remaining samples
            n = len(remaining)
            t1, t2 = int(n * split[0]), int(n * (split[0] + split[1])) + 1
            l_train += remaining[:t1]
            l_valid += remaining[t1:t2]

            if len(split) == 3:
                l_test += remaining[t2:]

            assert sum(len(lis) for lis in (l_train, l_valid, l_test)) == len(samples)

            fds[0].writelines([sample.to_bytes for sample in l_train])
            fds[1].writelines([sample.to_bytes for sample in l_valid])
            if len(split) == 3:
                fds[2].writelines([sample.to_bytes for sample in l_test])

            counts["train"] += len(l_train)
            counts["valid"] += len(l_valid)
            counts["test"] += len(l_test)

    print(
        textwrap.dedent(
            f"""
            split samples:
            - train: {counts['train']}
            - valid: {counts['valid']}
            - test: {counts['test']}
            """
        )
    )


def create_crossval():

    pass
