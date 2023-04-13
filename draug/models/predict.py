# -*- codingg: utf-8 -*-

import draug
from draug.homag import text
from draug.models import data
from draug.models import models
from draug.homag import crossval
from draug.homag.text import SEP
from draug.homag.graph import Graph
from draug.homag.graph import NID, EID, RID

from ktz import string as kstring
from ktz.collections import buckets
from ktz.filesystem import path as kpath


import yaml
from torch.utils import data as td
import torchmetrics.functional as tmf
from tqdm import tqdm as _tqdm

import csv
import gzip
import logging
from pathlib import Path
import statistics as stats
import multiprocessing as mp
from itertools import islice
from functools import partial
from collections import deque
from collections import Counter
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import astuple
from dataclasses import asdict

from typing import Any
from typing import Union
from typing import Optional
from typing import Iterable
from typing import Callable
from typing import Generator


tqdm = partial(_tqdm, ncols=80)
log = logging.getLogger(__name__)


def load_model_config(config_file: Path):
    assert config_file.is_file()

    log.info("loading configuration")
    with config_file.open(mode="r") as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)

    return config


def load_data(config: dict):
    graph_dir = kpath(config["graph_dir"], is_dir=True)
    dataset_dir = kpath(config["dataset_dir"], is_dir=True)

    graph = Graph.from_dir(path=graph_dir)

    log.info("loading datamodule")
    datamodule = data.DataModule.create(
        path=dataset_dir,
        graph=graph,
        config=config,
    )

    datamodule.setup()
    return datamodule


def load_model(config: dict, ckpt_file: Optional[Path] = None, **kwargs):
    Model = models.get_cls(name=config["model"])
    kwargs = (
        dict(bert=config["transformer"], config=config)
        | config["model_kwargs"]
        | kwargs
    )

    if ckpt_file is None:
        log.warning("no ckpt provided: loading uninitialized model")
        model = Model(**kwargs)
    else:
        log.info(f"loading model from checkpoint: {ckpt_file}")
        model = Model.load_from_checkpoint(checkpoint_path=str(ckpt_file), **kwargs)

    model = model.to(device="cuda")
    return model


# ------------------------------------------------------------
# VALIDATION
#


def _validate(model, datamodule, n_batches):

    dataloader = datamodule.val_dataloader()
    gen = islice(enumerate(dataloader), n_batches)
    total = n_batches if n_batches else len(dataloader)

    agg = defaultdict(set)
    for i, batch in tqdm(gen, total=total):
        batch = datamodule.transfer_batch_to_device(batch, model.device)
        graph_samples, kwargs = batch

        collation = data.GraphCollation(**kwargs)
        valids = model.validate(graph_samples=graph_samples, batch=collation)

        for kind, rset in valids.items():
            # kind \in {'head', 'tail'}
            agg[kind] |= rset

    return {k: sorted(v) for k, v in agg.items()}


def validate(
    config_file: Path,
    ckpt_file: Optional[Path] = None,
    n_batches: Optional[int] = None,
):
    # initialize data and models
    config = load_model_config(config_file)
    datamodule = load_data(config=config)

    model = load_model(
        ckpt_file=ckpt_file,
        config=config,
        datamodule=datamodule,
    )

    # predict labels

    log.info("running predictor on validation data")
    result_dict = _validate(
        model=model,
        datamodule=datamodule,
        n_batches=n_batches,
    )

    metrics = _create_metrics(model=model, result_dict=result_dict)

    # prepare report

    name = ckpt_file.stem if ckpt_file else "baseline"
    out_dir = kpath(config_file.parent / "report", create=True)

    write_validation(
        name=name,
        out_dir=out_dir,
        result_dict=result_dict,
        metrics=metrics,
    )


def _create_metrics(model, result_dict):
    metrics = dict(overall={}, relations={})

    # i.e. head/tail
    for kind, results in result_dict.items():

        # overall
        metrics["overall"][kind] = _calc_metrics(model=model, results=results)

        # by relation
        for r_idx, r_str in model.meta["relations"].items():
            dic = metrics["relations"][r_str] = {}

            rel_results = [r for r in results if r.prediction.r_idx == r_idx]
            if not len(rel_results):
                continue

            dic[kind] = _calc_metrics(model=model, results=rel_results)

    return metrics


def _calc_metrics(model, results):
    rlis, preds, target = model.get_tm_tensors(rset=results)

    return dict(
        predictions=len(preds),
        micro_f1=tmf.f1(
            preds,
            target,
            average="micro",
        ).item(),
        # macro_f1=tmf.f1(
        #     preds,
        #     target,
        #     average="macro",
        #     num_classes=model.meta["num_classes"],
        # ).item(),
    )


def write_validation(
    out_dir: Path,
    name: str,
    result_dict: dict[str, models.ValidationResult],
    metrics: dict,
):
    log.info("writing csv file")
    with (out_dir / f"{name}.validation.yml").open(mode="w") as fd:
        yaml.dump(metrics, fd)

    for kind, rset in result_dict.items():
        out_file = out_dir / f"{name}.validation-{kind}.predictions.csv"

        results = sorted(rset)
        rows = [pred.dic for pred in results]

        with out_file.open(mode="w") as fd:
            writer = csv.DictWriter(fd, delimiter=SEP, fieldnames=rows[0].keys())
            writer.writeheader()

            for row in tqdm(rows):
                writer.writerow(row)


# ------------------------------------------------------------
# RANKING
#


A = Any


def opt_primitive(fn: Callable[[str], A], s: str) -> A:
    return int(s) if s != "None" else None


def tuplify(fn: Callable[[str], A], s: str) -> tuple[A]:
    return tuple(map(fn, eval(s)))


# the amount of data to be loaded into RAM is too damn high
# so strip it to the necessary fields (for caching and analysis)
@dataclass(frozen=True)
class RankedQuery:

    N = 10  # how many predictions to keep per query

    # TODO use dataclasses.fields(K)
    DECODER = (
        int,  # query identifier
        str,  # query context
        partial(opt_primitive, int),  # query eid
        int,  # pred r_idx
        str,  # pred r_str
        partial(tuplify, int),  # pred idxs
        partial(tuplify, str),  # pred strs
        partial(tuplify, float),  # pred scores
        partial(tuplify, float),  # pred scores_norm
    )

    query_identifier: int
    query_context: str
    query_eid: Optional[int]  # required for crossvalidation

    pred_r_idx: int
    pred_r_str: str

    pred_idxs: tuple[int]
    pred_strs: tuple[str]

    pred_scores: tuple[float]
    pred_scores_norm: tuple[float]

    @property
    def pred_idx(self) -> int:
        return self.pred_idxs[0]

    @property
    def pred_str(self) -> str:
        return self.pred_strs[0]

    @property
    def pred_score(self) -> float:
        return self.pred_scores[0]

    @property
    def pred_score_norm(self) -> float:
        return self.pred_scores_norm[0]

    @classmethod
    def from_query_result(K, qr: models.QueryResult):
        return K(
            query_identifier=qr.query.nomatch.identifier,
            query_context=qr.query.nomatch.context,
            query_eid=qr.query.nomatch.eid,
            pred_r_idx=qr.prediction.r_idx,
            pred_r_str=qr.prediction.r_str,
            pred_idxs=qr.prediction.idxs[: RankedQuery.N],
            pred_strs=qr.prediction.strs[: RankedQuery.N],
            pred_scores=qr.prediction.scores[: RankedQuery.N],
            pred_scores_norm=qr.prediction.scores_norm[: RankedQuery.N],
        )

    @classmethod
    def from_bytes(K, enc: bytes):
        tup = kstring.decode_line(enc, sep=text.SEP)
        args = [fn(x) for x, fn in zip(tup, K.DECODER)]

        return K(*args)

    def to_bytes(self):
        rep = tuple(map(str, astuple(self)))
        return kstring.encode_line(rep, sep=text.SEP)

    def __lt__(self, other) -> bool:
        a = (
            self.pred_idx,
            other.pred_score_norm,
            other.pred_score,
            self.query_context,
        )

        b = (
            other.pred_idx,
            self.pred_score_norm,
            self.pred_score,
            other.query_context,
        )

        return a < b

    @property
    def dic(self):
        return {
            "score norm": self.pred_score_norm,
            "score": self.pred_score,
            "predicted nid": self.pred_idx,
            "predicted node": self.pred_str,
            "relation": self.pred_r_str,
            "query eid": self.query_eid,
            "query identifier": self.query_identifier,
            "query context": self.query_context,
        }


def writer(path: Path, wid: int, q: mp.Queue):
    path = path.parent / (path.name + f".{wid}")
    bar = tqdm(desc=f"worker-{wid}", unit=" results", position=wid + 1, leave=False)

    log.info(f"[{wid}] spawned and awaiting data")
    with gzip.open(path, mode="w") as fd:

        while True:
            result = q.get()

            if result is None:
                log.info(f"[{wid}] received POD")
                break

            fd.write(result.to_bytes())
            bar.update()

    bar.close()
    log.info(f"[{wid}] exiting")


def reader(path: Path, wid: int, q: mp.Queue):
    path = path.parent / (path.name + f".{wid}")
    bar = tqdm(desc=f"worker-{wid}", unit=" results", position=wid + 1, leave=False)

    log.info(f"[{wid}] spawned, open file for reading")
    with gzip.open(path, mode="r") as fd:
        for line in fd:
            result = RankedQuery.from_bytes(line)
            q.put((True, result))
            bar.update()

    log.info(f"[{wid}] finished reading, sending POD")
    q.put((False, wid))

    bar.close()
    log.info(f"[{wid}] exiting")


class ResultCache(Exception):

    cache_file: Path

    def write(
        self,
        results: Iterable[RankedQuery],
        chunks: int = 5,
    ):
        log.info(f"[parent] writing {chunks} cache files")

        worker = deque((wid, mp.Queue()) for wid in range(chunks))

        procs = [
            (wid, mp.Process(target=writer, args=(self.cache_file, wid, q)))
            for wid, q in worker
        ]

        for _, proc in procs:
            proc.start()

        log.info(f"[parent] sending {len(results)} data points")

        for result in tqdm(results, desc="sender", unit=" results"):
            wid, q = worker.popleft()
            q.put(result)
            worker.append((wid, q))

        log.info("[parent] consumed data, sending PODS")
        for _, q in worker:
            q.put(None)

        for wid, proc in procs:
            log.info(f"[parent] join on writer [{wid}]")
            proc.join()

    def __init__(self, cache_file: Path):
        self.cache_file = cache_file

    @classmethod
    def load(
        K,
        path: Path,
        name: str,
        hash_value: str,
    ) -> Generator[models.QueryResult, None, None]:

        cache_dir = kpath(path / ".cache", create=True)
        cache_file = cache_dir / f"{name}.{hash_value}"

        if not (cache_file.parent / (cache_file.name + ".0")).exists():
            log.info(f"! cache miss for {name} hash={hash_value}")
            raise K(cache_file=cache_file)

        log.info(f"! loading {name} from cache")
        n = len(list(cache_file.parent.glob(cache_file.name + ".?")))

        log.info(f"[parent] found {n} cache files, starting worker processes")
        wids, q = set(range(n)), mp.Queue()
        procs = [
            (wid, mp.Process(target=reader, args=(cache_file, wid, q))) for wid in wids
        ]

        for _, proc in procs:
            proc.start()

        bar = tqdm(desc="loaded", unit=" results")
        while len(wids):
            has_data, payload = q.get()
            if not has_data:
                log.info(f"[parent] received POD from loader {payload}")
                wids -= {payload}
            else:
                yield payload
                bar.update()

        bar.close()
        for wid, proc in procs:
            log.info(f"[parent] join on loader [{wid}]")


class Ranking:

    name: str  # vanilla or crossval slug + split-{i}
    model_config: dict

    model_config_file: Path
    rank_config_file: Path
    queries_file: Path
    ckpt_file: Path
    crossval_path: Optional[Path]

    n_batches: Optional[int]
    threshold: float  # unbounded
    threshold_norm: float  # [0, 1)
    report_dir: Path

    # if crossval_path is set
    crossval_samples: Optional[crossval.Samples]
    crossval_graph: Optional[Graph]

    def _init_name(self):
        if "crossvalidation" not in self.model_config:
            self.name = "vanilla"
            return

        config = self.model_config["crossvalidation"]
        name = crossval.create_name(config=config)
        self.name = name + f".split-{config['split']}"

    def _init_crossval(self):
        if "crossvalidation" not in self.model_config:
            self.crossval_path = None
            return

        log.info("crossvalidation setting detected")

        config = self.model_config["crossvalidation"]

        name = crossval.create_name(config=config)
        log.info(f"running crossvalidation: {name}")

        graph_name = self.model_config["graph"]
        path = kpath(
            draug.ENV.DIR.HOMAG_CROSSVAL / graph_name / name,
            is_dir=True,
        )

        samples = crossval.Samples(path=path)

        # checks for k, n and seed
        _ref = samples.config["split"]
        assert all(config[key] == _ref[key] for key in _ref)

        split = config["split"]
        assert split in samples.splits

        graph_dir = draug.ENV.DIR.HOMAG_GRAPH / graph_name / "crossval"
        graph_dir = kpath(graph_dir / name / f"split-{split}", is_dir=True)
        graph = Graph.from_dir(path=graph_dir)

        # --

        self.crossval_path = path
        self.crossval_samples = samples
        self.crossval_graph = graph

    def __init__(
        self,
        config: Union[str, Path],
        queries: Union[str, Path],
        ckpt: Union[str, Path],
        n_batches: Optional[int] = None,
        threshold: Optional[float] = None,
        threshold_norm: Optional[float] = None,
    ):
        model_config_file = kpath(config, is_file=True)
        queries_file = kpath(queries, is_file=True)
        ckpt_file = kpath(ckpt, is_file=True)

        self.model_config = load_model_config(config_file=model_config_file)
        self.model_config_file = model_config_file
        self.queries_file = queries_file
        self.ckpt_file = ckpt_file

        self.n_batches = n_batches

        # name and target
        self._init_name()
        log.info(f"create ranker for: {self.name}")

        self._init_crossval()
        if self.crossval_path is not None:
            log.info(f"initialized crossvalidation: {self.crossval_path.name}")

        self.report_dir = kpath(
            model_config_file.parent / "report" / ckpt_file.stem / self.name,
            create=True,
        )
        log.info(f"using report dir: {self.report_dir}")

        # threshold
        if threshold_norm is not None and not 0 <= threshold_norm < 1:
            raise draug.DraugError("threshold must be 0 <= threshold < 1")

        if threshold is None:
            threshold = 0.0

        self.threshold = threshold
        log.info(f"setting threshold to {self.threshold}")

        if threshold_norm is None:
            threshold = 0.0

        self.threshold_norm = threshold
        log.info(f"setting threshold_norm to {self.threshold}")

        # persistence
        prefix = f"n_{self.n_batches}." if self.n_batches is not None else ""

        self.rank_config_file = self.report_dir / f"{prefix}ranking.config.yml"
        with self.rank_config_file.open(mode="w") as fd:
            yaml.dump(self.dic, fd)

    @property
    def dic(self):
        def _norm_path(p: Optional[Path]):
            if p is None:
                return None
            return str(p.resolve().relative_to(draug.ENV.DIR.ROOT))

        return dict(
            name=self.name,
            model_config_file=_norm_path(self.model_config_file),
            rank_config_file=_norm_path(self.rank_config_file),
            queries_file=_norm_path(self.queries_file),
            ckpt_file=_norm_path(self.ckpt_file),
            crossvalidation=_norm_path(self.crossval_path),
            report_dir=_norm_path(self.report_dir),
            n_batches=self.n_batches,
        )

    # ---

    def _load_queries(self) -> td.DataLoader:

        ds_queries = data.QueryDataset(path=self.queries_file)
        dl_queries = td.DataLoader(
            ds_queries,
            collate_fn=ds_queries.collate_fn,
            batch_size=self.model_config["valid_loader"]["batch_size"],
        )

        return dl_queries

    def _run(self):

        dl_queries = self._load_queries()
        datamodule = load_data(config=self.model_config)

        model = load_model(
            config=self.model_config,
            ckpt_file=self.ckpt_file,
            datamodule=datamodule,
        )

        results = set()

        gen = islice(dl_queries, self.n_batches)
        for batch in tqdm(gen, total=self.n_batches or len(dl_queries)):
            batch = datamodule.transfer_batch_to_device(
                batch, model.device, 0
            )  # TODO hard-coded dataloader index
            query_results = model.query(*batch)
            results |= set(map(RankedQuery.from_query_result, query_results))

        return tuple(results)

    def get_results(self) -> tuple[RankedQuery]:
        try:
            results = tuple(
                ResultCache.load(
                    path=self.report_dir,
                    name="ranking",
                    hash_value=kstring.args_hash(repr(self.dic)),
                )
            )

        except ResultCache as cache:
            log.info("cache miss, commence ranking")
            results = self._run()

            log.info("saving rankings to cache")
            cache.write(results)

        return results

    # --

    def _write_rank_stats(self, fname, lis: list[models.QueryResult]):
        log.info("aggregating stats")

        counter = Counter()
        for res in lis:
            counter[str(res.prediction)] += 1

        counts = sorted(counter.items(), key=lambda t: t[1])

        with (self.report_dir / f"{fname}.counts.csv").open(mode="w") as fd:
            for name, count in counts:
                fd.write(f"{name}|{count}\n")

    def write(
        self,
        name: str,
        out: Union[str, Path],
        data: dict[str, RankedQuery],
    ):
        log.info("writing csv file")
        log.info(f"tresholding at {self.threshold} norm={self.threshold_norm}")

        out = kpath(out, create=True)
        for category, lis in data.items():

            fname = (
                f"{name}.{category}."
                f"{self.threshold:04.2f}.{self.threshold_norm:04.2f}"
            )

            if self.n_batches is not None:
                fname = f"n_{self.n_batches}.{fname}"

            log.info(f"writing {fname}")
            with (out / f"{fname}.csv").open(mode="w") as fd:

                header = list(lis[0].dic.keys())
                writer = csv.DictWriter(fd, delimiter=SEP, fieldnames=header)
                writer.writeheader()

                for result in lis:

                    if result.pred_score_norm < self.threshold_norm:
                        continue

                    if result.pred_score < self.threshold:
                        continue

                    writer.writerow(result.dic)


def rank(*args, **kwargs):
    ranking = Ranking(*args, **kwargs)

    log.info("partition results by prediction and sort")
    results = buckets(
        ranking.get_results(),
        key=lambda _, rp: (rp.pred_r_str, rp),
        mapper=lambda rps: sorted(rps),
    )

    log.info("writing predictions")
    ranking.write(
        name="ranking",
        out=ranking.report_dir,
        data=results,
    )


# ------------------------------------------------------------
# CROSSVALIDATION
#


@dataclass(frozen=True)
class Rank:

    rid: int
    nid: int
    name: str

    correct: tuple[EID]
    candidates: dict[EID, int]  # count of possible matches

    # e.g. [None, None, 23, None, 26]
    ranks: tuple[Optional[EID]]

    # e.g. [False, False, False, False, True]
    positions: tuple[bool]

    @property
    def position(self) -> Optional[int]:
        try:
            return self.positions.index(True) + 1
        except ValueError:
            return None

    @property
    def found_total(self):
        return tuple(x for x in self.ranks if x is not None)

    @property
    def found_correct(self):
        return tuple(x for x in self.found_total if x in self.correct)

    @property
    def dic(self) -> dict:
        return {
            "nid": self.nid,
            "name": self.name,
            "hit": self.position,
            "total": len(self.positions),
            "correct found": len(self.found_correct),
            "total correct": len(self.correct),
            "total found": len(self.found_total),
            "candidates": sum(self.candidates.values()),
        }

    @property
    def metrics(self) -> dict:
        return {"foo": 0}

    @classmethod
    def create(
        K,
        nid: NID,
        rid: RID,
        correct: set[EID],
        results: dict[NID, list[RankedQuery]],
        queries: dict[EID, list[text.Nomatch]],
        graph: Graph,
    ):

        # no predictions were made
        if nid not in results:
            ranks, truth = [], []
        else:
            ranks = [res.query_eid for res in results[nid]]
            truth = [eid in correct for eid in ranks]

        candidates = {eid: len(queries[eid]) for eid in correct}

        return K(
            nid=nid,
            rid=rid,
            name=graph.node_name(nid=nid),
            ranks=tuple(ranks),
            positions=tuple(truth),
            correct=tuple(correct),
            candidates=candidates,
        )


def get_crossval_ranks(ranking: Ranking) -> dict[RID, dict[NID, Rank]]:

    config = ranking.model_config["crossvalidation"]
    split = ranking.crossval_samples.splits[config["split"]]

    print(f"running crossvalidation for {ranking.name}")

    # create rid -> nid -> {eid, ...} mapping
    # i.e.
    #  for parents: parent -> { children }
    #  for synonyms: cluster -> { siblings }
    samples = buckets(
        split.samples,
        key=lambda _, sample: (sample.rid, sample),
        mapper=lambda samples: buckets(
            samples,
            key=lambda _, sample: (sample.nid, sample.eid),
            mapper=lambda eids: set(eids),
        ),
    )

    # creates rid -> nid -> [query results] mapping
    # i.e.
    #  for parents: predicted parent -> sorted([children context 1, ...])
    #  for synonyms: predicted cluster -> sorted([synonym context 1, ...])
    results = buckets(
        ranking.get_results(),
        key=lambda _, result: (result.pred_r_idx, result),
        mapper=lambda results: buckets(
            results,
            key=lambda _, result: (result.pred_idx, result),
            mapper=lambda results: sorted(results),
        ),
    )

    # creates Optional[eid] -> list[Nomatch] mapping
    queries = data.QueryDataset(path=ranking.queries_file).flat
    queries = buckets(
        [qs.nomatch for qs in queries],
        key=lambda _, nomatch: (nomatch.eid, nomatch),
    )

    # obtain ranks per (RID, NID)
    rankdic: dict[RID, list[Rank]] = {}
    for rid in samples:
        print(f"predictions for {Graph.RELATIONS(rid)}")

        rankdic[rid] = []
        for nid in samples[rid]:
            rankdic[rid].append(
                Rank.create(
                    nid=nid,
                    rid=rid,
                    correct=samples[rid][nid],
                    results=results[rid],
                    queries=queries,
                    graph=ranking.crossval_graph,
                )
            )

    return rankdic, results, queries


def _crossvalidate_write(
    ranking: Ranking,
    rid: RID,
    ranks: list[Rank],
    results: dict[RID, dict[NID, models.QueryResult]],
    selected: dict[str, list],
):
    rname = Graph.RELATIONS(rid).name
    fname = ranking.report_dir / f"crossval.{rname}.csv"

    print(f"writing rankings for {rname} to {fname.name}")
    with fname.open(mode="w") as fd:

        writer = csv.DictWriter(fd, fieldnames=list(ranks[0].dic.keys()))
        writer.writeheader()

        for rank in ranks:
            writer.writerow(rank.dic)
            if rank.nid in results[rid]:
                selected[f"{rname}.{rank.nid}"] += results[rid][rank.nid]


@dataclass
class CrossvalMetrics:

    hits: int
    expected: int
    precision: float
    mean: float
    median: float
    mrr: float

    @classmethod
    def create(
        Self,
        ranks: list[Rank],
    ):
        positions = [rank.position for rank in ranks]
        hits = [pos for pos in positions if pos is not None]

        precision = len(hits) / len(positions)
        mean = stats.mean(hits)
        median = stats.median(hits)
        mrr = 1 / stats.harmonic_mean(hits)

        return Self(
            hits=len(hits),
            expected=len(positions),
            precision=precision,
            mean=mean,
            median=median,
            mrr=mrr,
        )


def crossvalidate(
    *args,
    **kwargs,
):

    ranking = Ranking(*args, **kwargs)

    # rankdic: dict[RID, list[Rank]]
    # results: dict[RID, dict[NID, QueryResult]]
    # queries: dict[EID, list[Nomatch]]
    rankdic, results, queries = get_crossval_ranks(ranking=ranking)

    selected = defaultdict(list)
    metricdic: dict[str, CrossvalMetrics] = {}

    for rid, ranks in rankdic.items():
        _crossvalidate_write(
            ranking=ranking,
            rid=rid,
            ranks=ranks,
            results=results,
            selected=selected,
        )

        metrics = CrossvalMetrics.create(ranks=ranks)
        metricdic[Graph.RELATIONS(rid).name] = metrics

    ranking.write(
        name="ranking",
        out=ranking.report_dir / "crossval",
        data=selected,
    )

    # aggregate metrics over all relations
    metricdic["all"] = CrossvalMetrics.create(
        ranks=[rank for ranklis in rankdic.values() for rank in ranklis]
    )

    log.info("write crossvalidation metrics")
    fname = ranking.report_dir / "metrics.csv"
    with fname.open(mode="w") as fd:

        dics = [
            ({"relation": rstr} | asdict(metrics))
            for rstr, metrics in metricdic.items()
        ]

        writer = csv.DictWriter(fd, fieldnames=dics[0].keys())
        writer.writeheader()
        for dic in dics:
            writer.writerow(dic)
