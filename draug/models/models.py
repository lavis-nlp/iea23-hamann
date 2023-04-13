# -*- coding: utf-8 -*-

import draug
from draug.common import helper
from draug.homag.text import SEP
from draug.homag.graph import Graph
from draug.models import data

import torch
from torch import nn
import transformers as tf
import torchmetrics as tm
import pytorch_lightning as pl

from tqdm import tqdm as _tqdm

import enum
import logging
import textwrap
from functools import partial
from dataclasses import field
from dataclasses import dataclass
from collections import defaultdict

from typing import Any
from typing import Literal
from typing import Optional

log = logging.getLogger(__name__)
tqdm = partial(_tqdm, ncols=80)


OPTIMIZER = {
    "adam": torch.optim.Adam,
}


LOSSES = {
    "negative log likelihood": nn.NLLLoss,
    "multi margin": nn.MultiMarginLoss,
    "triplet margin": nn.TripletMarginLoss,
    "margin ranking": nn.MarginRankingLoss,
    "cross entropy": nn.CrossEntropyLoss,
    "binary cross entropy": nn.BCELoss,
    "binary cross entropy with logits": nn.BCEWithLogitsLoss,
}

INITIALIZATIONS = {
    "normal": nn.init.normal_,
    "uniform": nn.init.uniform_,
}


class Loglabel(enum.Enum):

    training_loss = "training/loss"
    validation_loss = "validation/loss"

    training_loss_enc2enc = "training/loss/enc2enc"
    training_loss_emb2emb = "training/loss/emb2emb"
    training_loss_emb2enc = "training/loss/emb2enc"
    training_loss_enc2emb = "training/loss/enc2emb"

    training_score_raw_f = "training/score/raw_false"
    training_score_raw_t = "training/score/raw_true"
    training_score_sig_f = "training/score/sigmoid_false"
    training_score_sig_t = "training/score/sigmoid_true"
    training_ember_hw2 = "training/embeddings/heads norm-2"
    training_ember_ew2 = "training/embeddings/entities norm-2"
    training_ember_rw2 = "training/embeddings/relations norm-2"
    training_ember_reg = "training/loss/regularization term"

    val_f1_micro = "validation/f1/micro"
    val_f1_micro_head = "validation/f1/micro/head"
    val_f1_micro_tail = "validation/f1/micro/tail"

    val_f1_macro = "validation/f1/macro"
    val_f1_macro_head = "validation/f1/macro/head"
    val_f1_macro_tail = "validation/f1/macro/tail"


class Metric(enum.Enum):

    micro_f1 = enum.auto()
    macro_f1 = enum.auto()


METRICS = {
    Metric.micro_f1: (
        tm.F1,
        dict(average="micro"),
        Loglabel.val_f1_micro,
    ),
    Metric.macro_f1: (
        tm.F1,
        dict(average="macro"),
        Loglabel.val_f1_macro,
    ),
}


@dataclass(frozen=True)
class Prediction:

    r_str: str
    r_idx: int

    # model scores
    idxs: tuple[int]
    strs: tuple[str]
    scores: tuple[float]

    # internally used cached values
    _cache: dict = field(
        default_factory=dict,
        compare=False,
        hash=False,
    )

    @property
    def to_bytes(self) -> bytes:
        def _join(col):
            return " ".join(map(str, col))

        return helper.encode_line(
            data=(
                self.r_str,
                str(self.r_idx),
                _join(self.idxs),
                _join(self.strs),
                _join(self.scores),
            ),
            sep=SEP,
        )

    @classmethod
    def from_bytes(K, encoded: bytes):
        cols = helper.decode_line(encoded, sep=SEP)
        r_str, r_idx, idxs, strs, scores = cols

        def _split(s, fn):
            return tuple(map(fn, s.split()))

        return K(
            r_str=r_str,
            r_idx=int(r_idx),
            idxs=_split(idxs, int),
            strs=_split(strs, str),
            scores=_split(scores, float),
        )

    def __lt__(self, other):
        # note: scores are switched to have them descending
        a = other.score_norm, other.score
        b = self.score_norm, self.score
        return a < b

    def __str__(self):
        return f"{self.strs[0]} ({self.score_norm:.2f}/{self.score:.3f})"

    @property
    def idx(self) -> int:
        return self.idxs[0]

    @property
    def score(self) -> float:
        return self.scores[0]

    @property
    def score_norm(self) -> float:
        return self.scores_norm[0]

    @property
    def scores_norm(self) -> tuple[float]:
        try:
            norm = self._cache["scores_norm"]
        except KeyError:
            t = torch.tensor(self.scores)
            norm = nn.functional.softmax(t, dim=0)
            norm = tuple(norm.tolist())
            self._cache["scores_norm"] = norm

        return norm


@dataclass(frozen=True)
class ValidationResult:

    query: data.TextSample  # heads for tail prediction
    prediction: Prediction

    target_idx: int
    target_str: str

    # same for all objects (not relevant for eq/hashing)
    g: Graph = field(compare=False, hash=False)

    def __str__(self) -> str:
        correct = "correct" if self.correct else "incorrect"
        return (
            f"Prediction for ({self.query.label_str}, {self.prediction.r_str}, ?)"
            f" -> {self.prediction} ({correct})"
        )

    def __lt__(self, other) -> bool:
        a = (
            self.prediction.idx,
            other.prediction,
            self.query.match.context,
        )
        b = (
            other.prediction.idx,
            self.prediction,
            other.query.match.context,
        )

        return a < b

    @property
    def description(self) -> str:
        n = 3

        query = textwrap.indent(
            self.query.description,
            prefix=" " * 4,
        )

        correct = "✓" if self.correct else "✗"

        target = f"{self.target_str} ({self.target_idx})"

        top = (
            f"Prediction {correct}\n"
            f"  Query:\n{query}\n"
            f"  Relation: {self.prediction.r_str}\n"
            f"  Target: {target}\n\n"
            f"Top {n} Predictions:\n"
        )

        pred = self.prediction
        zipped = list(zip(pred.idxs, pred.strs, pred.scores))
        preds = "\n".join(f"  {s} ({i}) [{m:.2f}]" for i, s, m in zipped[:n])

        return top + preds

    @property
    def correct(self) -> bool:
        return self.prediction.idxs[0] == self.target_idx

    @property
    def dic(self) -> dict[str, Any]:
        eid = self.query.match.eid
        nid = self.g.get_entity(eid=eid).nid

        return {
            # prediction
            "correct": self.correct,
            "p1 score": self.prediction.scores[0],
            "p1 norm score": self.prediction.scores_norm[0],
            "p1 name": self.prediction.strs[0],
            "p2 score": self.prediction.scores[1],
            "p2 norm score": self.prediction.scores_norm[1],
            "p2 name": self.prediction.strs[1],
            "p3 score": self.prediction.scores[2],
            "p3 norm score": self.prediction.scores_norm[2],
            "p3 name": self.prediction.strs[2],
            # gt
            "query relation": self.prediction.r_str,
            "target entity": self.target_str,
            # match
            "query ticket": self.query.match.ticket,
            "query nid": nid,
            "query node": self.g.node_name(nid=nid),
            "query eid": eid,
            "query entity": self.g.entity_name(eid=eid),
            "query mention": self.query.match.mention,
            "query context": self.query.match.context,
        }

        return (
            self.correct,
            self.prediction.score,
            str(self.prediction),
            self.target_str,
            self.prediction.r_str,
            # match
            self.query.match.ticket,
            self.query.match.eid,
            self.query.match.context,
        )


@dataclass(frozen=True)
class QueryResult:

    query: data.QuerySample
    prediction: Prediction

    def __lt__(self, other) -> bool:
        a = (
            self.prediction.idx,
            self.prediction,
            self.query.nomatch.context,
        )
        b = (
            other.prediction.idx,
            other.prediction,
            other.query.nomatch.context,
        )

        return a < b

    def __str__(self) -> str:
        return f'Query Result: {self.prediction}: "{self.query.nomatch.context}"'

    @property
    def dic(self) -> dict[str, Any]:
        return {
            "score": self.prediction.score,
            "score norm": self.prediction.score_norm,
            "predicted nid": self.prediction.idx,
            "predicted node": str(self.prediction),
            "relation": self.prediction.r_str,
        } | {f"query {k}": v for k, v in self.query.nomatch.dic.items()}


class DraugModule(pl.LightningModule):

    config: dict
    encoder: tf.BertModel

    metrics: dict[Loglabel, tuple]

    def _log_scores(self, scores, target, loglabel_t, loglabel_f):
        assert scores.shape == target.shape

        self.log(
            loglabel_t.value,
            scores[target != 0].mean(),
            on_step=True,
            on_epoch=False,
        )

        self.log(
            loglabel_f.value,
            scores[target == 0].mean(),
            on_step=True,
            on_epoch=False,
        )

    def _log_training_step(self, heads, scores, target, loss):

        self._log_scores(
            scores=scores,
            target=target,
            loglabel_t=Loglabel.training_score_raw_t,
            loglabel_f=Loglabel.training_score_raw_f,
        )

        # embedding weight logging

        self.log(
            Loglabel.training_ember_hw2.value,
            torch.norm(heads, p=2, dim=1).mean(),
            on_step=True,
            on_epoch=False,
        )

        self.log(
            Loglabel.training_ember_ew2.value,
            torch.norm(self.entities.weight, p=2, dim=1).mean(),
            on_step=True,
            on_epoch=False,
        )

        self.log(
            Loglabel.training_ember_rw2.value,
            torch.norm(self.relations.weight, p=2, dim=1).mean(),
            on_step=True,
            on_epoch=False,
        )

        # loss logging

        self.log(
            Loglabel.training_loss.value,
            loss,
            on_step=True,
            on_epoch=True,
        )

    def _check_config(self):
        loss = self.config["loss"]
        if loss not in self.supported_losses:
            raise draug.DraugError(
                f'unsupported loss: "{loss}\n"'
                f"supported losses: {self.supported_losses}"
            )

        training = self.config["train_dataset"]
        if training not in self.supported_training:
            raise draug.DraugError(
                f'unsupported training: "{training}"'
                f"supported trainings: {self.supported_training}"
            )

    def __init__(
        self,
        bert: tf.BertModel,
        config: dict[str, Any],
        datamodule: data.DataModule,
        **kwargs,
    ):
        super().__init__()

        self.config = config
        self._check_config()

        self.encoder = bert
        self.meta = datamodule.meta
        self.metrics = {}

        if "tokenizer" in datamodule.train_ds_kwargs:
            tok_path = datamodule.train_ds_kwargs["tokenizer"]
        else:
            tok_path = datamodule.path

        # TODO use data._init_tokenizer
        tokenizer = tf.BertTokenizer.from_pretrained(str(tok_path / "tokenizer"))
        self.mask_token = tokenizer.vocab[tokenizer.mask_token]

    def configure_optimizers(self):
        optimizer = OPTIMIZER[self.config["optimizer"]](
            self.parameters(), **self.config["optimizer_kwargs"]
        )

        log.info(
            f"initialized {self.config['optimizer']} with"
            f" {self.config['optimizer_kwargs']}"
        )

        return [optimizer]

    # ---

    def encode(self, batch):
        attention_mask = (batch > 0) | (batch == self.mask_token)
        attention_mask = attention_mask.to(dtype=torch.long)

        encoded = self.encoder(input_ids=batch, attention_mask=attention_mask)
        return encoded[0]

    def forward(self, batch):
        return self.encode(batch)

    # helper

    def reset_metrics(self):
        log.info("resetting metrics")
        for fn in self.metrics.values():
            fn.reset()

    def init_metric(self, metric: Metric, label: Optional[Loglabel]):
        K, kwargs, loglabel = METRICS[metric]
        loglabel = label or loglabel

        log.info(f"initializing {loglabel.value}: {kwargs}")
        self.metrics[loglabel] = K(num_classes=self.meta["num_classes"], **kwargs)

    # interface

    def predict(self, batch) -> dict[str, Prediction]:
        raise NotImplementedError()

    def training_step(self, *_):
        raise NotImplementedError()

    def validation_step(self, *_):
        raise NotImplementedError()


class JointKGC(DraugModule):
    """

    NTN/DKRL/ComplEx-like KGC

    B: batch size
    E: entity count
    R: relation type count
    T: token count for text sequences
    D: entity/relation embedding dimensionality
    N: negatives (when negative sampling is used)

    """

    def __init__(
        self,
        bert: str,
        config: dict[str, Any],
        datamodule: data.DataModule,
        **kwargs,
    ):
        bert = tf.BertModel.from_pretrained(
            bert, cache_dir=draug.ENV.DIR.CACHE / "lib.transformers"
        )

        super().__init__(bert=bert, config=config, datamodule=datamodule)
        self.graph = datamodule.graph

    def predict(
        self,
        r_idxs: list[int],  # B
        scoremat: torch.Tensor,  # B x E
    ) -> set[Prediction]:

        predictions = []
        scoremat = scoremat.sort(dim=1, descending=True)  # B x E

        for i, r_idx in enumerate(r_idxs):

            idxs = tuple(scoremat.indices[i].tolist())
            strs = tuple(self.meta["idx2str"][idx] for idx in idxs)

            scores = tuple(scoremat.values[i].tolist())

            pred = Prediction(
                r_idx=r_idx,
                r_str=self.meta["relations"][r_idx],
                idxs=idxs,
                strs=strs,
                scores=scores,
            )

            predictions.append(pred)

        return predictions

    def validate(
        self,
        graph_samples: list[data.GraphSample],  # B
        batch: data.GraphCollation,  # B
    ) -> dict[str, set[ValidationResult]]:

        h_enc = self.forward(batch=batch.h_tokens)  # B x D
        r_emb = self.relations(batch.r_idxs)  # B x D
        r_idxs = tuple(t.r_idx for t in graph_samples)  # B

        t_scores = self.score_tails(h=h_enc, r=r_emb)  # B x E
        t_preds = self.predict(r_idxs=r_idxs, scoremat=t_scores)  # B

        def _create_results(queries, targets, predictions):
            results = set()
            for query, (t_idx, t_str), prediction in zip(queries, targets, predictions):
                results.add(
                    ValidationResult(
                        g=self.graph,
                        query=query,
                        target_idx=t_idx,
                        target_str=t_str,
                        prediction=prediction,
                    )
                )
            return results

        t_results = _create_results(
            queries=[s.h for s in graph_samples],
            targets=[(s.t_idx, s.t_str) for s in graph_samples],
            predictions=t_preds,
        )

        return {"tail": t_results}

    def query(
        self,
        query_samples: tuple[str],
        batch: torch.Tensor,  # B x T
    ) -> list[QueryResult]:
        B = batch.shape[0]
        R = self.relations.num_embeddings

        h_enc = self.forward(batch=batch)  # B x D
        h_enc = torch.vstack([h_enc for _ in range(R)])  # B*R x D

        r_emb = self.relations.weight
        r_emb = r_emb.repeat_interleave(B, dim=0)  # B*R x D

        tail_scores = self.score_tails(h=h_enc, r=r_emb)

        results = set()
        for r_idx, scoremat in enumerate(tail_scores.chunk(R)):

            tail_preds = self.predict(
                r_idxs=[r_idx for _ in range(len(scoremat))],
                scoremat=scoremat,
            )

            results |= set(
                QueryResult(
                    query=query_samples[i],
                    prediction=tail_preds[i],
                )
                for i in range(B)
            )

        return results

    # ---

    def forward(
        self,
        batch: torch.Tensor,  # B x T
    ) -> torch.Tensor:  # B x E
        return self.ff(self.encode(batch)[:, 0])

    def score_heads(
        self,
        t: torch.Tensor,  # B x D
        r: torch.Tensor,  # B x D
    ) -> torch.Tensor:  # B x E

        D = r.shape[1]
        expand = -1, 1, D
        wrap = 1, -1, D

        return self.score(
            h=self.entities.weight.view(*wrap),
            t=t.view(*expand),
            r=r.view(*expand),
        )

    def score_tails(
        self,
        h: torch.Tensor,
        r: torch.Tensor,
    ) -> torch.Tensor:  # B x E

        D = r.shape[1]
        expand = -1, 1, D
        wrap = 1, -1, D

        return self.score(
            h=h.view(*expand),
            t=self.entities.weight.view(*wrap),
            r=r.view(*expand),
        )

    # --- lightning methods

    def validation_step(self, batch: dict, *_):
        graph_samples, collation = batch

        results = self.validate(graph_samples=graph_samples, batch=collation)

        for kind, rset in results.items():
            self._val_results[kind] |= rset

    # --- hooks

    def on_fit_start(self):
        log.info("starting to fit")

        for kwargs in (
            # heads
            # dict(metric=Metric.micro_f1, label=Loglabel.val_f1_micro_head),
            # dict(metric=Metric.macro_f1, label=Loglabel.val_f1_macro_head),
            # tails
            dict(metric=Metric.micro_f1, label=Loglabel.val_f1_micro_tail),
            # dict(metric=Metric.macro_f1, label=Loglabel.val_f1_macro_tail),
        ):
            self.init_metric(**kwargs)

    def on_train_epoch_start(self):
        log.info(
            f"! starting epoch {self.current_epoch}"
            f" (step={self.global_step});"
            f" running on {self.device}"
        )

    def on_validation_epoch_start(self):
        log.info("validation epoch start: clearing predictions")
        self._val_results = defaultdict(set)
        self.reset_metrics()

    def on_validation_epoch_end(self):
        if not len(self._val_results):
            log.info("skipping metric calculation (no validation results)")
            return

        log.info("logging aggregated predictions")
        for kind, rset in self._val_results.items():

            # if kind == "head":
            #     metrics = (
            #         Loglabel.val_f1_micro_head,
            #         # Loglabel.val_f1_macro_head,
            #     )
            if kind == "tail":
                metrics = (
                    Loglabel.val_f1_micro_tail,
                    # Loglabel.val_f1_macro_tail,
                )
            else:
                assert False

            _, preds, target = self.get_tm_tensors(rset)
            log.info(f'calculating "{kind}" metrics from {len(preds)} predictions')

            for label in metrics:
                val = self.metrics[label](preds, target)

                self.log(label.value, val, on_step=False, on_epoch=True)
                log.info(f"    {label.value}: {val:.3f}")

    # ---

    def get_tm_tensors(self, rset: set[ValidationResult]):
        # this is a tad hacky: to work with torchmetrics, we change
        # the actual prediction to the expected ones for all relations
        # that have the transitive closure: If class 1 is true but the
        # model predicted class 2 the predicted class is changed from
        # 2 to 1 if the both classes are in the same transitive closure

        rlis, target, preds = [], [], []
        for result in rset:
            target.append(result.target_idx)
            preds.append(target[-1] if result.correct else result.prediction.idx)
            rlis.append(result)

        return rlis, torch.tensor(preds), torch.tensor(target)

    # --- interface

    def score(self, h, t, r) -> torch.Tensor:
        raise NotImplementedError()


class JointKGC3(JointKGC):
    """

    changes:
    - heads are always encoded text
    - tails are always embeddings

    """

    @dataclass
    class EmbeddingConfig:

        dimensions: int

        # embedding initialization
        initialization: Optional[Literal["normal", "uniform"]] = None
        # e.g. mean=0.0, std=1.0 for "normal"
        # you can use "reciprocal": 1/self.dimensions
        initialization_kwargs: Optional[dict] = None

        # weight decay on embedding weights
        regularization: Optional[dict] = None

    def _reset_embeddings(self):
        config = self._embedding_config

        if config.initialization is None:
            log.info("not initializing the embeddings explicitly")
            return

        assert config.initialization_kwargs
        assert config.initialization in INITIALIZATIONS

        log.info(f"resetting embeddings: {config.initialization}")

        init = INITIALIZATIONS[config.initialization]
        kwargs = {}
        for k, v in config.initialization_kwargs.items():
            if v == "reciprocal":
                v = 1 / config.dimensions
            kwargs[k] = v

        log.info(f"intialization kwargs: {kwargs}")
        for ember in (self.entities, self.relations):
            init(ember.weight, **kwargs)

    def _init_embeddings(self):
        config = self._embedding_config

        # real and imaginary parts
        embedding_dim = config.dimensions * 2  # real + imag
        log.info(f"set up embeddings with {embedding_dim} dimensions")

        self.relations = torch.nn.Embedding(
            num_embeddings=len(self.meta["relations"]),
            embedding_dim=embedding_dim,
        )

        self.entities = torch.nn.Embedding(
            num_embeddings=self.meta["num_classes"],
            embedding_dim=embedding_dim,
        )

        self.ff = nn.Linear(
            in_features=self.encoder.config.hidden_size,
            out_features=embedding_dim,
        )

    # ---

    def __init__(
        self,
        embedding: dict,
        config: dict,
        regularization: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)

        self._embedding_config = JointKGC3.EmbeddingConfig(**embedding)
        self._init_embeddings()
        self._reset_embeddings()

        log.info(f"initializing loss: '{config['loss']}' {config['loss_kwargs']} ")
        self._loss = LOSSES[config["loss"]](**config["loss_kwargs"])
        self._regularization = self._embedding_config.regularization

    def loss(self, output, target):
        loss = self._loss(output, target)

        if self._regularization is not None:
            p = self._regularization["p"]
            factor = self._regularization["factor"]

            E = torch.vstack((self.entities.weight, self.relations.weight))
            value = factor * torch.norm(E, p=p, dim=1).mean()
            loss += value

            self.log(
                Loglabel.training_ember_reg.value,
                value,
                on_step=True,
                on_epoch=False,
            )

        return loss

    def score(self, h, t, r) -> torch.Tensor:
        # torch.sum(h * r * t, dim=-1)  distmult
        # complex kgc objective:

        # split in real and imaginary parts
        dim = self.entities.embedding_dim // 2

        h_r, h_i = h[..., :dim], h[..., dim:]
        t_r, t_i = t[..., :dim], t[..., dim:]
        r_r, r_i = r[..., :dim], r[..., dim:]

        # like distmult, but following the
        # rules of complex number algebra

        return sum(
            (hh * rr * tt).sum(dim=-1)
            for hh, rr, tt in [
                (h_r, r_r, t_r),
                (h_r, r_i, t_i),
                (h_i, r_r, t_i),
                (-h_i, r_i, t_r),
            ]
        )


class JointKGC3_CE1(JointKGC3):
    """

    BERT + ComplEx
    Cross Entropy Version 1

    """

    supported_losses = {"cross entropy"}
    supported_training = {"graph selective"}

    def training_step(self, batch: tuple, *_):
        graph_samples, collation = batch

        B = len(graph_samples)
        E = self.entities.num_embeddings

        # encode texts and select CLS
        h_enc = self.forward(batch=collation.h_tokens)  # B x D
        r_emb = self.relations(collation.r_idxs)  # B x D

        # B x E x D
        h, r = (
            # [0, 1, 2] -> [0, 0, 0, 1, 1, 1, 2, 2, 2]
            x.repeat_interleave(E, dim=0).view(B, E, -1)
            for x in (h_enc, r_emb)
        )

        # B x E x D
        t = torch.stack([self.entities.weight for _ in range(B)])

        # B x E
        scores = self.score(h, t, r)
        loss = self.loss(scores, collation.t_idxs)

        target = nn.functional.one_hot(
            collation.t_idxs,
            num_classes=self.meta["num_classes"],
        )

        self._log_training_step(
            heads=h_enc,
            scores=scores,
            target=target,
            loss=loss,
        )

        return loss


class JointKGC3_NS1(JointKGC3):
    """

    BERT + ComplEx
    Negative Sampling with BCE 1

    Takes positive and negative head samples
    Apply sigmoids to target (relation, tail) scores

    """

    supported_losses = {"binary cross entropy"}
    supported_training = {"graph negative heads"}

    def _log_training_step(self, heads, scores, output, target, loss):
        super()._log_training_step(
            heads=heads,
            scores=scores,
            target=target,
            loss=loss,
        )

        self._log_scores(
            scores=output,
            target=target,
            loglabel_t=Loglabel.training_score_sig_t,
            loglabel_f=Loglabel.training_score_sig_f,
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigmoid = nn.Sigmoid()

    def training_step(self, batch: tuple, *_):
        graph_samples, collation, target = batch

        # ---
        # graph samples:
        # (0, 0) -> first positive
        # (0, 1) -> first negative of first positive
        # (1, 0) -> second positive
        # (1, 1) -> first negative of second positive
        # etc...
        # text can be accessed like this:
        # graph_samples[0][0].h.match.context     - the first positive
        # graph_samples[0][1].h.nomatch.context   - the first negative
        # ---

        B = collation.h_tokens.shape[0]
        E = self.entities.num_embeddings
        P = len(graph_samples)  # num positives
        Q = self.config["train_dataset_kwargs"]["num_negatives"]

        assert (Q + 1) * P == B, "batch mismatch"

        # ---

        # encode texts and select CLS
        h_enc = self.forward(batch=collation.h_tokens)  # B x D
        r_emb = self.relations(collation.r_idxs)  # B x D

        # B x E x D
        h, r = (
            # [0, 1, 2] -> [0, 0, 0, 1, 1, 1, 2, 2, 2]
            x.repeat_interleave(E, dim=0).view(B, E, -1)
            for x in (h_enc, r_emb)
        )

        # B x E x D
        t = torch.stack([self.entities.weight for _ in range(B)])

        scores = self.score(h, t, r)  # B x E
        output = self.sigmoid(scores)  # B x E

        loss = self.loss(output, target)  # 1

        self._log_training_step(
            heads=h_enc,
            scores=scores,
            output=output,
            target=target,
            loss=loss,
        )

        return loss


# ---


def get_cls(name: str):
    mapping = {
        "joint kgc 3 ce 1": JointKGC3_CE1,
        "joint kgc 3 ns 1": JointKGC3_NS1,
        #
        # archived (see org/legacy.py):
        #
        # "joint kgc 4": JointKGC4,
        # "triplet distance 1": TripletDistance1,
        # "multiclass classifier 1": MulticlassClassifier1,
        # "joint kgc 1": JointKGC1,
        # "joint kgc 2": JointKGC2,
    }

    if name not in mapping:
        raise draug.DraugError(f"unknown model: {name}")

    return mapping[name]
