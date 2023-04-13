# -*- coding: utf-8 -*-

from draug import common
from draug.models import models
from draug.models import predict

import elasticsearch
import torch.utils.data as td
from tqdm import tqdm as _tqdm

import logging
import argparse
import dataclasses
from itertools import islice
from functools import partial

from typing import Any
from typing import Optional


log = logging.getLogger(__name__)
tqdm = partial(_tqdm, ncols=80)


class Elastic:

    idx_name: str

    def _init_es(self):
        self.es = elasticsearch.Elasticsearch()

    def _init_metrics(self):
        self.metrics = {}
        for name, (K, kwargs, loglabel) in models.METRICS.items():
            log.info(f"initializing {name}: {kwargs}")

            self.metrics[name] = (
                loglabel,
                K(num_classes=self.meta["num_classes"], **kwargs),
            )

    def __init__(self, meta: dict[str, Any], idx_name: str):
        self.meta = meta
        self.idx_name = idx_name
        self._init_metrics()
        self._init_es()

    def index(self, loader: td.DataLoader):
        res = self.es.indices.delete(index=self.idx_name, ignore_unavailable=True)
        assert res["acknowledged"]

        def gen():
            for batch in tqdm(loader, total=len(loader.dataset)):
                samples, _ = batch
                assert len(samples) == 1

                yield {
                    "_index": self.idx_name,
                } | dataclasses.asdict(samples[0])

        elasticsearch.helpers.bulk(self.es, gen())
        self.es.indices.refresh(index=self.idx_name)

    def predict(self, query: str) -> Optional[models.Prediction]:
        res = self.es.search(
            index=self.idx_name,
            body=dict(query=dict(match=dict(context=query))),
        )

        if not res["hits"]["total"]["value"]:
            return

        parts = [
            (
                hit["_source"]["label_idx"],
                hit["_source"]["label_str"],
                hit["_score"],
            )
            for hit in res["hits"]["hits"][:30]
        ]

        label_idxs, label_strs, metrics = zip(*parts)

        pred = models.Prediction(
            sample=query,
            label_idxs=label_idxs,
            label_strs=label_strs,
            metrics=metrics,
        )

        return pred


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        help="path the dataset",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--create-index",
        help="whether to create a new index",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--index",
        help="name of the elasticsearch index",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--queries",
        help="path to query samples",
        required=True,
    )
    parser.add_argument(
        "--n-batches",
        help="only run n batches",
        type=int,
        default=None,
    )

    return parser.parse_args()


CONFIG = {
    "sampler": "random",
    "sampler_kwargs": {},
    "train_dataset": "simple",
    "train_dataset_kwargs": {"dataset_mode": "query not masked"},
    "dataset_mode": "query not masked",
    "train_loader": {"batch_size": 1, "shuffle": False},
    "valid_dataset": "simple",
    "valid_dataset_kwargs": {"dataset_mode": "query not masked"},
    "valid_loader": {"batch_size": 1, "shuffle": False},
}


if __name__ == "__main__":
    args = parse_args()
    n_batches = args.n_batches

    dataset_path = common.path(args.dataset, is_dir=True)
    queries_file = common.path(args.queries, is_file=True)

    datamodule = predict.load_data(config=CONFIG, dataset_path=dataset_path)
    queries = predict.load_queries(queries_file, batch_size=1, shuffle=False)

    model = Elastic(meta=datamodule.meta, idx_name=args.index)
    if args.create_index:
        model.index(datamodule.train_dataloader())

    if n_batches:
        log.warn(f"only selecting {n_batches} batches")

    predictions, count = [], 0

    gen = islice(queries, n_batches)
    for batch in tqdm(gen, total=n_batches or len(queries)):
        samples, _ = batch
        assert len(samples) == 1
        prediction = model.predict(samples[0])
        if prediction:
            predictions.append(prediction)
        else:
            count += 1

    predictions.sort(key=lambda p: p.metrics[0], reverse=True)
    out_dir = common.path(dataset_path.parent / "baseline", create=True)

    predict.write_ranks(
        out_dir=out_dir,
        name="elasticsearch",
        predictions=predictions,
        reverse=True,
    )

    msg = f"{count} queries had no results"
    log.info(msg)
    print("finished", msg)
