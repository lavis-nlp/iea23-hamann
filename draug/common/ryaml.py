# -*- coding: utf-8 -*-

import draug
from ktz.filesystem import path

import yaml

from functools import partial
from collections import defaultdict

from collections.abc import Iterable


def join(d1, d2):
    for k, v in d2.items():
        if k in d1 and v is None:
            continue

        if k not in d1 or type(v) is not dict:
            d1[k] = v

        else:
            join(d1[k] or {}, d2[k])


def dic_from_kwargs(**kwargs):
    sep = "__"
    dic = defaultdict(dict)

    for k, v in kwargs.items():
        if sep in k:
            # only two levels deep
            k_head, k_tail = k.split(sep)
            dic[k_head][k_tail] = v
        else:
            dic[k] = v

    return dic


def load(configs: Iterable[str], **kwargs) -> dict:
    """

    Load and join configurations from yaml and kwargs

    """

    if not configs and not kwargs:
        raise draug.DraugError("no configuration provided")

    as_path = partial(path, exists=True, message="loading {path_abbrv}")

    # first join all yaml configs into one dictionary;
    # later dictionaries overwrite earlier ones
    result = {}
    for in_file in map(as_path, configs):
        with in_file.open(mode="r") as fd:
            new = yaml.load(fd, Loader=yaml.FullLoader)
            join(result, new)

    # then join all kwargs;
    # this is practically the reverse of what
    # print_click_arguments does
    dic = dic_from_kwargs(**kwargs)
    join(result, dic)

    return result
