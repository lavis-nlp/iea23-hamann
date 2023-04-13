# -*- coding: utf-8 -*-

import draug

import pathlib
import logging
import hashlib
from itertools import chain
from collections import defaultdict

import git

from typing import Any
from typing import Union
from typing import Callable
from typing import Optional
from collections.abc import Iterable
from collections.abc import Collection


log = logging.getLogger(__name__)


#
#   FILESYSTEM
#


# deprecated: use ktz.filesystem.path
def path__(
    name: Union[str, pathlib.Path],
    create: bool = False,
    exists: bool = False,
    is_dir: bool = False,
    is_file: bool = False,
    message: str = None,
) -> pathlib.Path:
    # TODO describe message (see kgc.config)
    path = pathlib.Path(name)

    if (exists or is_file or is_dir) and not path.exists():
        raise draug.DraugError(f"{path} does not exist")

    if is_file and not path.is_file():
        raise draug.DraugError(f"{path} exists but is not a file")

    if is_dir and not path.is_dir():
        raise draug.DraugError(f"{path} exists but is not a directory")

    if create:
        path.mkdir(exist_ok=True, parents=True)

    if message:
        path_abbrv = f"{path.parent.name}/{path.name}"
        log.info(message.format(path=path, path_abbrv=path_abbrv))

    return path


# deprecated: use ktz.filesystem.path
def path_rotate__(current: Union[str, pathlib.Path], keep: int = None):
    """

    Rotates a file

    Given a file "foo.tar", rotating it will produce "foo.1.tar".
    If "foo.1.tar" already exists then "foo.1.tar" -> "foo.2.tar".
    And so on. Also works for directories.

    If 'keep' is set to a positive integer, keeps at most
    that much files.

    """
    current = path__(current, message="rotating {path_abbrv}")
    if keep:
        assert keep > 0

    def _new(
        p: pathlib.Path,
        n: int = None,
        suffixes: list[str] = None,
    ):
        name = p.name.split(".")[0]  # .stem returns foo.tar for foo.tar.gz
        return p.parent / "".join([name, "." + str(n)] + suffixes)

    def _rotate(p: pathlib.Path):
        if p.exists():
            old_n, *suffixes = p.suffixes

            n = int(old_n[1:]) + 1
            new = _new(p, n=n, suffixes=suffixes)

            if keep <= n:
                _rotate(new)

            p.rename(new)

    if current.exists():
        new = _new(current, n=1, suffixes=current.suffixes)
        _rotate(new)
        current.rename(new)


def git_hash() -> str:
    repo = git.Repo(search_parent_directories=True)
    # dirty = '-dirty' if repo.is_dirty else ''
    return str(repo.head.object.hexsha)


def args_hash(*args) -> str:
    bytestr = "".join(map(str, args)).encode()
    return hashlib.sha224(bytestr).hexdigest()


#
#   PRIMITIVES
#


def decode_line(encoded: bytes, sep: str) -> list[str]:
    return list(map(str.strip, encoded.decode("unicode_escape").split(sep)))


def encode_line(data: list[str], sep: str) -> bytes:
    assert all(sep not in s for s in data)
    return ((f" {sep} ").join(data)).encode("unicode_escape") + b"\n"


#
#   COLLECTIONS
#

A, B, C, D = Any, Any, Any, Any


def agg(col: Iterable[tuple[A, B]]) -> dict[A, tuple[B]]:
    dic = defaultdict(list)

    for k, v in col:
        dic[k].append(v)

    return {k: tuple(v) for k, v in dic.items()}


def partition(col: Iterable[Any], key: Callable[[Any], Any]) -> dict[Any, Any]:
    dic = defaultdict(list)

    for it in col:
        k = key(it)
        dic[k].append(it)

    return dic


def split(col: Iterable[Any], key: Callable[[Any], bool]) -> tuple[list]:
    a, b = [], []
    for it in col:
        lis = a if key(it) else b
        lis.append(it)

    return a, b


def buckets(
    col: Collection[A],
    fn: Callable[[int, A], tuple[B, C]],
    reductor: Optional[Callable[[list[C]], D]] = None,
) -> Union[dict[B, list[C]], dict[B, D]]:

    dic = defaultdict(list)

    for i, elem in enumerate(col):
        k, v = fn(i, elem)
        dic[k].append(v)

    if reductor:
        dic = {k: reductor(v) for k, v in dic.items()}

    return dict(dic)


def unbucket(buckets: dict[A, list[B]]) -> list[tuple[A, B]]:
    yield from ((key, el) for key, lis in buckets.items() for el in lis)


def flat(col: Collection[Any], depth: int = 2) -> Collection[Any]:
    yield from col if depth == 1 else chain(
        *(flat(lis, depth=depth - 1) for lis in col)
    )
