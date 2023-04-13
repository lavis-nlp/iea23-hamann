# -*- coding: utf-8 -*-

from ktz.collections import buckets
from ktz.collections import unbucket
from ktz.filesystem import path as kpath

from draug.homag.text import Match
from draug.homag.text import Nomatch
from draug.homag.text import Filtered

from draug.homag.graph import EID
from draug.homag.graph import Graph
from draug.homag.graph import Entity

from draug.lib import cistem

import yaml
import spacy
from deepca import dumpr
from tqdm import tqdm as _tqdm

import re
import logging
import pathlib
import statistics
import contextlib

from datetime import datetime
from functools import cache
from functools import partial
from itertools import count
from itertools import takewhile
from collections import defaultdict
from collections import Counter
from dataclasses import asdict
from dataclasses import replace
from dataclasses import dataclass

from typing import Any
from typing import Union
from typing import Optional
from collections.abc import Callable
from collections.abc import Generator
from collections.abc import Collection


log = logging.getLogger(__name__)
tqdm = partial(_tqdm, ncols=80, disable=False)
_tqdm_pos = count()


def split(col: Collection[Any], fn: Callable[[Any], bool]):
    # if true -> first container, second container otherwise
    a, b = [], []
    for x in col:
        a.append(x) if fn(x) else b.append(x)
    return a, b


def spanify(lis):
    # [1,2,3,4,6,8,9] -> [(1, 3), (6, 7), (8, 10)]
    # [1, 3, 5] -> [(1,2), (3,4), (5,6)]
    # pretty c-like... find a more _pythonic_ solution?
    i = 0
    while i < len(lis):
        j = i

        while j + 1 < len(lis) and lis[j + 1] - lis[j] == 1:
            j += 1

        yield lis[i], lis[j] + 1
        i = j + 1


# -- SPACY PIPELINE


# all this global state fiddling makes me a sad panda :(
spacy.tokens.Token.set_extension("stem", getter=lambda t: cistem.stem(t.text.lower()))


class HomagSentencizer:

    # the spacy sentencizer with config=dict(punct_chars=['\n']) was
    # _sometimes_ not working... and I don't now why

    def __call__(self, doc):
        for i in range(len(doc)):
            doc[i].sent_start = False

        doc[0].sent_start = True
        for i in (i for i, token in enumerate(doc[:-1]) if "\n" in token.text):
            doc[i + 1].sent_start = True

        return doc

    @staticmethod
    @spacy.language.Language.factory("homag_sentencizer")
    def create(nlp, name: str):
        return HomagSentencizer()


@dataclass(frozen=True)
class Partial:

    id: int
    positions: tuple[int]

    def describe(self, doc, patterns) -> str:
        pattern = " ".join(patterns[self.id])
        mention = " ".join(map(str, (doc[i] for i in self.positions)))
        return f"Partial of [{pattern}]: matched '{mention}'"


# pre-compiled regexes
# e.g. () ,, results in: (^[^,()]+$|(,.*,|\(.*\)))
#
# which applies:
#   T: a b
#   T: a , b , c
#   T: a ( b ) c
#   F: a , b
#   F: a ( b
#   F: a ) b
def _re_compile_between():
    patterns = "()", ",,"
    escaped = (r"\(", r"\)"), (",", ",")

    # '()', ',,' -> '(),'
    disallowed = "".join(set("".join(patterns)))

    # ( ) , , -> (.*)|,.*,
    groups = "|".join(list(map(lambda t: ".*".join(t), escaped)))

    regex = f"^[^{disallowed}]*$|{groups}"
    return re.compile(regex, re.MULTILINE)


RE_BETWEEN = _re_compile_between()
HOMAG_MATCHER = "homag_matcher"


class HomagMatcher:

    # it seems as if stemming will never be a part of spacy
    # https://github.com/explosion/spaCy/issues/327 however, it fails
    # to identify "stumpfen" if "stumpf" is given which is not
    # acceptable -> hence HomagMatcher.match will also regard
    # "externally" stemmed phrases

    MIN_CHARS = 3  # inclusive
    MAX_TOKENS = 5  # exclusive
    MAX_SPACE = 7  # a b matches a c b but not a c d ... b

    patterns: list[tuple[str]]  # all patterns
    patternmap: dict[tuple[str], int]  # pattern to mention

    def expand_phrasemap(
        self,
        phrasemap: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        # rule based pattern creation

        sourcemap = {key: phrases.copy() for key, phrases in phrasemap.items()}

        for key, phrase in [
            (key, phrase.strip().split())
            for key, phrases in sourcemap.items()
            for phrase in phrases
        ]:

            def _add(old, new):
                old, new = (" ".join(x) for x in (old, new))

                # cannot use sets because of spacy :(
                if new not in phrasemap[key]:
                    phrasemap[key].append(new)

            # ---

            if len(phrase) == 2:

                # schlechte X -> X schlecht
                if phrase[0] == "schlechte":
                    _add(phrase, (phrase[1], "schlecht"))

                # X defekt -> defekter X
                if phrase[1] == "defekt":
                    _add(phrase, ("defekter", phrase[0]))

                # X aus -> aus X
                # schaltet aus -> aus schaltet
                if phrase[1] == "aus":
                    _add(phrase, ("aus", phrase[0]))

                # X nicht -> X keine, nicht X
                # kreist nicht -> kreist keine, nicht kreist
                if phrase[1] == "nicht":
                    _add(phrase, (phrase[0], "keine"))
                    _add(phrase, ("nicht", phrase[0]))

            # ---

            if len(phrase) == 3:

                # X nicht Y -> nicht X Y
                # geht nicht auf -> nicht auf geht
                # laeuft nicht sauber -> nicht sauber laeuft
                if phrase[1] == "nicht":
                    _add(phrase, ("nicht", phrase[0], phrase[2]))
                    _add(phrase, ("nicht", phrase[2], phrase[0]))

                # X mehr Y -> X Y
                # nicht mehr sichtbar -> nicht sichtbar
                if phrase[1] == "mehr":
                    _add(phrase, (phrase[0], phrase[2]))

                # X Y nicht -> X nicht Y
                # motor laeuft nicht -> motor nicht laeuft
                if phrase[2] == "nicht":
                    _add(phrase, (phrase[0], "nicht", phrase[1]))

                # X Y ist -> X ist Y
                if phrase[2] == "ist":
                    _add(phrase, (phrase[0], "ist", phrase[1]))

            # ---

            if len(phrase) == 4:

                # kann nicht X werden -> wird nicht X
                if (
                    phrase[0] == "kann"
                    and phrase[1] == "nicht"
                    and phrase[3] == "werden"
                ):
                    _add(phrase, ("wird", "nicht", phrase[2]))

                # laesst sich nicht X -> sich nicht X laesst
                # kann nicht X werden -> wird nicht X
                if (
                    phrase[0] == "laesst"
                    and phrase[1] == "sich"
                    and phrase[2] == "nicht"
                ):
                    _add(phrase, ("sich", "nicht", phrase[3], "laesst"))

        return phrasemap

    def __init__(self, phrasemap: dict[str, list[str]], nlp, n_process: int):
        def _count_phrases():
            return sum(len(phrases) for phrases in phrasemap.values())

        log.info(f"init matcher from {_count_phrases()} phrases")
        self.phrasemap = self.expand_phrasemap(phrasemap)
        log.info(f"expanded to {_count_phrases()} phrases")

        # see match() for intendet purpose
        # while inserting: items must be unique
        self.patterns: set[tuple[str]] = set()
        self.patternmap = defaultdict(set)

        raw = [
            (phrase, key) for key, phrases in phrasemap.items() for phrase in phrases
        ]

        # create key -> doc mapping for later matcher.add
        for doc, key in nlp.pipe(raw, as_tuples=True, n_process=n_process):

            stems = tuple(token._.stem for token in doc)
            lemmata = tuple(token.lemma_ for token in doc)

            for tokens in (stems, lemmata):
                self.patterns.add(tokens)
                self.patternmap[tokens].add(key)

        # when using: items must be indexable
        self.patterns: list[tuple[str]] = list(self.patterns)
        log.info(f"created {len(self.patternmap)} patterns")

    def _match_retain(self, doc, pos: int, part: Partial) -> bool:
        if pos - part.positions[-1] >= HomagMatcher.MAX_SPACE:
            return False

        if len(part.positions) > 1:
            lower, upper = part.positions[-2:]

            # remove partial matches that exceeded the maximum token span
            if upper - lower >= HomagMatcher.MAX_SPACE:
                return False

            # remove partial matches that have undesired content in between
            between = str(doc[lower + 1 : upper])
            matches = RE_BETWEEN.search(between)

            if "\n" in between or not matches:
                return False

        return True

    def _match_group(self, doc, matches: set[Partial]):
        groups = {}
        for part in matches:
            keys = self.patternmap[self.patterns[part.id]]

            for key in keys:

                # create unique identifier
                posrep = ".".join(map(str, part.positions))
                name = f"match:{key}:{posrep}"

                spans = []
                for rg in spanify(part.positions):
                    span = spacy.tokens.Span(doc, *rg, label=key)
                    spans.append(span)

                group = spacy.tokens.SpanGroup(doc=doc, name=name, spans=spans)

                # eliminate duplicate matches
                groups[name] = group

        return groups

    def _filter_partials(
        self,
        doc,
        pos: int,
        partials: dict[str, list[Partial]],
        matches: set[Partial],
    ):
        new_part = defaultdict(list)

        # get all matching base patterns and partial matches
        for old_part in partials:

            # add current position to the position aggregator
            positions = old_part.positions + (pos,)

            # create new partial match (updated positions)
            part = replace(old_part, positions=positions)

            # look up original pattern
            pattern = self.patterns[part.id]

            if not self._match_retain(doc=doc, pos=pos, part=part):
                continue

            if len(pattern) == len(part.positions):
                matches.add(part)
                continue

            # ---
            # debug eid=38
            # if "38" in self.patternmap[self.patterns[part.id]]:
            #     print(part)
            #     breakpoint()
            # ---

            # advance matched partial match by the next token
            new_part[pattern[len(positions)]].append(part)

        return new_part

    def match(self, doc) -> list[spacy.tokens.Span]:
        # Track matched patterns in dictionaries:
        # Each key is a possible match which is compared to
        # the token at hand. The values are partial match
        # tuples: (<ID>, <MATCHED>) where <MATCHED> tracks
        # the matched tokens.

        # Should be a near O(n*m) solution, mostly independent of
        # pattern count, where n is the number of texts and m the
        # number of tokens.

        def by_first_token(idx, pattern):
            return pattern[0], Partial(id=idx, positions=())

        # keep track of base patterns
        # patterns: [('zu', 'spaet'), ... ]
        # pat_base['zu'] -> [(0, ()), ...]
        pat_base = defaultdict(list, buckets(self.patterns, by_first_token))

        # keep track of partial matches (same structure as pat_base)
        pat_part = defaultdict(list)

        # aggregate matched patterns
        matches: set[Partial] = set()

        # get position lemma, and stem for each token in the document
        gen = ((pos, token._.stem, token.lemma_) for pos, token in enumerate(doc))
        # look at lemmas and stems independently
        gen = ((pos, token) for pos, stem, lemma in gen for token in (stem, lemma))

        # look at every token and aggregate matches (as it is opaque
        # to the caller we don't need to differentiate between lemma
        # and stem matches as long as the mentions are unique in the end)
        for pos, token in gen:

            # MATCH
            # consume all partial and base patterns and either add them to
            # the matches or retain them as partial matches
            joined = pat_base[token] + pat_part[token]
            new_part = self._filter_partials(
                doc,
                pos=pos,
                partials=joined,
                matches=matches,
            )

            # FILTER AND RESCUE
            # former partials are filtered and if they are
            # allowed to stay moved to the new partial matches
            for key, part in unbucket(pat_part):
                retain = self._match_retain(doc=doc, pos=pos, part=part)

                if not retain:
                    continue

                new_part[key].append(part)

            pat_part = new_part

        # create span groups
        groups = self._match_group(doc=doc, matches=matches)
        return groups

    def __call__(self, doc):
        for name, group in self.match(doc).items():
            doc.spans[name] = group

        return doc

    @staticmethod
    @spacy.language.Language.factory(HOMAG_MATCHER)
    def create(nlp, name: str, phrasemap: dict[str, list[str]], n_process: int):
        return HomagMatcher(phrasemap=phrasemap, nlp=nlp, n_process=n_process)


class HomagFilter:

    min_len: int

    def __init__(self, nlp, min_len: int):
        self.nlp = nlp
        self.min_len = min_len

    def __call__(self, doc):
        # cannot figure out how to discard documents from the pipeline
        # -> create empy documents that are filtered later :(

        filtered = filter(lambda s: len(s.text) >= self.min_len, doc.sents)
        mapped = list(map(lambda s: s.as_doc(), filtered))

        Doc = spacy.tokens.Doc
        new_doc = Doc.from_docs(mapped) if len(mapped) else Doc(self.nlp.vocab)

        # monkey patch (introduced with spacy 3.2 - it worked before)
        # the ._context property is set when a pipeline is executed
        # with as_tuples=True. To retain the context, it needs to be
        # set for the new document explicitly.
        # new_doc._context = doc._context

        return new_doc

    @staticmethod
    @spacy.language.Language.factory("homag_filter")
    def create(nlp, name: str, min_len: int):
        return HomagFilter(nlp=nlp, min_len=min_len)


# -- RESULT PROCESSING


class Handler(contextlib.AbstractContextManager):
    def __enter__(self) -> "Handler":
        raise NotImplementedError()

    def __exit__(self, *args):
        pass

    def handle_match(self, match: Match):
        raise NotImplementedError()

    def handle_filtered(self, match: Match, phrase: str):
        raise NotImplementedError()

    def handle_nomatch(self, nomatch: Nomatch):
        raise NotImplementedError()

    def handle_phrases(self, phrases: dict, phrasemap: dict):
        raise NotImplementedError()

    def handle_config(self, config: dict):
        raise NotImplementedError()


class SimpleHandler(Handler):
    """
    Simply retain all matches etc. internally
    """

    matches: list[Match]
    filtered: list[Filtered]
    nomatches: list[Nomatch]

    config: dict
    phrases: dict
    phrasemap: dict

    def __enter__(self) -> "Handler":
        self.matches = []
        self.filtered = []
        self.nomatches = []
        return self

    def handle_match(self, match: Match):
        self.matches.append(match)

    def handle_filtered(self, filtered: Filtered):
        self.filtered.append(filtered)

    def handle_nomatch(self, nomatch: Nomatch):
        self.nomatches.append(nomatch)

    def handle_phrases(self, phrases: dict, phrasemap: dict):
        self.phrases = phrases
        self.phrasemap = phrasemap

    def handle_config(self, config: dict):
        self.config = config


N_MATCH = "match"
N_FILTERED = "filtered"
N_NOMATCH = "nomatch"
N_SKIPPED = "skipped"


class ResultWriter(Handler):
    """
    This class writes all pipeline results to disk
    """

    graph: Graph
    out_dir: pathlib.Path

    # bookkeeping
    skipped: dict[str, int]

    def __init__(self, graph: Graph, out_dir: Union[str, pathlib.Path]):
        super().__init__()

        self.graph = graph
        self.out_dir = kpath(out_dir, create=True)

    def open(self, fname: str):
        log.info(f"opening {self.out_dir / fname} for writing")

        return self._stack.enter_context(
            (self.out_dir / fname).open(mode="w"),
        )

    def _init_tq(self, name: str):
        return tqdm(
            desc=f"{name:>15s}",
            unit=" contexts",
            position=next(_tqdm_pos),
            total=None,
        )

    def __enter__(self):
        log.info("entering result writer context")
        self._stack = contextlib.ExitStack().__enter__()

        self._cnt = Counter()
        self._fd, self._ctx, self._tq = {}, {}, {}
        for kind in (N_MATCH, N_FILTERED, N_NOMATCH):
            self._fd[kind] = self.open(kind + ".txt")
            self._ctx[kind]: set[str] = set()
            self._tq[kind] = self._init_tq(name=kind)

        self._agg_match: set[Match] = set()
        self._agg_filtered: dict[str, Filtered] = {}

        self._tq[N_SKIPPED] = self._init_tq(name=N_SKIPPED)

        return self

    def __exit__(self, *args):
        self._write_filtered()
        self._write_stats()
        log.info("leaving result writer context")
        return self._stack.__exit__(*args)

    def _write_stats(self):
        log.info("accumulate and write stats")

        stats = {}
        for key, agg in self._ctx.items():
            lens = [len(context) for context in agg] or [-1]
            stdev = round(statistics.stdev(lens)) if len(lens) > 1 else -1

            stats[key] = dict(
                contexts=len(agg),
                skipped=self._cnt[key + N_SKIPPED],
                maxlen=max(lens),
                minlen=min(lens),
                mean=round(statistics.mean(lens)),
                stdev=stdev,
            )

        yaml.dump(stats, self.open("stats.yml"))

    def _write(self, kind: str, line: str, context: str):
        self._fd[kind].write(line + "\n")
        self._ctx[kind].add(context)
        self._tq[kind].update()
        self._cnt[kind] += 1

    # --- handler

    def handle_match(self, match: Match):
        if match in self._agg_match:
            self._cnt[N_MATCH + N_SKIPPED] += 1
            self._tq[N_SKIPPED].update()
            return

        self._write(kind=N_MATCH, line=match.as_str, context=match.context)
        self._agg_match.add(match)

    def handle_nomatch(self, nomatch: Nomatch):
        if nomatch.context in self._ctx[N_NOMATCH]:
            self._cnt[N_NOMATCH + N_SKIPPED] += 1
            self._tq[N_SKIPPED].update()
            return

        self._write(kind=N_NOMATCH, line=nomatch.as_str, context=nomatch.context)

    def handle_filtered(self, filtered: Filtered):
        self._write(kind=N_FILTERED, line=filtered.as_str, context=filtered.context)
        self._agg_filtered[filtered.context] = filtered

    def _write_filtered(self):
        log.info("determine filtered contexts to be written as nomatches")
        assert set(self._agg_filtered) == self._ctx[N_FILTERED]

        ctx_filtered = self._ctx[N_FILTERED]
        ctx_other = self._ctx[N_MATCH] | self._ctx[N_NOMATCH]

        for context in ctx_filtered - ctx_other:
            filtered = self._agg_filtered[context]
            nomatch = Nomatch(identifier=filtered.ticket, context=filtered.context)
            self._write(kind=N_NOMATCH, line=nomatch.as_str, context=nomatch.context)

        self._cnt[N_FILTERED + N_SKIPPED] += len(ctx_filtered & ctx_other)

    def handle_phrases(
        self,
        phrases: dict[int, dict[str, int]],
        phrasemap: dict[str, list[str]],
    ):

        n_phrases = sum(map(len, phrases.values()))
        log.info(f"encountered {n_phrases} mentions from {len(phrases)} patterns")

        dic = {
            eid: {
                "entity": self.graph.get_entity(eid=eid).name,
                "phrases": phrasemap[str(eid)],
                "mentions": mentions,
            }
            for eid, mentions in phrases.items()
        }

        log.info("writing phrases")
        yaml.dump(dic, self.open("phrases.yml"))

    def handle_config(self, config):
        yaml.dump(config, self.open("config.yml"))


# --


def _create_mention_idxs(group, offset=0):
    flat = [
        idx
        for sub in [(span.start - offset, span.end - offset) for span in group]
        for idx in sub
    ]

    # eliminate unnecessary pairs for consecutive tokens
    # .e.g (0, 1, 2, 3, 5, 6, 7, 8) -> (0, 3, 5, 8)

    # (0, 1, 2, 3, 5, 6, 7, 8) -> [(3, 5)]
    filtered = filter(lambda t: t[1] - t[0] - 1, zip(flat[1::2], flat[2::2]))

    # [0] + [(3, 5)] [9] -> [0, 3, 5, 8]
    joined = [flat[0]] + [x for sub in filtered for x in sub] + [flat[-1]]

    assert len(joined) % 2 == 0
    return tuple(joined)


def _create_match(phrases, identifier, sentence, group) -> Match:

    # obtain mentions from span groups
    mention = " ".join(span.text for span in group)
    mention_idxs = _create_mention_idxs(group, offset=sentence.start)

    # update phrase dict
    eid = int(group[0].label_)
    phrases[eid][mention] += 1

    # obtain tokens
    tokens = [t.text for t in sentence]

    # check if they can be addressed with the mention_idxs
    _groups = list(zip(mention_idxs[::2], mention_idxs[1::2]))
    _s_tokens = sum(len(tokens[a:b]) for a, b in _groups)
    _s_ranges = sum(b - a for a, b in _groups)

    try:
        assert _s_tokens == _s_ranges
    except AssertionError:
        breakpoint()

    # to string
    context = " ".join(tokens).strip()

    match = Match(
        identifier=identifier,
        eid=eid,
        mention=mention,
        mention_idxs=mention_idxs,
        context=context,
    )

    return match


def _create_nomatch(identifier, sentence):
    context = " ".join(t.text for t in sentence).strip()
    nomatch = Nomatch(
        identifier=identifier,
        context=context,
    )

    return nomatch


@cache
def _filter_match_regify(s: str):
    # "sauger kein" -> (^| )sauger.*kein\w*( |$)

    rep = s.replace("*", r"\w*")  # expand '*'
    sub = re.sub(r"\s+", ".*", rep)
    pat = rf"(^| ){sub}( |$)"

    return re.compile(pat)


def _filter_match(entities, match) -> Optional[str]:
    undesired = entities[match.eid].undesired_phrases or []
    mention = match.mention.lower()

    for phrase in undesired:
        regex = _filter_match_regify(phrase)
        matches = regex.match(mention)

        if matches:
            return phrase


def _match(nlp, entities, yielder, ctx: Handler, procs: int = 1):
    docs = nlp.pipe(yielder, as_tuples=True, n_process=procs)

    # filter empty docs (see HomagFilter)
    phrases = defaultdict(Counter)

    # partition sentences into (1) matches, and (2) nomatches
    for doc, meta in filter(lambda t: bool(t[0]), docs):
        assert meta is not None, "no meta information provided"
        identifier = meta["identifier"]

        # create sentence -> [group, ...] mapping
        matched = buckets(
            col=doc.spans.values(),
            key=lambda _, group: (group[0].sent, group),
        )

        # (1) matches
        # ----------------------------------------

        for sentence, group in unbucket(matched):

            match = _create_match(
                phrases=phrases,
                identifier=identifier,
                sentence=sentence,
                group=group,
            )

            # if an undesired phrase matches the mention
            filtered = _filter_match(entities=entities, match=match)
            if filtered:
                ctx.handle_filtered(
                    filtered=Filtered(
                        phrase=filtered,
                        **asdict(match),
                    )
                )
                continue

            # the match was not filtered and can be handled
            ctx.handle_match(match=match)

        # (2) nomatches
        # ----------------------------------------

        for sentence in set(doc.sents) - set(matched.keys()):

            nomatch = _create_nomatch(
                identifier=identifier,
                sentence=sentence,
            )

            ctx.handle_nomatch(nomatch=nomatch)

    return {k: dict(v) for k, v in phrases.items()}


def yield_tickets(path: Union[str, pathlib.Path], n: int = None):
    log.info(f"yielding tickets from {path}")
    with dumpr.BatchReader(str(path)) as reader:

        n = n or reader.count
        gen = (raw for _, raw in takewhile(lambda t: t[0] < n, enumerate(reader)))

        # cased.lang version
        for raw in tqdm(
            gen,
            total=n,
            unit="docs",
            desc="reader",
            position=next(_tqdm_pos),
        ):

            content = raw.content.strip()
            yield content, {"identifier": f'{raw.meta["identifier"]}'}

    log.info(f"finished yielding {n} tickets")


SPACY_MODEL = "de_core_news_lg"


def get_nlp():
    nlp = spacy.load(
        SPACY_MODEL,
        disable=[
            "tok2vec",
            "tagger",
            "ner",
            "parser",
            "attribute_ruler",
        ],
    )

    # the input text is already tokenized
    # -> only split whitespace
    nlp.tokenizer = spacy.tokenizer.Tokenizer(
        nlp.vocab,
        token_match=re.compile(r"\S+").match,
    )

    # only works if "parser" is deactivated
    # otherwise it does _something strange_ ...
    # currently uses pre-split sentences
    nlp.add_pipe("homag_sentencizer")

    # it needs to come before the matcher
    nlp.add_pipe("homag_filter", config=dict(min_len=30))

    return nlp


def match(
    nlp,
    entities: dict[EID, Entity],
    yielder: Generator[tuple[str, dict]],  # see yield_tickets
    handler: Handler,
    procs: int,
    config: Optional[dict] = None,
) -> None:

    if config is None:
        config = {}

    # multiple constraints apply to the phrasemap due to spacy:
    #  (1) no integer keys
    #  (2) no sets/tuples (not json serializable)
    phrasemap = {str(eid): [ent.name] for eid, ent in entities.items()}

    try:

        # but it must not apply to the mention phrases
        # (they would all be filtered out when addded)
        with nlp.select_pipes(disable=["homag_filter"]):
            nlp.add_pipe(
                HOMAG_MATCHER,
                config=dict(
                    phrasemap=phrasemap,
                    n_process=procs,
                ),
            )

        # configure pipeline and run keyphrase matcher
        with handler as ctx:

            # gotta match 'em all
            phrases = _match(
                nlp=nlp,
                entities=entities,
                yielder=yielder,
                ctx=ctx,
                procs=procs,
            )

            name, matcher = nlp.pipeline[4]
            assert name == HOMAG_MATCHER

            config |= dict(
                created=datetime.now(),
                spacy_model=SPACY_MODEL,
                spacy_pipeline=list(nlp.pipe_names),
            )

            ctx.handle_phrases(phrases, matcher.phrasemap)
            ctx.handle_config(config)

    finally:
        nlp.remove_pipe(HOMAG_MATCHER)
