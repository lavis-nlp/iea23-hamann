# -*- coding: utf-8 -*-

"""

Working with Klaus' taxonomy files

"""

import draug
from draug.homag import open_kwargs

import csv
import enum
import logging
from itertools import count
from dataclasses import field
from dataclasses import dataclass

import yaml
import numpy as np
import networkx as nx

from ktz.collections import buckets
from ktz.filesystem import path as kpath

from typing import Any
from typing import Union
from typing import Iterable
from typing import Optional
from typing import Iterator
from pathlib import Path


log = logging.getLogger(__name__)


# type hints
NID, EID, RID = int, int, int
Triple = tuple[int]

# constant keys for meta attributes
M_NAME = "name"

# constant keys for node properties
N_ENT = "entities"


@dataclass(eq=True)
class Entity:

    name: str

    # mutable parts

    undesired_phrases: tuple[str] = field(
        default_factory=list,
        compare=False,
    )

    undesired_pattern: tuple[str] = field(
        default_factory=list,
        compare=False,
    )

    # set by the graph after adding

    @property
    def nid(self) -> int:
        return self._nid

    @property
    def eid(self) -> int:
        return self._eid


class Graph:
    """

    Graph of taxonomy nodes

    Reminder: dict are ordered by insertion (node eid -> name):
    as such list(dic.values())[0] will always return
    the first inserted pairs' value

    data model summary
    ------------------

    * each node:
      * has a single, invariant node id (nid)
      * has a set of entities assigned
      * has computed name
    * each entity:
      * may be ambiguous
      * has a single, invariant entity id (eid)
      * there's a eid -> nid mapping
    * relations:
      * have different ids (defined in Graph.RELATIONS)
      * may have graph-wide properties (such as reflexivity)

    naming conventions
    ------------------

    * node: n (for node), h (for head), or t (for tail)
    * entity: e (for entity)
    * relation (type): r
    * ids:
      * node id: nid
      * entity id: eid
      * parent id: pid

    """

    class RELATIONS(enum.Enum):
        parent = 0
        synonym = 1

    nxg: nx.MultiDiGraph
    meta: dict[str, Any]
    path: Optional[Path]

    def __init__(
        self,
        meta: dict[str, Any],
        nodes: Optional[Iterable[tuple[NID, dict]]] = None,
        triples: Optional[Iterable[Triple]] = None,
        path: Optional[Path] = None,
    ):
        assert M_NAME in meta, f"missing meta attribute: {M_NAME}"

        self.nxg = nx.MultiDiGraph()
        self.meta = meta
        self.path = path

        self._eid2nid: dict[EID, NID] = {}

        # populate

        if nodes:
            for nid, kwargs in nodes:
                self.nxg.add_node(nid, **kwargs)
                for eid, ent in kwargs[N_ENT].items():
                    self._eid2nid[eid] = nid
                    ent._eid = eid
                    ent._nid = nid

        if triples:
            rids = {m.value for m in Graph.RELATIONS}
            assert all(r in rids for _, _, r in triples)

            nids = set(self.nxg.nodes)
            assert all(n in nids for h, t, _ in triples for n in (h, t))

            self.nxg.add_edges_from(triples)

    # --

    @property
    def name(self):
        return self.meta[M_NAME]

    @property
    def nids(self) -> set[NID]:
        return set(self.nxg.nodes)

    @property
    def eids(self) -> set[EID]:
        return set(self._eid2nid)

    @property
    def rels(self) -> set[RID]:
        return {rid for _, _, rid in self.nxg.edges}

    @property
    def names(self) -> dict[str, tuple[NID]]:
        flat = ((name, nid) for nid in self.nxg.nodes for name in self.node_names(nid))
        return buckets(flat)

    @property
    def roots(self) -> tuple[NID]:
        """
        Find all nodes that do not have a parent
        """
        return tuple(n for n in self.nids if self.get_parent(nid=n) is None)

    @property
    def leaves(self) -> tuple[NID]:
        """
        Find all nodes that do not have children
        """
        return tuple(n for n in self.nids if not self.get_children(nid=n))

    # private

    def _entity_nodeprop(
        self,
        eid: Optional[EID] = None,
        nid: Optional[NID] = None,
    ) -> dict[EID, Entity]:
        assert eid is not None or nid is not None

        nid = nid if nid is not None else self._get_entity_nid(eid=eid)
        return self.nxg.nodes[nid][N_ENT]

    def _get_entity_nid(self, eid: EID) -> NID:
        return self._eid2nid[eid]

    # ---

    @property
    def description(self) -> str:
        s = (
            f"draug homag taxonomy\n"
            f"  name: {self.name}\n"
            f"  nodes: {self.nxg.number_of_nodes()}\n"
            f"  edges: {self.nxg.number_of_edges()}"
            f" ({len(self.rels)} types)\n"
        )

        # statistical values

        try:
            degrees = np.array(list(d for _, d in self.nxg.degree()))
            s += (
                f"  degree:\n"
                f"    mean {np.mean(degrees):.2f}\n"
                f"    median {int(np.median(degrees)):d}\n"
            )
        except IndexError:
            s += "  cannot measure degree\n"

        return s

    def __str__(self) -> str:
        return f"draug.graph: [{self.name}] ({self.nxg.number_of_nodes()} nodes)"

    # --- accessors

    def cluster(self, nid: NID, rid: RID) -> set[NID]:  # that set contains nids
        q, agg = {nid}, set()

        while len(q):
            nid = q.pop()
            nn = self.neighbours(nid=nid, rid=rid)

            agg.add(nid)
            q |= nn - agg

        return agg

    def neighbours(self, nid: NID, rid: RID) -> set[NID]:
        """
        Get all neighbours of a specific relation type.
        """
        return set(i for i, view in self.nxg[nid].items() if rid in set(view.keys()))

    def get_children(self, nid: NID) -> tuple[NID]:
        """
        Get the child entities of a specified entity. An entity might have
        zero, one, or multiple children.

        TODO: Currently O(num_nodes) -> cannot efficiently use
        self.nxg.reverse() because the graph is mutable

        """
        return tuple(n for n in self.nids if self.get_parent(n) == nid)

    # --- modifier

    def _gen_eids(self) -> Iterator[int]:
        return count(max(list(self.eids or {-1})) + 1)

    # -- nodes

    def node_repr(self, nid: NID) -> dict:
        return dict(self.nxg.nodes[nid]) | {
            "nid": nid,
            "name": self.node_name(nid=nid),
        }

    def node_name(self, nid: NID) -> str:
        names = self.node_names(nid=nid)
        if not names:
            return "<no entities>"
        else:
            return f"{names[0]} (+{len(names) - 1})"

    def node_names(self, nid: NID) -> tuple[str]:
        return tuple(ent.name for ent in self.get_entities(nid=nid).values())

    def add_node(
        self,
        nid: Optional[NID] = None,
        entities: Optional[Iterable[Entity]] = None,
    ) -> NID:
        """
        Returns the newly assigned node id.
        """

        # get next free node id
        nid = nid or (max(self.nids or [-1]) + 1)
        assert (nid is not None) and (nid not in self.nids)

        # insert into nxg
        kwargs = {N_ENT: {}}
        self.nxg.add_node(nid, **kwargs)

        if "reflexive" in self.meta:
            for rid in self.meta["reflexive"]:
                self.nxg.add_edge(nid, nid, rid)

        # add entities
        for ent in entities:
            self.add_entity(nid=nid, ent=ent)

        log.info(f"adding node [{nid}] {self.node_name(nid=nid)}")
        return nid

    def del_node(self, nid: int) -> list[NID]:
        """
        Delete an entity from the taxonomy.
        Child entities are deleted recursively.
        Node ids are rewritten to avoid gaps.

        Returns count of deleted entities
        """

        def _rec(nid: int):
            log.info(f"deleting node [{nid}] {self.node_name(nid)}")
            deleted = []

            children = self.get_children(nid)
            for child in children:
                deleted += _rec(child)

            # disconnect from old parent
            old_parent = self.get_parent(nid)
            if old_parent is not None:
                self.nxg.remove_edge(nid, old_parent)

            for eid in self.get_entities(nid=nid).keys():
                self.del_entity(eid=eid)

            self.nxg.remove_node(nid)
            return deleted + [nid]

        return _rec(nid)

    # -- entities

    def get_entity(self, eid: EID) -> Entity:
        nid = self._get_entity_nid(eid=eid)
        return self.nxg.nodes[nid][N_ENT][eid]

    def get_entities(self, nid: NID) -> dict[EID, Entity]:
        return self._entity_nodeprop(nid=nid).copy()

    def add_entity(self, nid: NID, ent: Entity) -> EID:
        edic = self._entity_nodeprop(nid=nid)
        ents = tuple(edic.values())

        names = {ent.name: ent for ent in ents}
        if ent.name in names:
            dup = names[ent.name]
            log.info(f'entity "{dup.name}" already exists (eid={dup.eid})')
            eid = dup.eid
        else:
            eid = next(self._gen_eids())
            log.info(f"add [{eid=}] '{ent.name}' to [{nid=}]")

            assert eid not in self.eids
            assert ent not in edic.values()

        self._eid2nid[eid] = nid
        edic[eid] = ent

        ent._eid = eid
        ent._nid = nid

        return eid

    def del_entity(self, eid: EID) -> None:
        assert eid in self._eid2nid, f"'{eid=}' does not exist"
        log.info(f"deleting entity [{eid}] {self.get_entity(eid=eid).name}")

        del self._entity_nodeprop(eid=eid)[eid]
        del self._eid2nid[eid]

    def set_entity(self, eid: EID, ent: Entity) -> None:
        assert eid in self._eid2nid, f"'{eid=}' does not exist"

        nid = self.get_entity(eid=eid).nid
        self.del_entity(eid=eid)
        self.add_entity(nid=nid, ent=Entity)

    # -- parents

    def get_parent(self, nid: NID) -> Optional[NID]:
        """
        Get the parent entity of a specified entity.
        Root entities do not have a parent.
        """
        rid = Graph.RELATIONS.parent.value
        parents = list(t for _, t, r in self.nxg.edges(nid, keys=True) if r == rid)

        assert len(parents) <= 1
        return parents[0] if parents else None

    def set_parent(self, nid: NID, pid: NID) -> Optional[NID]:
        rid = Graph.RELATIONS.parent.value

        # remove old parent relation if any
        old_pid = None
        if self.get_parent(nid=nid) is not None:
            old_pid = self.del_parent(nid=nid)

        # connect to new parent
        self.nxg.add_edge(nid, pid, rid)
        return old_pid

    def del_parent(self, nid: NID) -> Optional[NID]:
        rid = Graph.RELATIONS.parent.value

        old_pid = self.get_parent(nid=nid)
        assert old_pid is not None, f"node {nid=} has no parent"

        self.nxg.remove_edge(nid, old_pid, key=rid)
        return old_pid

    # --- persistence

    def __repr__(self):
        meta = repr(self.meta)
        nodes = tuple((nid, self.nxg.nodes[nid]) for nid in self.nxg.nodes)
        triples = tuple(self.nxg.edges)
        return f"Graph(meta={meta}, nodes={nodes}, triples={triples})"

    def to_dir(self, path: Path):
        out_dir = kpath(path, create=True)

        log.info(f"writing graph to {out_dir}")
        kwargs = dict(mode="w") | open_kwargs

        # meta
        with (out_dir / "meta.yml").open(**kwargs) as fd:
            yaml.dump(self.meta, fd)

        # edgelist (nx.write_edgelist does not write keys)
        with (out_dir / "edges.txt").open(**kwargs) as fd:
            lines = [" ".join(map(str, triple)) for triple in self.nxg.edges]
            fd.writelines("\n".join(lines))

        # nodelist
        with (out_dir / "nodes.txt").open(**kwargs) as fd:
            lines = [f"{nid} {repr(self.nxg.nodes[nid])}" for nid in self.nxg.nodes]
            fd.writelines("\n".join(lines))

    @classmethod
    def from_dir(Cls, path: Union[str, Path]):
        in_dir = kpath(path, is_dir=True)

        log.info(f"loading graph from {in_dir}")
        kwargs = dict(mode="r") | open_kwargs

        with (in_dir / "meta.yml").open(**kwargs) as fd:
            meta = yaml.load(fd, Loader=yaml.FullLoader)

        with (in_dir / "nodes.txt").open(**kwargs) as fd:
            lines = (line.split(" ", maxsplit=1) for line in fd)
            nodes = tuple((int(nid), eval(kwargs)) for nid, kwargs in lines)

        with (in_dir / "edges.txt").open(**kwargs) as fd:
            triples = tuple(tuple(map(int, line.split())) for line in fd)

        # --

        self = Cls(meta=meta, nodes=nodes, triples=triples, path=in_dir)
        return self


# ----------


def filter_csv(line):
    tid, name, desc, *_ = line
    # if desc == "xxx":
    #     return False
    return True


def normalize_name(name: str):
    assert name
    name = name.lower()

    for tar, sub in (("ä", "ae"), ("ü", "ue"), ("ö", "oe"), ("ß", "ss")):
        name = name.replace(tar, sub)

    return name


def populate_graph_raw(g: Graph, rows):
    """
    do not do any transformations and import the graph as-is
    """
    id2nid = {}

    def get_or_add_node(name, _id=None) -> int:
        if _id and _id in id2nid:
            return id2nid[_id]

        name = normalize_name(name)
        nid = g.add_node(entities=[Entity(name=name)])

        if _id:
            id2nid[_id] = nid

        return nid

    # external:
    # _tid: taxonmy id (ID123412_1231241)
    # _pid: parent id (ID123412_1231241)
    #
    # internal:
    # pid: parent id: int
    # sid: synonym id: int
    # nid: node id: int
    # eid: entity id: int
    #
    for _tid, name, _, syn, _pid, parent in rows:

        nid = get_or_add_node(name=name, _id=_tid)
        pid = get_or_add_node(name=parent, _id=_pid) if _pid else None
        sid = get_or_add_node(name=syn) if syn else None

        if pid:
            g.set_parent(nid=nid, pid=pid)

        if sid:
            rid = Graph.RELATIONS.synonym.value
            g.nxg.add_edge(sid, nid, rid)


def populate_graph(g: Graph, rows):
    idmap = {}

    def get_or_add_node(_tid, name):
        if _tid in idmap:
            return idmap[_tid]

        name = normalize_name(name)
        nid = idmap[_tid] = g.add_node(entities=[Entity(name=name)])
        log.info(f"added node [nid]: {g.node_name(nid=nid)}")

        return nid

    # external:
    # _tid: taxonmy id (ID123412_1231241)
    # _pid: parent id (ID123412_1231241)
    #
    # internal:
    # pid: parent id: int
    # sid: synonym id: int
    # nid: node id: int
    # eid: entity id: int
    #
    for _tid, name, _, syn, _pid, parent in rows:
        nid = get_or_add_node(_tid, name=name)

        if syn:
            g.add_entity(nid=nid, ent=Entity(name=normalize_name(syn)))

        if _pid:
            pid = get_or_add_node(_pid, name=parent)
            g.set_parent(nid=nid, pid=pid)


def save_gml(graph, out):
    gml = graph.nxg.copy()

    for nid in list(gml.nodes):
        name = graph.node_name(nid=nid)
        eids = ", ".join(map(str, graph.get_entities(nid=nid).keys()))
        gml.nodes[nid][N_ENT] = f"{name} ({eids})"

    nx.write_gml(gml, str(out))


def import_csv(name: str, path: Path, raw: bool = False):

    with path.open(mode="r") as fd:
        next(fd)
        rows = list(filter(filter_csv, csv.reader(fd)))

    if raw:
        g = Graph(
            meta=dict(
                name=name,
                relmap={m.value: m.name for m in Graph.RELATIONS},
            ),
        )

        populate_graph_raw(g=g, rows=rows)

    else:
        g = Graph(
            meta=dict(
                name=name,
                reflexive=[Graph.RELATIONS.synonym.value],
                relmap={m.value: m.name for m in Graph.RELATIONS},
            ),
        )

        populate_graph(g=g, rows=rows)

    # ---

    out_dir = kpath(draug.ENV.DIR.HOMAG_GRAPH / name, create=True)
    log.info(f"saving graph to {out_dir}")

    g.to_dir(path=out_dir)
    print(g.description)

    save_gml(graph=g, out=out_dir / "graph.gml")


def import_undesired(graph: Graph, undesired: str):
    with kpath(undesired, is_file=True).open(mode="r") as fd:
        undesired = yaml.load(fd, Loader=yaml.FullLoader)

    for eid, dic in undesired.items():
        ent = graph.get_entity(eid=eid)

        if "entity" in dic:
            assert (
                dic["entity"] == ent.name
            ), f"{eid=}\n  provided: {dic['entity']}\n  graph: {ent.name}"

        if "phrases" in dic:
            ent.undesired_phrases.extend(dic["phrases"])

        if "patterns" in dic:
            ent.undesired_patterns.extend(dic["patterns"])

    graph.to_dir(path=graph.path)
