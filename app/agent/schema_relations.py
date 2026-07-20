"""Deterministic schema relationships derived from FK/PK metadata."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True, order=True)
class JoinEdge:
    left_table: str
    left_column: str
    right_table: str
    right_column: str

    @property
    def column_ids(self) -> frozenset[str]:
        return frozenset({self.left_column, self.right_column})

    def other_table(self, table_id: str) -> str:
        normalized = normalize_identifier(table_id)
        if normalized == self.left_table:
            return self.right_table
        if normalized == self.right_table:
            return self.left_table
        raise ValueError(f"table {table_id!r} is not part of join edge")


@dataclass(frozen=True)
class SchemaGraph:
    adjacency: dict[str, tuple[JoinEdge, ...]]
    ambiguous_adjacency: dict[str, tuple[JoinEdge, ...]]


@dataclass(frozen=True)
class JoinClosureResult:
    status: Literal["success", "unresolved", "ambiguous"]
    table_ids: frozenset[str] = frozenset()
    column_ids: frozenset[str] = frozenset()
    edges: tuple[JoinEdge, ...] = ()


def normalize_identifier(value: Any) -> str:
    return str(value or "").strip().casefold()


def is_valid_join_pair(left: Any, right: Any) -> bool:
    """Return whether two columns satisfy the shared same-name FK/PK rule."""

    left_name = normalize_identifier(_value(left, "name"))
    right_name = normalize_identifier(_value(right, "name"))
    roles = {
        normalize_identifier(_value(left, "role")),
        normalize_identifier(_value(right, "role")),
    }
    return bool(
        left_name
        and left_name == right_name
        and roles == {"foreign_key", "primary_key"}
    )


def build_schema_graph(column_infos: list[Any]) -> SchemaGraph:
    """Build edges only when an FK has one global same-name PK target."""

    table_ids: set[str] = set()
    foreign_keys: dict[str, dict[str, Any]] = defaultdict(dict)
    primary_keys: dict[str, dict[str, Any]] = defaultdict(dict)
    for column in column_infos:
        table_id = normalize_identifier(_value(column, "table_id"))
        name = normalize_identifier(_value(column, "name"))
        role = normalize_identifier(_value(column, "role"))
        if not table_id or not name:
            continue
        table_ids.add(table_id)
        column_id = f"{table_id}.{name}"
        if role == "foreign_key":
            foreign_keys[name][column_id] = column
        elif role == "primary_key":
            primary_keys[name][column_id] = column

    adjacency: dict[str, set[JoinEdge]] = {table_id: set() for table_id in table_ids}
    ambiguous_adjacency: dict[str, set[JoinEdge]] = {
        table_id: set() for table_id in table_ids
    }
    for name, foreign_key_map in foreign_keys.items():
        for foreign_key in foreign_key_map.values():
            foreign_table = normalize_identifier(_value(foreign_key, "table_id"))
            candidates = [
                primary_key
                for primary_key in primary_keys.get(name, {}).values()
                if normalize_identifier(_value(primary_key, "table_id"))
                != foreign_table
            ]
            if len(candidates) == 1:
                _add_edge(adjacency, _join_edge(foreign_key, candidates[0]))
            elif len(candidates) > 1:
                for primary_key in candidates:
                    _add_edge(
                        ambiguous_adjacency,
                        _join_edge(foreign_key, primary_key),
                    )

    return SchemaGraph(
        adjacency=_freeze_adjacency(adjacency),
        ambiguous_adjacency=_freeze_adjacency(ambiguous_adjacency),
    )


def unique_join_edges(graph: SchemaGraph) -> tuple[JoinEdge, ...]:
    """Return each deterministic relationship once in stable order."""

    return tuple(
        sorted(
            {
                edge
                for edges in graph.adjacency.values()
                for edge in edges
            }
        )
    )


def build_relationship_graph(relationships: list[Any]) -> SchemaGraph:
    """Build a graph from already-authoritative relationship records."""

    adjacency: dict[str, set[JoinEdge]] = defaultdict(set)
    for relationship in relationships:
        endpoints = sorted(
            (
                (
                    normalize_identifier(_value(relationship, "left_table_id")),
                    normalize_identifier(_value(relationship, "left_column_id")),
                ),
                (
                    normalize_identifier(_value(relationship, "right_table_id")),
                    normalize_identifier(_value(relationship, "right_column_id")),
                ),
            )
        )
        if not all(table_id and column_id for table_id, column_id in endpoints):
            continue
        edge = JoinEdge(
            left_table=endpoints[0][0],
            left_column=endpoints[0][1],
            right_table=endpoints[1][0],
            right_column=endpoints[1][1],
        )
        _add_edge(adjacency, edge)
    return SchemaGraph(
        adjacency=_freeze_adjacency(adjacency),
        ambiguous_adjacency={},
    )


def stable_relationship_id(edge: JoinEdge) -> str:
    """Build a stable ID from the sorted authoritative join endpoints."""

    left, right = sorted(edge.column_ids)
    return f"relationship:{left}:{right}"


def find_unique_shortest_join_closure(
    graph: SchemaGraph, required_tables: set[str]
) -> JoinClosureResult:
    """Return the conservative v1 closure without global Steiner inference.

    Version one auto-accepts only deterministic incremental shortest paths in a
    tree/forest relationship component. A relevant cycle, a locally tied path,
    or an ambiguous FK target is rejected as ambiguous even when a global
    Steiner solver could identify one optimum.
    """

    terminals = sorted(
        {
            normalize_identifier(table)
            for table in required_tables
            if normalize_identifier(table)
        }
    )
    if not terminals:
        return JoinClosureResult(status="success")
    if len(terminals) == 1:
        return JoinClosureResult(
            status="success", table_ids=frozenset(terminals)
        )
    if _required_tables_share_cyclic_component(graph, terminals):
        return JoinClosureResult(status="ambiguous")

    closure_tables = {terminals[0]}
    closure_edges: set[JoinEdge] = set()
    for target in terminals[1:]:
        if target in closure_tables:
            continue
        path_status, path = _unique_shortest_path(
            graph, closure_tables, target
        )
        if path_status != "success":
            return JoinClosureResult(status=path_status)
        for edge in path:
            closure_edges.add(edge)
            closure_tables.update({edge.left_table, edge.right_table})

    return JoinClosureResult(
        status="success",
        table_ids=frozenset(closure_tables),
        column_ids=frozenset(
            column_id
            for edge in closure_edges
            for column_id in edge.column_ids
        ),
        edges=tuple(sorted(closure_edges)),
    )


def _required_tables_share_cyclic_component(
    graph: SchemaGraph, terminals: list[str]
) -> bool:
    start = terminals[0]
    visited = {start}
    stack: list[tuple[str, JoinEdge | None]] = [(start, None)]
    has_cycle = False
    while stack:
        table_id, parent_edge = stack.pop()
        for edge in graph.adjacency.get(table_id, ()):
            neighbor = edge.other_table(table_id)
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append((neighbor, edge))
            elif edge != parent_edge:
                has_cycle = True
    return set(terminals) <= visited and has_cycle


def _unique_shortest_path(
    graph: SchemaGraph, sources: set[str], target: str
) -> tuple[Literal["success", "unresolved", "ambiguous"], tuple[JoinEdge, ...]]:
    distances = {source: 0 for source in sources}
    path_counts = {source: 1 for source in sources}
    uncertain = {source: False for source in sources}
    predecessors: dict[str, tuple[str, JoinEdge] | None] = {
        source: None for source in sources
    }
    queue = deque(sorted(sources))

    while queue:
        table_id = queue.popleft()
        for edge, is_ambiguous in _neighbors(graph, table_id):
            neighbor = edge.other_table(table_id)
            next_distance = distances[table_id] + 1
            if neighbor not in distances:
                distances[neighbor] = next_distance
                path_counts[neighbor] = path_counts[table_id]
                uncertain[neighbor] = uncertain[table_id] or is_ambiguous
                predecessors[neighbor] = (table_id, edge)
                queue.append(neighbor)
            elif distances[neighbor] == next_distance:
                path_counts[neighbor] = min(
                    2, path_counts[neighbor] + path_counts[table_id]
                )
                uncertain[neighbor] = (
                    uncertain[neighbor]
                    or uncertain[table_id]
                    or is_ambiguous
                )

    if target not in distances:
        return "unresolved", ()
    if path_counts[target] != 1 or uncertain[target]:
        return "ambiguous", ()

    path: list[JoinEdge] = []
    current = target
    while predecessors[current] is not None:
        previous, edge = predecessors[current]
        path.append(edge)
        current = previous
    path.reverse()
    return "success", tuple(path)


def _neighbors(
    graph: SchemaGraph, table_id: str
) -> tuple[tuple[JoinEdge, bool], ...]:
    return tuple(
        [(edge, False) for edge in graph.adjacency.get(table_id, ())]
        + [
            (edge, True)
            for edge in graph.ambiguous_adjacency.get(table_id, ())
        ]
    )


def _join_edge(left: Any, right: Any) -> JoinEdge:
    columns = sorted(
        (
            (
                normalize_identifier(_value(left, "table_id")),
                f"{normalize_identifier(_value(left, 'table_id'))}."
                f"{normalize_identifier(_value(left, 'name'))}",
            ),
            (
                normalize_identifier(_value(right, "table_id")),
                f"{normalize_identifier(_value(right, 'table_id'))}."
                f"{normalize_identifier(_value(right, 'name'))}",
            ),
        )
    )
    return JoinEdge(
        left_table=columns[0][0],
        left_column=columns[0][1],
        right_table=columns[1][0],
        right_column=columns[1][1],
    )


def _add_edge(adjacency: dict[str, set[JoinEdge]], edge: JoinEdge) -> None:
    adjacency[edge.left_table].add(edge)
    adjacency[edge.right_table].add(edge)


def _freeze_adjacency(
    adjacency: dict[str, set[JoinEdge]],
) -> dict[str, tuple[JoinEdge, ...]]:
    return {
        table_id: tuple(sorted(edges))
        for table_id, edges in sorted(adjacency.items())
    }


def _value(item: Any, name: str) -> Any:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)
