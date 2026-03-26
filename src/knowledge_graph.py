

from __future__ import annotations

from typing import Any


def build_country_knowledge_graph(processed_events: list[dict[str, Any]]) -> Any:
    

    try:
        import networkx as nx  # type: ignore
    except Exception:
        nx = None

    if nx is None:
        # Minimal fallback graph so CLI can still print counts.
        class _FallbackGraph:
            def __init__(self) -> None:
                self._nodes: set[str] = set()
                self._edges: int = 0

            def add_node(self, node: str) -> None:
                self._nodes.add(node)

            def add_edge(self, _u: str, _v: str, **_kwargs: Any) -> None:
                self._edges += 1

            def has_edge(self, _u: str, _v: str) -> bool:
                return False

            def number_of_nodes(self) -> int:
                return len(self._nodes)

            def number_of_edges(self) -> int:
                return self._edges

        graph: Any = _FallbackGraph()
    else:
        graph = nx.Graph()

    for event in processed_events:
        entities = event.get("entities") or {}
        countries = entities.get("GPE") or []
        countries = [c for c in countries if isinstance(c, str) and c.strip()]

        if not countries:
            continue

        sentiment = float(event.get("sentiment") or 0.0)
        confidence = float(event.get("event_confidence") or 0.0)
        edge_weight = sentiment * confidence

        # Add nodes.
        for c in countries:
            graph.add_node(c)

        # Add/update edges for co-occurrence pairs.
        unique_countries = sorted(set(countries))
        for i in range(len(unique_countries)):
            for j in range(i + 1, len(unique_countries)):
                u = unique_countries[i]
                v = unique_countries[j]
                try:
                    if nx is not None and graph.has_edge(u, v):
                        graph[u][v]["weight"] = float(graph[u][v].get("weight") or 0.0) + edge_weight
                    else:
                        graph.add_edge(u, v, weight=edge_weight)
                except Exception:
                    graph.add_edge(u, v, weight=edge_weight)

    return graph


def draw_country_knowledge_graph(graph: Any, ax: Any) -> None:
    """
    Draw the country knowledge graph using matplotlib.
    """

    ax.clear()
    ax.set_title("Country Knowledge Graph")
    ax.axis("off")

    if graph.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No countries to visualize.", ha="center", va="center")
        return

    try:
        import networkx as nx  # type: ignore
    except Exception:
        ax.text(0.5, 0.5, "networkx not installed; cannot draw graph.", ha="center", va="center")
        return

    # Layout.
    pos = nx.spring_layout(graph, seed=42)

    weights = [abs(graph[u][v].get("weight") or 0.0) for u, v in graph.edges()]
    max_w = max(weights) if weights else 1.0
    widths = [1.0 + 4.0 * (w / max_w) for w in weights]

    # Node sizing by degree.
    degrees = dict(graph.degree())
    node_sizes = [300.0 + 100.0 * degrees[n] for n in graph.nodes()]

    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=node_sizes, alpha=0.9, node_color="#4C78A8")
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=9, font_color="white")
    nx.draw_networkx_edges(graph, pos, ax=ax, width=widths, alpha=0.7, edge_color="#F58518")

    # Optional edge labels if graph is small.
    if graph.number_of_edges() <= 12:
        edge_labels = {}
        for u, v, data in graph.edges(data=True):
            edge_labels[(u, v)] = f"{float(data.get('weight') or 0.0):.2f}"
        nx.draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=edge_labels, font_size=7)

