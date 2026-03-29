

from __future__ import annotations

from datetime import datetime
from typing import Any


_COUNTRY_ALIASES = {
    # United States
    "us": "United States",
    "u.s.": "United States",
    "u.s": "United States",
    "usa": "United States",
    "u.s.a.": "United States",
    "united states of america": "United States",
    # United Kingdom
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "u.k": "United Kingdom",
    "britain": "United Kingdom",
    "great britain": "United Kingdom",
    # Russia
    "russian federation": "Russia",
    # UAE
    "uae": "United Arab Emirates",
}


def _normalize_country_name(name: str) -> str | None:
    """
    Normalize and clean country-like strings.

    - merges aliases (US/USA/U.S.) into canonical names
    - removes obviously incomplete / noisy names
    """

    if not isinstance(name, str):
        return None

    raw = name.strip()
    if not raw:
        return None

    # Remove obvious noise.
    if "…" in raw or "..." in raw:
        return None
    if raw.lower() in {"unknown", "n/a", "none", "null"}:
        return None
    if len(raw) <= 2 and raw.lower() not in {"us", "uk"}:
        return None

    key = raw.lower().strip()
    key = key.replace("’", "'")

    canonical = _COUNTRY_ALIASES.get(key)
    if canonical:
        return canonical

    # Title-case most names (keeps multi-word countries readable).
    # Special-case common acronyms.
    if raw.isupper() and len(raw) <= 5:
        return raw  # e.g., "UAE" if it comes through that way

    return raw


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
        countries_norm: list[str] = []
        for c in countries:
            normalized = _normalize_country_name(c)
            if normalized:
                countries_norm.append(normalized)

        if not countries_norm:
            continue

        sentiment = float(event.get("sentiment") or 0.0)
        confidence = float(event.get("event_confidence") or 0.0)
        edge_weight = sentiment * confidence

        # Add nodes.
        for c in countries_norm:
            graph.add_node(c)

        # Add/update edges for co-occurrence pairs.
        unique_countries = sorted(set(countries_norm))
        for i in range(len(unique_countries)):
            for j in range(i + 1, len(unique_countries)):
                u = unique_countries[i]
                v = unique_countries[j]
                try:
                    if nx is not None and graph.has_edge(u, v):
                        graph[u][v]["weight"] = float(graph[u][v].get("weight") or 0.0) + edge_weight
                        graph[u][v]["count"] = int(graph[u][v].get("count") or 1) + 1
                    else:
                        graph.add_edge(u, v, weight=edge_weight, count=1)
                except Exception:
                    graph.add_edge(u, v, weight=edge_weight, count=1)

    return graph


def draw_country_knowledge_graph(
    graph: Any,
    ax: Any,
    min_abs_weight: float = 0.05,
    min_count: int = 1,
) -> None:
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
    # Filter weak edges for readability.
    edges = []
    for u, v, data in graph.edges(data=True):
        w = float(data.get("weight") or 0.0)
        c = int(data.get("count") or 1)
        if abs(w) < min_abs_weight:
            continue
        if c < min_count:
            continue
        edges.append((u, v, w, c))

    # If filtering removed everything, fall back to showing the strongest edges.
    if not edges:
        tmp = []
        for u, v, data in graph.edges(data=True):
            w = float(data.get("weight") or 0.0)
            c = int(data.get("count") or 1)
            tmp.append((u, v, w, c))
        tmp.sort(key=lambda x: abs(x[2]), reverse=True)
        edges = tmp[: max(1, min(15, len(tmp)))]

    # Build a subgraph for drawing (so weak edges don't clutter the layout).
    g = graph.edge_subgraph([(u, v) for u, v, _, _ in edges]).copy()
    for n in graph.nodes():
        if n in g.nodes:
            continue
        # keep isolated nodes only if small graph
        if graph.number_of_nodes() <= 10:
            g.add_node(n)

    pos = nx.spring_layout(g, seed=42, k=0.8)

    weights = [abs(w) for _, _, w, _ in edges]
    max_w = max(weights) if weights else 1.0
    widths = [1.0 + 5.0 * (abs(w) / max_w) for _, _, w, _ in edges]
    edge_colors = ["#2CA02C" if w >= 0 else "#D62728" for _, _, w, _ in edges]  # green / red

    # Node sizing by degree.
    degrees = dict(g.degree())
    node_sizes = [500.0 + 250.0 * degrees.get(n, 0) for n in g.nodes()]

    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=node_sizes, alpha=0.92, node_color="#4C78A8", linewidths=1.0, edgecolors="white")
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=9, font_color="white")
    nx.draw_networkx_edges(
        g,
        pos,
        ax=ax,
        width=widths,
        alpha=0.75,
        edge_color=edge_colors,
    )

    # Optional edge labels if graph is small.
    if g.number_of_edges() <= 14:
        edge_labels = {}
        for u, v, data in g.edges(data=True):
            w = float(data.get("weight") or 0.0)
            c = int(data.get("count") or 1)
            edge_labels[(u, v)] = f"{w:+.2f} ({c}x)"
        nx.draw_networkx_edge_labels(g, pos, ax=ax, edge_labels=edge_labels, font_size=7)


def draw_timeline_country_graph(processed_events: list[dict[str, Any]], ax: Any) -> None:
    """
    Timeline-style knowledge graph:
    - X-axis: event date
    - Y-axis: country names
    - Vertical connectors at each date link co-mentioned countries
    - Connector color = sentiment (green/red/gray)
    """

    ax.clear()
    ax.set_title("Timeline Knowledge Graph (Country Relationships Over Time)")

    if not processed_events:
        ax.text(0.5, 0.5, "No events to visualize.", ha="center", va="center")
        ax.axis("off")
        return

    try:
        import dateparser  # type: ignore
        import matplotlib.dates as mdates  # type: ignore
    except Exception:
        ax.text(0.5, 0.5, "dateparser/matplotlib.dates unavailable.", ha="center", va="center")
        ax.axis("off")
        return

    rows: list[tuple[datetime, list[str], float]] = []
    all_countries: set[str] = set()

    for event in processed_events:
        published_at = (event.get("publishedAt") or "").strip()
        dt = dateparser.parse(published_at) if published_at else None
        if dt is None:
            continue

        entities = event.get("entities") or {}
        countries = entities.get("GPE") or []
        countries_clean = []
        for c in countries:
            cc = _normalize_country_name(c)
            if cc:
                countries_clean.append(cc)
                all_countries.add(cc)

        if len(set(countries_clean)) < 2:
            continue

        sentiment = float(event.get("sentiment") or 0.0)
        rows.append((dt, sorted(set(countries_clean)), sentiment))

    if not rows or not all_countries:
        ax.text(0.5, 0.5, "Insufficient country co-occurrence data.", ha="center", va="center")
        ax.axis("off")
        return

    countries_sorted = sorted(all_countries)
    y_map = {c: i for i, c in enumerate(countries_sorted)}

    # Draw points and vertical relation connectors at each event date.
    for dt, countries, sentiment in rows:
        y_vals = [y_map[c] for c in countries if c in y_map]
        if len(y_vals) < 2:
            continue

        if sentiment > 0.02:
            color = "#2CA02C"  # positive
        elif sentiment < -0.02:
            color = "#D62728"  # negative
        else:
            color = "#9E9E9E"  # neutral

        x_vals = [dt] * len(y_vals)
        ax.scatter(x_vals, y_vals, color="#4C78A8", s=45, alpha=0.9, zorder=3)
        ax.plot([dt, dt], [min(y_vals), max(y_vals)], color=color, linewidth=2.2, alpha=0.75, zorder=2)

    ax.set_yticks(list(range(len(countries_sorted))))
    ax.set_yticklabels(countries_sorted, fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Countries")

