"""
Interactive knowledge graph rendering with PyVis.

Converts an existing NetworkX graph (countries as nodes, sentiment-weighted edges)
into an interactive HTML visualization.
"""

from __future__ import annotations

from typing import Any


def _edge_style_from_weight(weight: float, neutral_eps: float = 0.02) -> tuple[str, str]:
    """
    Map sentiment weight to (color, relationship_type).

    - green: positive (cooperation)
    - red: negative (conflict)
    - gray: neutral
    """

    if weight > neutral_eps:
        return ("#2CA02C", "cooperation")
    if weight < -neutral_eps:
        return ("#D62728", "conflict")
    return ("#9E9E9E", "neutral")


def nx_to_pyvis_html(
    graph: Any,
    output_html_path: str = "graph.html",
    height: str = "650px",
    width: str = "100%",
) -> str:
    """
    Convert a NetworkX graph into a PyVis HTML file and return the HTML content.

    Requirements:
    - Nodes = countries
    - Edges = relationships
    - Edge color encodes sentiment weight sign
    - Edge hover shows relationship type + weight (+ count if available)
    """

    try:
        from pyvis.network import Network  # type: ignore
    except Exception as e:
        raise RuntimeError("pyvis is not installed. Install with: pip install pyvis") from e

    # Basic interactive graph settings.
    net = Network(height=height, width=width, bgcolor="#0E1117", font_color="#FAFAFA", directed=False)
    net.barnes_hut(gravity=-25000, central_gravity=0.3, spring_length=180, spring_strength=0.03, damping=0.12)

    # Node sizing by degree for readability.
    try:
        degrees = dict(graph.degree())
    except Exception:
        degrees = {}

    for node in getattr(graph, "nodes", lambda: [])():
        deg = int(degrees.get(node, 0))
        size = 18 + 6 * deg
        title = f"{node}<br>degree={deg}"
        net.add_node(str(node), label=str(node), title=title, size=size)

    for u, v, data in getattr(graph, "edges", lambda **_kwargs: [])(data=True):
        w = float((data or {}).get("weight") or 0.0)
        count = int((data or {}).get("count") or 1)
        color, rel = _edge_style_from_weight(w)
        title = f"type={rel}<br>weight={w:+.3f}<br>count={count}"

        # PyVis expects node ids to match what we added above (strings).
        net.add_edge(str(u), str(v), color=color, title=title, value=max(1.0, abs(w) * 10.0))

    net.write_html(output_html_path, notebook=False, open_browser=False)
    with open(output_html_path, "r", encoding="utf-8") as f:
        return f.read()

