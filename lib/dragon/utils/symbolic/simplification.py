"""DAG simplification (composition penalty) for DRAGON.

Penalises deep nesting of the same operator family (e.g. ``sin(sin(x))``)
to encourage simpler, more interpretable expressions.
"""

from __future__ import annotations

import numpy as np


def composition_penalty(
    adj,
    nodes,
    groups: dict[str, str],
    base: float = 2.0,
    weight: float = 1.0,
) -> float:
    """Compute a complexity penalty for same-family operator nesting.

    Parameters
    ----------
    adj : array-like
        Adjacency matrix ``(n, n)`` where ``adj[i, j] == 1`` means
        node *i* feeds into node *j*.
    nodes : list
        Node objects. Each must have a ``name`` attribute (the brick class).
    groups : dict
        Mapping of brick class ``__name__`` to a family label
        (e.g. ``{"Sin": "trig", "Cos": "trig"}``).
    base : float
        Exponential growth base per extra composition depth.
    weight : float
        Overall weight of the penalty term.  ``0.0`` disables it.

    Returns
    -------
    float
        The penalty value (>= 0).
    """
    if weight <= 0.0 or not groups:
        return 0.0

    n = len(nodes)
    if n == 0:
        return 0.0

    # Map each node to its family label
    fam = []
    for nd in nodes:
        cls = getattr(nd, "name", None)
        cname = getattr(cls, "__name__", None)
        if cname is None:
            cname = str(cls).split(".")[-1].strip("'>\" ")
        fam.append(groups.get(cname))

    # Compute nesting depth per node (longest path within same family)
    depth = [1 if fam[i] else 0 for i in range(n)]
    for _ in range(n):
        changed = False
        for j in range(n):
            if not fam[j]:
                continue
            best = 0
            for i in range(n):
                if fam[i] == fam[j] and adj[i, j] and depth[i] > best:
                    best = depth[i]
            if best + 1 > depth[j]:
                depth[j] = best + 1
                changed = True
        if not changed:
            break

    raw = 0.0
    for j in range(n):
        if fam[j] and depth[j] >= 2:
            raw += base ** (depth[j] - 1) - 1.0
    return weight * raw
