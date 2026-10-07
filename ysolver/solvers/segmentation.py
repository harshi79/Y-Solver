"""Glyph segmentation for classic captchas.

Union-find over pixel *runs* keeps this fast in pure Python: a 200x60 captcha
has a few hundred runs, not tens of thousands of pixels to flood fill.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class Glyph:
    """One candidate character: a cropped boolean mask plus its position."""

    mask: np.ndarray
    x0: int = 0
    y0: int = 0

    @property
    def w(self) -> int:
        return int(self.mask.shape[1])

    @property
    def h(self) -> int:
        return int(self.mask.shape[0])

    @property
    def x1(self) -> int:
        return self.x0 + self.w

    @property
    def y1(self) -> int:
        return self.y0 + self.h

    @property
    def area(self) -> int:
        return int(self.mask.sum())


def _shift(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Translate ink by ``(dy, dx)``, filling the exposed border with False."""
    height, width = mask.shape
    out = np.zeros_like(mask)
    ys = slice(max(0, dy), height + min(0, dy))
    yd = slice(max(0, -dy), height + min(0, -dy))
    xs = slice(max(0, dx), width + min(0, dx))
    xd = slice(max(0, -dx), width + min(0, -dx))
    out[yd, xd] = mask[ys, xs]
    return out


def binary_erode(mask: np.ndarray, size: int = 3) -> np.ndarray:
    radius = size // 2
    out = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy or dx:
                out &= _shift(mask, dy, dx)
    return out


def binary_dilate(mask: np.ndarray, size: int = 3) -> np.ndarray:
    radius = size // 2
    out = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy or dx:
                out |= _shift(mask, dy, dx)
    return out


def binary_opening(mask: np.ndarray, size: int = 3) -> np.ndarray:
    """Morphological opening: erases anti-bot lines thinner than ``size``."""
    return binary_dilate(binary_erode(mask, size), size)


def _row_runs(row: np.ndarray) -> List[Tuple[int, int]]:
    """Half-open [start, end) spans of True values in a 1-D boolean row."""
    if not row.any():
        return []
    padded = np.concatenate(([False], row, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


class _UnionFind:
    def __init__(self) -> None:
        self.parent: List[int] = []

    def add(self) -> int:
        self.parent.append(len(self.parent))
        return len(self.parent) - 1

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _runs_to_glyphs(runs_per_node: Sequence[Sequence[Tuple[int, int, int]]]) -> List[Glyph]:
    glyphs = []
    for runs in runs_per_node:
        if not runs:
            continue
        y0 = min(r[0] for r in runs)
        y1 = max(r[0] for r in runs) + 1
        x0 = min(r[1] for r in runs)
        x1 = max(r[2] for r in runs)
        mask = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        for y, s, e in runs:
            mask[y - y0, s - x0 : e - x0] = True
        glyphs.append(Glyph(mask=mask, x0=x0, y0=y0))
    return glyphs


def connected_components(mask: np.ndarray) -> List[Glyph]:
    """8-connected components via union-find over horizontal runs."""
    if not mask.any():
        return []
    uf = _UnionFind()
    previous: List[Tuple[int, int]] = []  # (node, start, end)
    nodes: List[List[Tuple[int, int, int]]] = []

    for y in range(mask.shape[0]):
        row = mask[y]
        runs = _row_runs(row)
        current: List[Tuple[int, int]] = []
        for start, end in runs:
            node = uf.add()
            nodes.append([])
            nodes[node].append((y, start, end))
            for prev_node, ps, pe in previous:
                if ps - 1 < end and start - 1 < pe:  # 8-connectivity: touch or overlap
                    uf.union(node, prev_node)
            current.append((node, start, end))
        previous = current

    grouped: dict = {}
    for node, runs in enumerate(nodes):
        if not runs:
            continue
        grouped.setdefault(uf.find(node), []).extend(runs)
    return _runs_to_glyphs(list(grouped.values()))


def _merge_pair(a: Glyph, b: Glyph) -> Glyph:
    x0, y0 = min(a.x0, b.x0), min(a.y0, b.y0)
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    canvas = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    canvas[a.y0 - y0 : a.y1 - y0, a.x0 - x0 : a.x1 - x0] |= a.mask
    canvas[b.y0 - y0 : b.y1 - y0, b.x0 - x0 : b.x1 - x0] |= b.mask
    return Glyph(mask=canvas, x0=x0, y0=y0)


def _x_overlap(a: Glyph, b: Glyph) -> float:
    lo, hi = max(a.x0, b.x0), min(a.x1, b.x1)
    if hi <= lo:
        return 0.0
    return (hi - lo) / float(max(1, min(a.w, b.w)))


def merge_stacked(glyphs: List[Glyph]) -> List[Glyph]:
    """Re-attach detached dots and accents ('i', 'j', 'ä') to their stem."""
    items = sorted(glyphs, key=lambda g: (g.x0, g.y0))
    merged = True
    while merged:
        merged = False
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                if _x_overlap(a, b) < 0.55:
                    continue
                if a.y1 <= b.y0:  # b sits below a
                    gap = b.y0 - a.y1
                elif b.y1 <= a.y0:  # b sits above a
                    gap = a.y0 - b.y1
                else:
                    continue  # already touching horizontally-overlapping
                if gap <= max(a.h, b.h) * 0.35:
                    items[i] = _merge_pair(a, b)
                    items.pop(j)
                    merged = True
                    break
            if merged:
                break
    return items


def _min_ink_column(glyph: Glyph) -> Tuple[int, int]:
    profile = glyph.mask.sum(axis=0)
    lo = max(1, int(glyph.w * 0.28))
    hi = max(lo + 1, min(glyph.w - 1, int(glyph.w * 0.72) + 1))
    window = profile[lo:hi]
    if window.size == 0:
        return glyph.w // 2, int(profile.min())
    idx = int(np.argmin(window))
    return lo + idx, int(window[idx])


def split_wide(glyphs: List[Glyph], ratio: float = 1.7, max_depth: int = 4) -> List[Glyph]:
    """Cut glyphs that are far wider than their neighbours (touching chars)."""
    if not glyphs:
        return []
    widths = sorted(g.w for g in glyphs if g.w > 0)
    median_w = widths[len(widths) // 2] if widths else 1
    out: List[Glyph] = []
    queue = [(g, 0) for g in glyphs]
    while queue:
        glyph, depth = queue.pop(0)
        threshold = max(median_w * ratio, 8)
        # A character is never twice as wide as it is tall; single-glyph inputs
        # (nothing to compare against) still get split by this aspect rule.
        too_wide = glyph.w > threshold or glyph.w > 1.6 * glyph.h
        if depth < max_depth and too_wide and glyph.h > 4:
            col, ink = _min_ink_column(glyph)
            if ink <= max(1, int(glyph.h * 0.16)) and 0 < col < glyph.w - 1:
                left = glyph.mask[:, :col]
                right = glyph.mask[:, col:]
                queue.append((Glyph(left, glyph.x0, glyph.y0), depth + 1))
                queue.append((Glyph(right, glyph.x0 + col, glyph.y0), depth + 1))
                continue
        out.append(glyph)
    return out


def density(glyph: Glyph) -> float:
    """Ink coverage inside the bounding box — characters are solid, lines are not."""
    return glyph.area / float(max(1, glyph.w * glyph.h))


def _median(values: List[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2] if ordered else 0.0


def elongation(glyph: Glyph) -> float:
    """Longest side over shortest side — characters stay near 1, lines do not."""
    return max(glyph.w, glyph.h) / float(max(1, min(glyph.w, glyph.h)))


def is_clutter(glyph: Glyph) -> bool:
    """Heuristics for anti-bot lines, arc fragments and speckle dots."""
    ratio = elongation(glyph)
    if ratio > 8.0:
        return True  # a hairline sweeping across the image
    # A thin diagonal bar or arc fragment: long but far from solid.
    return ratio > 3.5 and density(glyph) < 0.35


def drop_noise(glyphs: List[Glyph]) -> List[Glyph]:
    """Remove anti-bot clutter: thin sweeping lines, arcs and speckles.

    A glyph is kept when it is either reasonably *solid* (characters are) or a
    compact blob about the right size to be the dot of an 'i'/'j'.
    """
    if len(glyphs) < 3:
        return glyphs

    solid = [g for g in glyphs if density(g) >= 0.12 and not is_clutter(g)]
    reference_h = _median([g.h for g in solid]) or _median([g.h for g in glyphs])
    reference_area = _median([g.area for g in solid]) or _median([g.area for g in glyphs])
    blob_min = max(12.0, reference_area * 0.10)

    kept: List[Glyph] = []
    for glyph in glyphs:
        if is_clutter(glyph):
            continue
        if density(glyph) >= 0.12:
            kept.append(glyph)
            continue
        if glyph.area >= blob_min and glyph.h <= reference_h * 0.45:
            kept.append(glyph)  # detached dot / accent
            continue
        if glyph.area >= blob_min and glyph.w <= reference_h * 0.45:
            kept.append(glyph)  # narrow stem of 'i', 'l', '1'
    return kept or glyphs


def segment(mask: np.ndarray) -> List[Glyph]:
    """Full pipeline: components -> merge dots -> denoise -> split touch -> sort."""
    glyphs = connected_components(mask)
    if not glyphs:
        return []
    glyphs = merge_stacked(glyphs)
    glyphs = drop_noise(glyphs)
    glyphs = split_wide(glyphs)
    return sorted(glyphs, key=lambda g: g.x0)
