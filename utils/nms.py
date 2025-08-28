# ... existing code ...
from typing import List, Tuple
import numpy as np

Rect = Tuple[int, int, int, int]  # (x, y, w, h)

def _iou(a: Rect, b: Rect) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_w = max(0, min(ax2, bx2) - max(ax, bx))
    inter_h = max(0, min(ay2, by2) - max(ay, by))
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / union

def nms(rects: List[Rect], scores: List[float], iou_thresh: float, top_k: int) -> List[int]:
    """
    Return indices of kept rectangles after NMS.
    """
    if not rects:
        return []
    order = np.argsort(-np.array(scores))  # descending
    keep = []
    taken = np.zeros(len(rects), dtype=bool)
    for idx in order:
        if taken[idx]:
            continue
        keep.append(idx)
        if len(keep) >= top_k:
            break
        # Suppress overlaps
        for j in order:
            if taken[j] or j == idx:
                continue
            if _iou(rects[idx], rects[j]) >= iou_thresh:
                taken[j] = True
    return keep
# ... existing code ...