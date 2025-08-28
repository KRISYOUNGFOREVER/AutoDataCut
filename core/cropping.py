# ... existing code ...
from typing import List, Tuple, Dict
import os
import cv2
import numpy as np
from utils.saliency import get_saliency_map
from utils.texture import get_texture_map
from utils.nms import nms

Rect = Tuple[int, int, int, int]  # (x, y, w, h)

def _integral_sum(int_img: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    """
    Compute sum over window using integral image.
    cv2.integral returns shape (H+1, W+1).
    """
    x2, y2 = x + w, y + h
    # Note: integral image is indexed as [y, x]
    return float(int_img[y2, x2] - int_img[y, x2] - int_img[y2, x] + int_img[y, x])

def _window_scores(score_map: np.ndarray, win_w: int, win_h: int, stride_x: int, stride_y: int) -> Tuple[List[Rect], List[float]]:
    H, W = score_map.shape[:2]
    if W < win_w or H < win_h:
        return [], []
    # Integral image for fast window sum
    int_img = cv2.integral(score_map.astype(np.float32))
    rects: List[Rect] = []
    scores: List[float] = []
    y = 0
    while y + win_h <= H:
        x = 0
        while x + win_w <= W:
            s = _integral_sum(int_img, x, y, win_w, win_h)
            avg = s / (win_w * win_h)
            rects.append((x, y, win_w, win_h))
            scores.append(avg)
            x += stride_x
        y += stride_y
    return rects, scores

def grid_tiles(img_shape: Tuple[int, int], win_w: int, win_h: int, overlap: float) -> List[Rect]:
    """
    网格裁切：按指定窗口大小和重叠比例在整图上生成裁切框
    """
    H, W = img_shape
    if W < win_w or H < win_h:
        # 单个窗口：裁剪到图像边界
        return [(0, 0, min(W, win_w), min(H, win_h))]
    
    step_x = max(1, int(win_w * (1 - overlap)))
    step_y = max(1, int(win_h * (1 - overlap)))
    rects = []
    
    y = 0
    while y + win_h <= H:
        x = 0
        while x + win_w <= W:
            rects.append((x, y, win_w, win_h))
            x += step_x
        y += step_y
    
    # 确保右边缘和底边缘被覆盖
    if rects:
        last_row_y = H - win_h
        last_col_x = W - win_w
        # 添加最后一列
        ys = sorted({r[1] for r in rects})
        for yy in ys:
            rects.append((last_col_x, yy, win_w, win_h))
        # 添加最后一行
        xs = sorted({r[0] for r in rects})
        for xx in xs:
            rects.append((xx, last_row_y, win_w, win_h))
    
    # 去重
    rects = list({(x, y, w, h) for (x, y, w, h) in rects})
    return rects

def grid_tiles_strict(img_shape: Tuple[int, int], win_w: int, win_h: int, overlap: float) -> List[Rect]:
    """
    严格网格裁切：只返回完整铺满的裁切框，不包含边缘不完整的部分
    """
    H, W = img_shape
    if W < win_w or H < win_h:
        return []  # 无法放置任何完整窗口
    
    step_x = max(1, int(win_w * (1 - overlap)))
    step_y = max(1, int(win_h * (1 - overlap)))
    rects = []
    
    y = 0
    while y + win_h <= H:
        x = 0
        while x + win_w <= W:
            rects.append((x, y, win_w, win_h))
            x += step_x
        y += step_y
    
    # 新增：补齐右边缘与底边缘（保持裁片尺寸不变，严格仅指“全尺寸窗口”）
    if rects:
        last_row_y = H - win_h
        last_col_x = W - win_w
        ys = sorted({r[1] for r in rects})  # 现有各行的 y
        xs = sorted({r[0] for r in rects})  # 现有各列的 x
        # 补齐最后一列（贴右边）
        for yy in ys:
            rects.append((last_col_x, yy, win_w, win_h))
        # 补齐最后一行（贴底边）
        for xx in xs:
            rects.append((xx, last_row_y, win_w, win_h))
        # 去重
        rects = list({(x, y, w, h) for (x, y, w, h) in rects})
    
    return rects

def propose_by_map(score_map: np.ndarray, target_w: int, target_h: int, stride_frac: float, top_k: int, iou_thresh: float) -> List[Rect]:
    H, W = score_map.shape[:2]
    win_w, win_h = target_w, target_h
    stride_x = max(1, int(win_w * stride_frac))
    stride_y = max(1, int(win_h * stride_frac))
    rects, scores = _window_scores(score_map, win_w, win_h, stride_x, stride_y)
    if not rects:
        return []
    keep_idx = nms(rects, scores, iou_thresh=iou_thresh, top_k=top_k)
    return [rects[i] for i in keep_idx]

def draw_rects(bgr: np.ndarray, rects: List[Rect]) -> np.ndarray:
    out = bgr.copy()
    color = (0, 255, 0)
    for (x, y, w, h) in rects:
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
    return out

def ensure_dir(path: str):
    """
    确保目录存在，兼容 Streamlit Cloud 环境
    """
    try:
        os.makedirs(path, exist_ok=True)
    except (OSError, PermissionError) as e:
        # 在 Streamlit Cloud 环境中，某些路径可能无法创建
        # 使用临时目录作为备选方案
        import tempfile
        temp_dir = tempfile.mkdtemp()
        print(f"警告：无法创建目录 {path}，使用临时目录 {temp_dir}")
        return temp_dir
    return path

def save_crop(bgr: np.ndarray, rect: Rect, out_path: str, resize_to: Tuple[int, int] = None):
    """
    保存裁切片段
    """
    x, y, w, h = rect
    crop = bgr[y:y + h, x:x + w]
    if resize_to is not None:
        crop = cv2.resize(crop, resize_to, interpolation=cv2.INTER_LANCZOS4)
    
    # 确保输出目录存在
    out_dir = os.path.dirname(out_path)
    if out_dir:
        out_dir = ensure_dir(out_dir) or out_dir
        # 如果目录被重定向，更新输出路径
        if out_dir != os.path.dirname(out_path):
            out_path = os.path.join(out_dir, os.path.basename(out_path))
    
    ext = os.path.splitext(out_path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise IOError(f"Failed to encode image for saving: {out_path}")
    
    try:
        buf.tofile(out_path)
    except (OSError, PermissionError) as e:
        # 在某些环境中，直接写入可能失败，尝试使用标准文件操作
        with open(out_path, 'wb') as f:
            f.write(buf.tobytes())

def maybe_save_full(bgr: np.ndarray, out_path: str, resize_to: Tuple[int, int] = None):
    """
    保存完整图像
    """
    # 确保输出目录存在
    out_dir = os.path.dirname(out_path)
    if out_dir:
        out_dir = ensure_dir(out_dir) or out_dir
        # 如果目录被重定向，更新输出路径
        if out_dir != os.path.dirname(out_path):
            out_path = os.path.join(out_dir, os.path.basename(out_path))
    
    img = bgr
    if resize_to is not None:
        img = cv2.resize(img, resize_to, interpolation=cv2.INTER_LANCZOS4)
    
    ext = os.path.splitext(out_path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise IOError(f"Failed to encode image for saving: {out_path}")
    
    try:
        buf.tofile(out_path)
    except (OSError, PermissionError) as e:
        # 在某些环境中，直接写入可能失败，尝试使用标准文件操作
        with open(out_path, 'wb') as f:
            f.write(buf.tobytes())

def compute_maps(bgr: np.ndarray, need_saliency: bool, need_texture: bool):
    sal_map = get_saliency_map(bgr) if need_saliency else None
    tex_map = get_texture_map(bgr) if need_texture else None
    return sal_map, tex_map

# 智能均匀抽样：在网格裁切结果中等间距挑选 total_k 个，保证覆盖且数量固定
def propose_grid_uniform(img_shape: Tuple[int, int], win_w: int, win_h: int, overlap: float, total_k: int) -> List[Rect]:
    rects = grid_tiles(img_shape, win_w, win_h, overlap)
    if total_k <= 0 or len(rects) <= total_k:
        return rects[:total_k] if total_k > 0 else []
    n = len(rects)
    # 等间距选择，尽量覆盖全图
    idxs = []
    for i in range(total_k):
        pos = int(round((i + 0.5) * n / total_k))
        pos = max(0, min(n - 1, pos))
        idxs.append(pos)
    # 去重
    idxs = sorted(set(idxs))
    # 若因去重导致数量不足，补齐
    j = 0
    while len(idxs) < total_k:
        if j not in idxs:
            idxs.append(j)
        j += 1
    idxs = sorted(idxs[:total_k])
    return [rects[i] for i in idxs]

# 颜色掩码裁切：按色相范围生成热度图（结合纹理），再用滑窗+NMS 选 top_k
def propose_color_hue(
    bgr: np.ndarray,
    target_w: int,
    target_h: int,
    stride_frac: float,
    top_k: int,
    iou_thresh: float,
    hue_ranges: List[Tuple[int, int]],  # OpenCV HSV: H in [0,180]
    sat_thresh: float = 0.2,           # S in [0,1]
    val_thresh: float = 0.1,           # V in [0,1]
    texture_weight: float = 0.3,
) -> List[Rect]:
    if top_k <= 0:
        return []
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)  # H:0-180, S:0-255, V:0-255
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask_total = np.zeros(H.shape, dtype=np.uint8)
    s_min = int(sat_thresh * 255)
    v_min = int(val_thresh * 255)
    for (lo, hi) in hue_ranges:
        if lo <= hi:
            mask = ((H >= lo) & (H <= hi) & (S >= s_min) & (V >= v_min)).astype(np.uint8)
        else:
            # 环绕 180 情况，例如红色(170-180 与 0-10)
            mask = (((H >= lo) | (H <= hi)) & (S >= s_min) & (V >= v_min)).astype(np.uint8)
        mask_total = np.maximum(mask_total, mask)
    score = mask_total.astype(np.float32)
    if texture_weight > 0:
        tex = get_texture_map(bgr)  # [0,1]
        score = (1.0 - texture_weight) * score + texture_weight * tex
    score = cv2.GaussianBlur(score, (9, 9), 0)
    score = (score - score.min()) / (score.max() - score.min() + 1e-8)
    rects = propose_by_map(score, int(target_w), int(target_h), float(stride_frac), int(top_k), float(iou_thresh))
    return rects

# 智能混合策略：融合显著性与纹理，返回指定数量的最佳裁片；不足则用网格补齐
def propose_smart_mix(
    bgr: np.ndarray,
    target_w: int,
    target_h: int,
    stride_frac: float,
    total_k: int,
    iou_thresh: float,
    saliency_weight: float = 0.6,
    texture_weight: float = 0.4,
    fallback_grid_overlap: float = 0.2,
) -> List[Rect]:
    H, W = bgr.shape[:2]
    if total_k <= 0:
        return []

    # 计算显著性和纹理热度图（均为[0,1]）
    sal_map = get_saliency_map(bgr)
    tex_map = get_texture_map(bgr)

    # 归一化权重
    w_sum = max(1e-8, float(saliency_weight) + float(texture_weight))
    sw = float(saliency_weight) / w_sum
    tw = float(texture_weight) / w_sum

    # 融合热度图并用 propose_by_map 选出前 total_k
    fused_map = sw * sal_map + tw * tex_map
    rects = propose_by_map(
        fused_map, int(target_w), int(target_h), float(stride_frac), int(total_k), float(iou_thresh)
    )

    # 若数量不足，用网格补齐（选与已选 IoU 小的）
    if len(rects) < total_k:
        need = total_k - len(rects)
        existing = rects[:]
        # 用网格候选补齐
        grid_rects = grid_tiles((H, W), int(target_w), int(target_h), float(fallback_grid_overlap))

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

        for r in grid_rects:
            if need <= 0:
                break
            ok = True
            for e in existing:
                if _iou(r, e) >= iou_thresh:
                    ok = False
                    break
            if ok:
                rects.append(r)
                existing.append(r)
                need -= 1

    return rects