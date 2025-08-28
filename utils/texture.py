# ... existing code ...
import numpy as np
import cv2

def _normalize_map(m: np.ndarray) -> np.ndarray:
    m = m.astype(np.float32)
    m -= m.min()
    denom = (m.max() - m.min() + 1e-8)
    m = m / denom
    return m

def get_texture_map(bgr: np.ndarray) -> np.ndarray:
    """
    Return texture-strength map in [0,1] using gradient magnitude and Laplacian.
    Designed to highlight brush strokes / fine details.
    """
    if bgr.ndim == 3:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = bgr.copy()

    gray = gray.astype(np.float32)

    # Strong gradients: Scharr is more sensitive than Sobel
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    grad_mag = cv2.magnitude(gx, gy)

    # Laplacian detects edges/lines
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    lap_abs = np.abs(lap)

    # Combine
    tex = 0.7 * grad_mag + 0.3 * lap_abs

    # Smooth a bit then normalize
    tex = cv2.GaussianBlur(tex, (5, 5), 0)
    tex = _normalize_map(tex)
    return tex
# ... existing code ...