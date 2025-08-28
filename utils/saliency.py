# ... existing code ...
import numpy as np
import cv2

def _normalize_map(m: np.ndarray) -> np.ndarray:
    m = m.astype(np.float32)
    m -= m.min()
    denom = (m.max() - m.min() + 1e-8)
    m = m / denom
    return m

def _opencv_spectral_residual(gray: np.ndarray) -> np.ndarray:
    try:
        # Requires opencv-contrib-python
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        ok, sal_map = saliency.computeSaliency(gray)
        if ok:
            sal_map = _normalize_map(sal_map)
            sal_map = cv2.GaussianBlur(sal_map, (9, 9), 0)
            return _normalize_map(sal_map)
    except Exception:
        pass
    return None

def _opencv_fine_grained(gray: np.ndarray) -> np.ndarray:
    try:
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        ok, sal_map = saliency.computeSaliency(gray)
        if ok:
            sal_map = _normalize_map(sal_map)
            sal_map = cv2.GaussianBlur(sal_map, (9, 9), 0)
            return _normalize_map(sal_map)
    except Exception:
        pass
    return None

def _spectral_residual_numpy(gray: np.ndarray) -> np.ndarray:
    # Fallback implementation without OpenCV saliency
    g = gray.astype(np.float32) / 255.0
    # Avoid DC bias
    g = (g - g.mean()) / (g.std() + 1e-6)
    F = np.fft.fft2(g)
    log_amp = np.log(np.abs(F) + 1e-8)
    phase = np.angle(F)
    # Average filtering on log amplitude
    avg_log_amp = cv2.blur(log_amp, (3, 3))
    spectral_residual = log_amp - avg_log_amp
    # Reconstruct
    exp_spec = np.exp(spectral_residual)
    F_ = exp_spec * np.exp(1j * phase)
    sal = np.abs(np.fft.ifft2(F_)) ** 2
    sal = cv2.GaussianBlur(sal.astype(np.float32), (9, 9), 0)
    return _normalize_map(sal)

def get_saliency_map(bgr: np.ndarray) -> np.ndarray:
    """
    Return saliency map in range [0, 1], same HxW as input.
    Prefers OpenCV saliency; falls back to Numpy spectral residual.
    """
    if bgr.ndim == 3:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = bgr.copy()
    # Try OpenCV spectral residual first (fast & good)
    sal = _opencv_spectral_residual(gray)
    if sal is not None:
        return sal
    # Try fine-grained
    sal = _opencv_fine_grained(gray)
    if sal is not None:
        return sal
    # Fallback to numpy spectral residual
    return _spectral_residual_numpy(gray)
# ... existing code ...