"""Light RANSAC worker (numpy + OpenCV only), so process pools can be forked before torch/CUDA
are loaded. Same MAGSAC fundamental-matrix inlier count as scripts/oracle_label.py."""
import numpy as np


def ransac_inliers(args):
    import cv2
    k0, k1 = args
    if len(k0) < 8:
        return 0
    cv2.setNumThreads(1)
    try:
        _, mask = cv2.findFundamentalMat(np.asarray(k0), np.asarray(k1), cv2.USAC_MAGSAC, 1.0, 0.999, 10000)
    except cv2.error:
        return 0
    return int(mask.sum()) if mask is not None else 0
