import numpy as np
from grow_region import grow_region
from centroid import centroid
import numpy as np

def detection(img, thresh):
    """
    Detect bright regions in image and compute centroid of each region.

    Parameters
    ----------
    img : 2D numpy array
        Grayscale image.
    thresh : float or int
        Threshold for bright pixel detection.

    Returns
    -------
    reg_img : 2D numpy array
        Region-labeled image. 0 means no region, 1..N are region labels.
    region_count : int
        Number of detected regions.
    cent : 2D numpy array
        Centroids of each region, shape (region_count, 2).
    """

    # image size
    row, col = img.shape

    # initialize region image
    region_count = 0
    reg_img = np.zeros_like(img, dtype=np.int32)

    # scan image
    for i in range(row):
        for j in range(col):
            if reg_img[i, j] == 0 and img[i, j] > thresh:
                region_count += 1
                reg_img = grow_region(img, reg_img, i, j, thresh, region_count)

    # compute centroids
    cent = np.zeros((region_count, 2), dtype=float)
    for region_id in range(1, region_count + 1):
        cent[region_id - 1, 0], cent[region_id - 1, 1] = centroid(img, reg_img, region_id)

    return reg_img, region_count, cent