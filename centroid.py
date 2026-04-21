import numpy as np

def centroid(img, reg_img, region_count):
    """
    Compute brightness-weighted centroid of a region.
    """

    row, col = img.shape

    num_row = 0
    num_col = 0
    den = 0

    for i in range(row):
        for j in range(col):
            if reg_img[i, j] == region_count:
                num_row += i * img[i, j]
                num_col += j * img[i, j]
                den += img[i, j]

    x = num_col / den
    y = num_row / den

    return x, y