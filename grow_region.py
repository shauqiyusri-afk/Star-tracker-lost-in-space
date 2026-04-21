import numpy as np

def grow_region(img, reg_img, row, col, thresh, region_count):
    """
    8-neighbour recursive region growing.
    """

    r, c = img.shape

    neighbors = [
        (0,1), (1,0), (0,-1), (-1,0),
        (1,1), (1,-1), (-1,-1), (-1,1)
    ]

    reg_img[row, col] = region_count

    for dx, dy in neighbors:
        rown = row + dx
        coln = col + dy

        if 0 <= rown < r and 0 <= coln < c:
            if reg_img[rown, coln] == 0 and img[rown, coln] > thresh:
                reg_img[rown, coln] = region_count
                grow_region(img, reg_img, rown, coln, thresh, region_count)

    return reg_img