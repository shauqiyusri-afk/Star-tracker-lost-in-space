import numpy as np


def build_tab_image(star_vectors, fov_deg, min_sep_deg):
    """
    Build image pair-angle table from detected star unit vectors.
    """

    n = len(star_vectors)
    rows = []

    for i in range(n - 1):
        for j in range(i + 1, n):

            dot = np.dot(star_vectors[i], star_vectors[j])
            dot = np.clip(dot, -1.0, 1.0)

            d = np.degrees(np.arccos(dot))

            if d < fov_deg and d > min_sep_deg:
                rows.append([i + 1, j + 1, d])

    if len(rows) == 0:
        return np.zeros((0, 3))

    tab_image = np.array(rows)

    tab_image = tab_image[np.argsort(tab_image[:, 2])]

    return tab_image