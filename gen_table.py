import numpy as np


def gen_table(cat, fov_deg, min_separation_deg):
    """
    Generate catalog pair-angle lookup table.

    cat columns:
    [row_index, hip_id, ra_deg, dec_deg, ux, uy, uz]
    """
    n = cat.shape[0]
    rows = []

    for i in range(n - 1):
        for j in range(i + 1, n):
            dot = cat[i, 4] * cat[j, 4] + cat[i, 5] * cat[j, 5] + cat[i, 6] * cat[j, 6]
            dot = np.clip(dot, -1.0, 1.0)

            d = np.degrees(np.arccos(dot))

            if d < fov_deg and d > min_separation_deg:
                rows.append([i + 1, j + 1, d])

    if len(rows) == 0:
        return np.zeros((0, 3), dtype=float)

    table = np.array(rows, dtype=float)
    return table