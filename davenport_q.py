import numpy as np


def davenport_q(cat, star_vectors, star_id, v2=None, min_votes=0):
    """
    Estimate attitude quaternion using Davenport's q-method.

    Parameters
    ----------
    cat : ndarray
        Catalog array with columns:
        [row_index, hip_id, ra_deg, dec_deg, ux, uy, uz]
    star_vectors : ndarray
        Observed sensor/body-frame unit vectors, shape (N, 3)
    star_id : ndarray
        Identified catalog row indices for each observed star, shape (N,)
    v2 : ndarray or None
        Second-round support votes for each observed star.
    min_votes : int
        Minimum v2 votes required to keep a star.

    Returns
    -------
    q : ndarray
        Quaternion [q0, q1, q2, q3]
    used_count : int
        Number of stars used
    """
    n = len(star_id)
    B = np.zeros((3, 3), dtype=float)
    used_count = 0

    for i in range(n):
        if star_id[i] == 0:
            continue

        if v2 is not None and v2[i] < min_votes:
            continue

        row_idx = int(star_id[i]) - 1
        u_inertial = cat[row_idx, 4:7]
        u_sensor = star_vectors[i, :]

        B += np.outer(u_sensor, u_inertial)
        used_count += 1

    if used_count < 2:
        raise ValueError("Davenport Q failed: not enough reliable identified stars.")

    S = B + B.T
    sigma = np.trace(B)

    Z = np.array([
    -B[1, 2] + B[2, 1],
    -B[2, 0] + B[0, 2],
    -B[0, 1] + B[1, 0]
    ], dtype=float)

    K = np.zeros((4, 4), dtype=float)
    K[0, 0] = sigma
    K[0, 1:4] = Z
    K[1:4, 0] = Z
    K[1:4, 1:4] = S - sigma * np.eye(3)

    eigvals, eigvecs = np.linalg.eigh(K)
    q = eigvecs[:, np.argmax(eigvals)]

    q = q / np.linalg.norm(q)

    # keep scalar part positive for easier comparison with QUEST
    if q[0] < 0:
        q = -q

    return q, used_count
