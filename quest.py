import numpy as np


def quest(cat, star_vectors, star_id, v2=None, min_votes=0):
    """
    Proper QUEST implementation using Newton-Raphson on Shuster's quartic.

    Returns
    -------
    q : ndarray
        Quaternion [q0, q1, q2, q3] (scalar first)
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
        raise ValueError("QUEST failed: not enough reliable identified stars.")

    S = B + B.T
    sigma = np.trace(B)

    Z = np.array([
        -B[1, 2] + B[2, 1],
        -B[2, 0] + B[0, 2],
        -B[0, 1] + B[1, 0]
    ], dtype=float)

    # Shuster QUEST auxiliary terms
    detS = np.linalg.det(S)
    adjS = detS * np.linalg.inv(S) if abs(detS) > 1e-12 else np.zeros((3, 3))
    kappa = np.trace(adjS)

    ZTZ = float(np.dot(Z, Z))
    ZTSZ = float(Z @ S @ Z)
    ZTS2Z = float(Z @ S @ S @ Z)

    a = sigma**2 - kappa
    b = sigma**2 + ZTZ
    c = detS + ZTSZ
    d = ZTS2Z

    # QUEST characteristic polynomial
    def f(lam):
        return lam**4 - (a + b) * lam**2 - c * lam + (a * b + c * sigma - d)

    def fp(lam):
        return 4 * lam**3 - 2 * (a + b) * lam - c

    # Newton-Raphson initial guess
    lam = float(used_count)

    for _ in range(50):
        fl = f(lam)
        fpl = fp(lam)
        if abs(fpl) < 1e-12:
            break
        lam_new = lam - fl / fpl
        if abs(lam_new - lam) < 1e-12:
            lam = lam_new
            break
        lam = lam_new

    # Recover quaternion from QUEST formulas
    alpha = lam**2 - sigma**2 + kappa
    beta = lam - sigma
    gamma = (lam + sigma) * alpha - detS
    X = (alpha * np.eye(3) + beta * S + S @ S) @ Z

    q = np.array([gamma, X[0], X[1], X[2]], dtype=float)

    norm_q = np.linalg.norm(q)
    if norm_q < 1e-12:
        raise ValueError("QUEST failed: quaternion norm is zero.")

    q /= norm_q

    if q[0] < 0:
        q = -q

    return q, used_count