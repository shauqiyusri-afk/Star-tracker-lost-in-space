import numpy as np


def bsearch(x, var):
    """
    Binary search for nearest values in sorted vector x.

    Parameters
    ----------
    x : array-like
        Sorted 1D array (ascending or descending).
    var : float or array-like
        Value(s) to search for.

    Returns
    -------
    index : int or np.ndarray
        0-based index/indices of nearest value(s) in x.
    """
    x = np.asarray(x).flatten()
    var = np.atleast_1d(var)

    x_len = len(x)

    if x_len == 0:
        raise ValueError("Input array x is empty.")

    # Detect order
    if x[0] > x[-1]:
        x = x[::-1]
        flipped = True
    elif x[0] < x[-1]:
        flipped = False
    else:
        raise ValueError("Badly formatted data: x must be strictly ascending or descending.")

    indices = []

    for value in var:
        low = 0
        high = x_len - 1

        if value <= x[low]:
            idx = low
            indices.append(idx)
            continue
        elif value >= x[high]:
            idx = high
            indices.append(idx)
            continue

        found_exact = False

        while low <= high:
            mid = round((low + high) / 2)

            if value < x[mid]:
                high = mid
            elif value > x[mid]:
                low = mid
            else:
                idx = mid
                found_exact = True
                break

            if low == high - 1:
                break

        if not found_exact:
            if low == high:
                idx = low
            elif (x[low] - value) ** 2 > (x[high] - value) ** 2:
                idx = high
            else:
                idx = low

        if flipped:
            idx = x_len - idx - 1

        indices.append(idx)

    if len(indices) == 1:
        return indices[0]
    return np.array(indices, dtype=int)