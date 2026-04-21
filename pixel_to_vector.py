import numpy as np


def pixel_to_unit_vectors(centroids, image_shape, fov_deg):
    """
    Convert image centroid coordinates into approximate camera-frame unit vectors.

    Parameters
    ----------
    centroids : ndarray of shape (N, 2)
        Each row is [x, y] in pixel coordinates.
    image_shape : tuple
        Image shape as (rows, cols).
    fov_deg : float
        Horizontal field of view in degrees.

    Returns
    -------
    vectors : ndarray of shape (N, 3)
        Unit vectors [vx, vy, vz] for each centroid in camera frame.
    """
    rows, cols = image_shape

    cx = cols / 2.0
    cy = rows / 2.0

    fov_rad = np.radians(fov_deg)

    # Approximate focal length in pixel units using horizontal FOV
    f_pixels = (cols / 2.0) / np.tan(fov_rad / 2.0)

    vectors = []

    for x, y in centroids:
        # shift origin to image center
        x_cam = x - cx
        y_cam = cy - y   # flip y so image-up becomes positive

        # pinhole model
        vec = np.array([x_cam, y_cam, f_pixels], dtype=float)

        # normalize to unit vector
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        vectors.append(vec)

    return np.array(vectors)