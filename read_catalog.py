import numpy as np


def read_catalog(file_path, limiting_mag):
    """
    Read star catalog text file and filter by limiting magnitude.

    Expected columns:
    col1 = HIP ID
    col2 = magnitude
    col3 = RA in degrees
    col4 = Dec in degrees

    Returns
    -------
    star_catalog : ndarray of shape (N, 7)
        Columns:
        [row_index, hip_id, ra_deg, dec_deg, ux, uy, uz]
    """
    data = np.genfromtxt(
        file_path,
        usecols=(0, 1, 2, 3),
        invalid_raise=False
    )

    if data.ndim == 1:
        data = data.reshape(1, -1)

    # remove rows with NaN
    data = data[~np.isnan(data).any(axis=1)]

    star_rows = []
    count = 0

    for i in range(data.shape[0]):
        hip_id = int(data[i, 0])
        mag = data[i, 1]
        ra_deg = data[i, 2]
        dec_deg = data[i, 3]

        if mag < limiting_mag:
            count += 1

            ux = np.cos(np.radians(dec_deg)) * np.cos(np.radians(ra_deg))
            uy = np.cos(np.radians(dec_deg)) * np.sin(np.radians(ra_deg))
            uz = np.sin(np.radians(dec_deg))

            star_rows.append([count, hip_id, ra_deg, dec_deg, ux, uy, uz])

    star_catalog = np.array(star_rows, dtype=float)

    print(f"Limiting Magnitude = {limiting_mag:.2f}")
    print(f"Number of stars = {len(star_catalog)}")

    return star_catalog