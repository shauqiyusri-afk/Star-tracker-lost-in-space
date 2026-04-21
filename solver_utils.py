import numpy as np
from gvalgo import gvalgo
from gvalgo_ransac import gvalgo_ransac
from gvalgo_hough import gvalgo_hough
from gvalgo_ransac_hough import gvalgo_ransac_hough
from quest import quest
from davenport_q import davenport_q


GV_METHODS = {
    "base": gvalgo,
    "ransac": gvalgo_ransac,
    "hough": gvalgo_hough,
    "ransac_hough": gvalgo_ransac_hough,
}


def run_gvalgo_variant(method_name, cat, tab_cat, tab_image, loc_err, star_vectors=None):
    method_name = method_name.lower()
    if method_name not in GV_METHODS:
        raise ValueError(f"Unknown gvalgo method: {method_name}")

    if method_name in ("ransac", "ransac_hough"):
        return GV_METHODS[method_name](cat, tab_cat, tab_image, loc_err, star_vectors=star_vectors)
    return GV_METHODS[method_name](cat, tab_cat, tab_image, loc_err)


def solve_with_all_methods(cat, star_vectors, ids, v2, min_votes):
    results = {}
    for solver_name, solver_fn in [("QUEST", quest), ("Davenport Q", davenport_q)]:
        try:
            q, used_count = solver_fn(cat, star_vectors, ids, v2=v2, min_votes=min_votes)
            if used_count < 4:
                results[solver_name] = {"q": None, "used": used_count, "error": "not enough reliable stars"}
            else:
                results[solver_name] = {"q": q, "used": used_count, "error": None}
        except Exception as e:
            results[solver_name] = {"q": None, "used": 0, "error": str(e)}
    return results


def quaternion_angle_difference_deg(q1, q2):
    dot_q = np.clip(np.abs(np.dot(q1, q2)), -1.0, 1.0)
    return 2.0 * np.degrees(np.arccos(dot_q))
