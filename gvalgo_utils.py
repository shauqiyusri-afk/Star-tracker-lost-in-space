import numpy as np
from collections import Counter, defaultdict
from bsearch import bsearch


def accumulate_vote_lists(cat, tab_cat, tab_image, loc_err, weighted=False):
    n_image = tab_image.shape[0]
    n_stars_img = int(np.max(tab_image[:, 1]))
    if weighted:
        vote_lists = [defaultdict(float) for _ in range(n_stars_img)]
    else:
        vote_lists = [[] for _ in range(n_stars_img)]

    for i in range(n_image):
        img_star_1 = int(tab_image[i, 0])
        img_star_2 = int(tab_image[i, 1])
        img_angle = tab_image[i, 2]

        index_min = bsearch(tab_cat[:, 2], img_angle - loc_err)
        index_max = bsearch(tab_cat[:, 2], img_angle + loc_err)

        if index_min > index_max:
            index_min, index_max = index_max, index_min

        for j in range(index_min, index_max + 1):
            cat_star_1 = int(tab_cat[j, 0])
            cat_star_2 = int(tab_cat[j, 1])
            cat_angle = tab_cat[j, 2]

            if weighted:
                weight = max(0.0, 1.0 - abs(cat_angle - img_angle) / max(loc_err, 1e-12))
                vote_lists[img_star_1 - 1][cat_star_1] += weight
                vote_lists[img_star_1 - 1][cat_star_2] += weight
                vote_lists[img_star_2 - 1][cat_star_1] += weight
                vote_lists[img_star_2 - 1][cat_star_2] += weight
            else:
                vote_lists[img_star_1 - 1].extend([cat_star_1, cat_star_2])
                vote_lists[img_star_2 - 1].extend([cat_star_1, cat_star_2])

    return vote_lists


def choose_ids_from_votes(vote_lists, weighted=False):
    n_stars_img = len(vote_lists)
    ids = np.zeros(n_stars_img, dtype=int)

    for i in range(n_stars_img):
        votes = vote_lists[i]
        if len(votes) == 0:
            ids[i] = 0
            continue

        if weighted:
            sorted_votes = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
        else:
            counts = Counter(votes)
            sorted_votes = counts.most_common()

        best_id, best_votes = sorted_votes[0]
        second_votes = sorted_votes[1][1] if len(sorted_votes) > 1 else 0

        if weighted:
            if best_votes >= 1.5 and (best_votes - second_votes) >= 0.35:
                ids[i] = int(best_id)
            else:
                ids[i] = 0
        else:
            if best_votes >= 3 and (best_votes - second_votes) >= 2:
                ids[i] = int(best_id)
            else:
                ids[i] = 0

    return ids


def geometric_consistency_votes(cat, ids, tab_image, loc_err):
    n_stars_img = len(ids)
    n_image = tab_image.shape[0]
    v2 = np.zeros(n_stars_img, dtype=int)

    for i in range(n_image):
        img_star_1 = int(tab_image[i, 0])
        img_star_2 = int(tab_image[i, 1])
        a = ids[img_star_1 - 1]
        b = ids[img_star_2 - 1]

        if a == 0 or b == 0:
            continue

        ux1, uy1, uz1 = cat[a - 1, 4], cat[a - 1, 5], cat[a - 1, 6]
        ux2, uy2, uz2 = cat[b - 1, 4], cat[b - 1, 5], cat[b - 1, 6]

        dot_val = np.clip(ux1 * ux2 + uy1 * uy2 + uz1 * uz2, -1.0, 1.0)
        d_cat = np.degrees(np.arccos(dot_val))
        d_image = tab_image[i, 2]

        if (d_image - loc_err) < d_cat < (d_image + loc_err):
            v2[img_star_1 - 1] += 1
            v2[img_star_2 - 1] += 1

    return v2


def estimate_rotation_kabsch(body_vecs, inertial_vecs):
    H = body_vecs.T @ inertial_vecs
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def ransac_refine_ids(cat, star_vectors, ids, v2, loc_err, min_votes_seed=2, iterations=80):
    reliable_idx = np.where((ids > 0) & (v2 >= min_votes_seed))[0]
    if len(reliable_idx) < 3:
        return ids, v2

    inertial_all = np.array([cat[int(ids[i]) - 1, 4:7] for i in reliable_idx], dtype=float)
    body_all = np.array([star_vectors[i] for i in reliable_idx], dtype=float)

    best_inlier_local = None
    best_count = 0
    threshold_deg = max(2.0 * loc_err, 0.12)

    rng = np.random.default_rng(42)
    for _ in range(iterations):
        sample_size = 2 if len(reliable_idx) >= 2 else len(reliable_idx)
        sample_local = rng.choice(len(reliable_idx), size=sample_size, replace=False)

        try:
            R = estimate_rotation_kabsch(body_all[sample_local], inertial_all[sample_local])
        except np.linalg.LinAlgError:
            continue

        pred = body_all @ R
        dots = np.sum(pred * inertial_all, axis=1)
        dots = np.clip(dots, -1.0, 1.0)
        errors_deg = np.degrees(np.arccos(dots))
        inlier_local = np.where(errors_deg < threshold_deg)[0]

        if len(inlier_local) > best_count:
            best_count = len(inlier_local)
            best_inlier_local = inlier_local

    if best_inlier_local is None or best_count < 3:
        return ids, v2

    keep_global = set(reliable_idx[best_inlier_local].tolist())
    refined_ids = ids.copy()
    refined_ids[[i for i in reliable_idx if i not in keep_global]] = 0
    refined_v2 = geometric_consistency_votes(cat, refined_ids, _rebuild_pairs_from_ids(ids, v2), loc_err) if False else v2.copy()
    for i in reliable_idx:
        if i not in keep_global:
            refined_v2[i] = 0
    return refined_ids, refined_v2


def finalize_ids(cat, ids, tab_image, loc_err):
    v2 = geometric_consistency_votes(cat, ids, tab_image, loc_err)
    return ids, v2
