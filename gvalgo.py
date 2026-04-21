import numpy as np
from collections import Counter
from bsearch import bsearch


def gvalgo(cat, tab_cat, tab_image, loc_err):
    n_image = tab_image.shape[0]
    n_stars_img = int(np.max(tab_image[:, 1]))

    vote_lists = [[] for _ in range(n_stars_img)]

    # -------------------------
    # First round: accumulate votes
    # -------------------------
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

            vote_lists[img_star_1 - 1].extend([cat_star_1, cat_star_2])
            vote_lists[img_star_2 - 1].extend([cat_star_1, cat_star_2])

    # -------------------------
    # Choose IDs using confidence rule
    # -------------------------
    ids = np.zeros(n_stars_img, dtype=int)

    for i in range(n_stars_img):
        votes = vote_lists[i]

        if len(votes) == 0:
            ids[i] = 0
            continue

        counts = Counter(votes)
        sorted_votes = counts.most_common()

        best_id, best_votes = sorted_votes[0]

        if len(sorted_votes) > 1:
            second_votes = sorted_votes[1][1]
        else:
            second_votes = 0

        # confidence rule:
        # accept only if best vote is strong enough
        # and clearly better than second-best
        if best_votes >= 3 and (best_votes - second_votes) >= 2:
            ids[i] = best_id
        else:
            ids[i] = 0

    # -------------------------
    # Second round: geometric consistency check
    # -------------------------
    v2 = np.zeros(n_stars_img, dtype=int)

    for i in range(n_image):
        img_star_1 = int(tab_image[i, 0])
        img_star_2 = int(tab_image[i, 1])

        if img_star_1 != 0 and img_star_2 != 0:
            a = ids[img_star_1 - 1]
            b = ids[img_star_2 - 1]

            if a != 0 and b != 0:
                ux1, uy1, uz1 = cat[a - 1, 4], cat[a - 1, 5], cat[a - 1, 6]
                ux2, uy2, uz2 = cat[b - 1, 4], cat[b - 1, 5], cat[b - 1, 6]

                dot_val = ux1 * ux2 + uy1 * uy2 + uz1 * uz2
                dot_val = np.clip(dot_val, -1.0, 1.0)

                d_cat = np.degrees(np.arccos(dot_val))
                d_image = tab_image[i, 2]

                if (d_image - loc_err) < d_cat < (d_image + loc_err):
                    v2[img_star_1 - 1] += 1
                    v2[img_star_2 - 1] += 1

    return ids, v2