from gvalgo_utils import accumulate_vote_lists, choose_ids_from_votes, finalize_ids, ransac_refine_ids


def gvalgo_ransac_hough(cat, tab_cat, tab_image, loc_err, star_vectors=None):
    vote_lists = accumulate_vote_lists(cat, tab_cat, tab_image, loc_err, weighted=True)
    ids = choose_ids_from_votes(vote_lists, weighted=True)
    ids, v2 = finalize_ids(cat, ids, tab_image, loc_err)

    if star_vectors is not None:
        ids, v2 = ransac_refine_ids(cat, star_vectors, ids, v2, loc_err)
        ids, v2 = finalize_ids(cat, ids, tab_image, loc_err)

    return ids, v2
