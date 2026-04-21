from gvalgo_utils import accumulate_vote_lists, choose_ids_from_votes, finalize_ids


def gvalgo_hough(cat, tab_cat, tab_image, loc_err):
    vote_lists = accumulate_vote_lists(cat, tab_cat, tab_image, loc_err, weighted=True)
    ids = choose_ids_from_votes(vote_lists, weighted=True)
    return finalize_ids(cat, ids, tab_image, loc_err)
