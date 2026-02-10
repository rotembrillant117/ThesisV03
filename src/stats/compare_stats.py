import numpy as np
from scipy.optimize import linprog



def earth_movers_dist(categories, l1, l2, source, target, track_target=None):
    """
    Computes the Earth Movers Distance metric between to distributions. Also able to track how much earth was moved
     to a specific target (track_target)
    :param categories: the categories
    :param l1: English
    :param l2: other language
    :param source: source distribution
    :param target: target distribution
    :param track_target: target category to track
    :return: emd, moved
    """
    s = np.array([source[c] for c in categories], dtype=np.float64)
    t = np.array([target[c] for c in categories], dtype=np.float64)

    # Normalizing
    s /= s.sum()
    t /= t.sum()

    n = len(s)

    # Create distance matrix
    D = np.array([[dist(l1, l2, c1, c2) for c1 in categories] for c2 in categories], dtype=np.float64)
    # we are trying to minimize c.T@x where x is the solution for the linear program. So, c is the cost
    c = D.flatten()

    # Creating equality constraints
    A_eq = []
    b_eq = []

    # Supply constraints
    # [[ f00, f01, f02 ],
    # [ f10, f11, f12 ], ---> [f00, f01, f02, f10, f11, f12, f20, f21, f22]
    # [ f20, f21, f22 ]]
    # We add the row constraints. A_eq[i][j] for all j must sum to s[i]. This means we cannot move more "dirt" than we have in s[i]
    for i in range(n):
        matrix = np.zeros((n, n))
        matrix[i, :] = 1
        A_eq.append(matrix.flatten())
        b_eq.append(s[i])

    # We add more constraints. A_eq[i][j] for all i must sum to t[j]. This means we want to get exactly the amount of "dirt" at t[j]
    for j in range(n):
        matrix = np.zeros((n, n))
        matrix[:, j] = 1  # All rows in column j (incoming flows)
        A_eq.append(matrix.flatten())
        b_eq.append(t[j])

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='highs')
    flow_matrix = res.x.reshape((n, n))
    # Elementwise multiplication
    emd = np.sum(flow_matrix * D)
    if track_target is not None and track_target in categories:
        j = categories.index(track_target)
        moved = {categories[i]: flow_matrix[i][j] for i in range(n)}
        return emd, moved
    return emd


def dist(l1, l2, source, target):
    """
    The distance function for Earth Movers target function
    :param l1: English
    :param l2: other language
    :param source: source category
    :param target: target category
    :return:
    """
    d = {
        "same_splits": {f"{l1}_t==multi_t": 1, f"{l2}_t==multi_t": 1, f"{l1}_t=={l2}_t": 1, "different_splits": 2,
                        "same_splits": 0},
        "different_splits": {f"{l1}_t==multi_t": 1, f"{l2}_t==multi_t": 1, f"{l1}_t=={l2}_t": 1, "same_splits": 2,
                             "different_splits": 0},
        f"{l1}_t==multi_t": {f"same_splits": 1, f"{l2}_t==multi_t": 0.5, f"{l1}_t=={l2}_t": 0.7, "different_splits": 1,
                             f"{l1}_t==multi_t": 0},
        f"{l2}_t==multi_t": {f"{l1}_t==multi_t": 0.5, f"same_splits": 1, f"{l1}_t=={l2}_t": 0.7, "different_splits": 1,
                             f"{l2}_t==multi_t": 0},
        f"{l1}_t=={l2}_t": {f"{l1}_t==multi_t": 0.7, f"{l2}_t==multi_t": 0.7, f"same_splits": 1, "different_splits": 1,
                            f"{l1}_t=={l2}_t": 0}
    }

    return d[source][target]


def words_moved_to_target(num_tokens_diff1, num_tokens_diff2, categories, target):
    """
    This function checks which words moved from num_tokens_diff1 to a certain category in num_tokens_diff2
    :param num_tokens_diff1: tokenization cases 1
    :param num_tokens_diff2:tokenization cases 2
    :param categories:categories
    :param target: which words moved to target in tokenization cases 2
    :return: words moved to target
    """
    words_moved = {c: [] for c in categories}
    for c, words in num_tokens_diff1.items():
        added = set(num_tokens_diff1[c]) & set(num_tokens_diff2[target])
        for w in added:
            words_moved[c].append(w)
    return words_moved


def words_removed_from_target(num_tokens_diff1, num_tokens_diff2, categories, target):
    """
    This function checks which words moved from target in num_tokens_diff2 to other categories in num_tokens_diff1
    :param num_tokens_diff1: tokenization cases 1
    :param num_tokens_diff2: tokenization cases 2
    :param categories: categories
    :param target: which words moved out from target in tokenization cases 2
    :return: words moved out from target
    """
    words_moved = {c: [] for c in categories if c != target}
    for w in num_tokens_diff1[target]:
        if w not in set(num_tokens_diff2[target]):
            for c in words_moved.keys():
                if w in set(num_tokens_diff2[c]):
                    words_moved[c].append(w)
    return words_moved


def words_moved_to_target_ff(num_tokens_diff1, num_tokens_diff2, ff_words, categories, target):
    words_moved = words_moved_to_target(num_tokens_diff1, num_tokens_diff2, categories, target)
    ff_words_moved = {c: [] for c in categories}
    for c, words in words_moved.items():
        for ff in ff_words:
            if ff in words:
                ff_words_moved[c].append(ff)
    return ff_words_moved