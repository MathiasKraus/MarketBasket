from sklearn.model_selection import train_test_split
from collections import Counter


def nested_change(item, func):
    if isinstance(item, list):
        return [nested_change(x, func) for x in item]
    return func(item)


def remove_products_which_are_uncommon(all_baskets, max_num=500):
    print('Removing all but {} most common products'.format(max_num))
    p = []
    for s in all_baskets:
        for b in s:
            p.extend(b)
    product_counter = Counter(p)
    most_common_products = [x for x, _ in product_counter.most_common(max_num)]
    all_baskets_filtered = []
    for s in all_baskets:
        s_cp = []
        for b in s:
            b_cp = [x for x in b if x in most_common_products]
            if len(b_cp) > 0:
                s_cp.append(b_cp)
        if len(s_cp) > 0:
            all_baskets_filtered.append(s_cp)
    return all_baskets_filtered


def remove_short_baskets(all_baskets, l_b = 5, l_s = 10):
    all_baskets_filtered = []
    for s in all_baskets:
        s_cp = []
        for b in s:
            if len(b) > l_b:
                s_cp.append(b)
        if len(s_cp) > l_s:
            all_baskets_filtered.append(s_cp)
    return all_baskets_filtered


def split_data(all_baskets):
    train_ub, test_ub = train_test_split(all_baskets, test_size=0.05, random_state=0)
    train_ub, val_ub = train_test_split(train_ub, test_size=0.05, random_state=0)
    
    test_ub_input = [x[:-1] for x in test_ub]
    test_ub_target = [x[-1] for x in test_ub]
    
    val_ub_input = [x[:-1] for x in val_ub]
    val_ub_target = [x[-1] for x in val_ub]
    
    return train_ub, val_ub_input, val_ub_target, test_ub_input, test_ub_target