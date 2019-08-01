from data import BasketConstructor
from knndtw import KnnDtw
from embedding_wrapper import EmbeddingWrapper
from helper import nested_change, remove_products_which_are_uncommon, remove_short_baskets, split_data


def run():
    embedding_wrapper = EmbeddingWrapper('product')
    bc = BasketConstructor('./data/', './data/')
    ub_basket = bc.get_baskets('prior', reconstruct=False)

    all_baskets = ub_basket.basket.values
    all_baskets = nested_change(list(all_baskets), str)

    all_baskets = embedding_wrapper.remove_products_wo_embeddings(all_baskets)
    all_baskets = remove_products_which_are_uncommon(all_baskets)
    all_baskets = remove_short_baskets(all_baskets)
    all_baskets = nested_change(all_baskets, embedding_wrapper.lookup_ind_f)

    train_ub, val_ub_input, val_ub_target, test_ub_input, test_ub_target = split_data(all_baskets)

    knndtw = KnnDtw(n_neighbors=[5])
    preds_all, distances = knndtw.predict(train_ub, val_ub_input, embedding_wrapper.basket_dist_EMD, 
                                          embedding_wrapper.basket_dist_REMD)
    return preds_all, distances
    

if __name__ == "__main__":
    run()
