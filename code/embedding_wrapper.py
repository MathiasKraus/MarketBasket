import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from scipy.spatial.distance import cdist, pdist, squareform, euclidean
import ot 


class EmbeddingWrapper(object):
    def __init__(self, type):
        if type == 'product':
            self.model = Word2Vec.load("product2vec_min_count_50.model")
            
        elif type == 'aisles':
            self.model = Word2Vec.load("aisles2vec_min_count_50.model")
            self.products = pd.read_csv("./data/products.csv")
            self.p2aisles = dict(zip(self.products.product_id.astype(str), self.products.aisle_id.astype(str)))
            
        elif type == 'tafeng_products':
            self.model = Word2Vec.load("tafeng2vec_min_count_50.model")
            
        self.vocab_len = len(self.model.wv.vocab)
        self.word2index = dict(zip([self.model.wv.index2word[i] for i in range(self.vocab_len)],
                              [i for i in range(self.vocab_len)]))
        self.word_index_df = pd.DataFrame(data=list(self.word2index.items()), columns=['product_id', 'emb_id'])
        
    def p2aisle_f(self, i):
        return self.p2aisles[i]

    def lookup_ind_f(self, i):
        return self.word2index[i]

    def get_closest_of_set(self, item_id, set_of_candidates):
        vec_of_interest = self.model.wv.vectors[item_id]
        closest = np.argmin([euclidean(vec_of_interest, self.model.wv.vectors[x]) for x in set_of_candidates])
        return set_of_candidates[closest]
    
    def find_closest_from_preds(self, pred, candidates_l_l):
        closest_from_history = []
        for p in pred:
            closest_from_history.append(self.get_closest_of_set(p, [x for seq in candidates_l_l for x in seq]))
        return closest_from_history
        
    def basket_dist_REMD(self, baskets):
        #Relaxed EMD as lower bound. It is basically a nearest neighborhood search to 
        #find the closest word in doc B for each word in doc A and then take sum of all minimum distances.    
        basket1_vecs = self.model.wv.vectors[[x for x in baskets[0]]]
        basket2_vecs = self.model.wv.vectors[[x for x in baskets[1]]]
        
        distance_matrix = cdist(basket1_vecs, basket2_vecs)
        
        return max(np.mean(np.min(distance_matrix, axis=0)),
                   np.mean(np.min(distance_matrix, axis=1)))
        
    def basket_dist_EMD(self, baskets):
        basket1 = baskets[0]
        basket2 = baskets[1]
        dictionary = np.unique(list(basket1) + list(basket2))
        vocab_len_ = len(dictionary)
        product2ind = dict(zip(dictionary, np.arange(vocab_len_)))

        # Compute distance matrix.
        dictionary_vecs = self.model.wv.vectors[[x for x in dictionary]]
        distance_matrix = squareform(pdist(dictionary_vecs))

        if np.sum(distance_matrix) == 0.0:
            # `emd` gets stuck if the distance matrix contains only zeros.
            return float('inf')

        def nbow(document):
            bow = np.zeros(vocab_len_, dtype=np.float)
            for d in document:
                bow[product2ind[d]] += 1.
            return bow / len(document)

        # Compute nBOW representation of documents.
        d1 = nbow(basket1)
        d2 = nbow(basket2)

        # Compute WMD.
        return ot.emd2(d1, d2, distance_matrix)
        
    def remove_products_wo_embeddings(self, all_baskets):
        all_baskets_filtered = []
        for s in all_baskets:
            s_cp = []
            for b in s:
                b_cp = [x for x in b if x in self.model.wv.vocab]
                if len(b_cp) > 0:
                    s_cp.append(b_cp)
            if len(s_cp) > 0:
                all_baskets_filtered.append(s_cp)
        return all_baskets_filtered
