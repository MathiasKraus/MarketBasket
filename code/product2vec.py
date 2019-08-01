import pandas as pd
import gensim

path_train = "./data/order_products__train.csv"
path_prior = "./data/order_products__prior.csv"
path_products = "./data/products.csv"

train_orders = pd.read_csv(path_train)
prior_orders = pd.read_csv(path_prior)
products = pd.read_csv(path_products)

#Turn the product ID to a string
#This is necessary because Gensim's Word2Vec expects sentences, so we have to resort to this dirty workaround
train_orders["product_id"] = train_orders["product_id"].astype(str)
prior_orders["product_id"] = prior_orders["product_id"].astype(str)

train_products = train_orders.groupby("order_id").apply(lambda order: order['product_id'].tolist())
prior_products = prior_orders.groupby("order_id").apply(lambda order: order['product_id'].tolist())

#Create the final sentences
sentences = prior_products.append(train_products).values

#Train Word2Vec model
model = gensim.models.Word2Vec(sentences, size=50, window=5, min_count=50, workers=4)

model.save("product2vec.model")
model.wv.save_word2vec_format("product2vec.model.bin", binary=True)