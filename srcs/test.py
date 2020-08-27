import pickle
import pandas as pd
from utility import ft_count_pairs


with open("data/lda/corps.pkl", "rb") as f:
    docs = pickle.load(f)
    print(type(docs))

with open("data/df.pkl", "rb") as f:
    df = pickle.load(f)
    print(type(df))

print(df.head())

# with open("data/pmi/pair_count.pkl", 'rb') as f:
#     data = pickle.load(f)
#     print(data)
#     print(data.shape)