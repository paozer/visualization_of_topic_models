import os
import utility
import pickle
import pandas as pd
from itertools import combinations

"""
This script is used to pre-compute the PMI scores for the
n most probable words for all topics and storing the score in
dataframes (one per topic).
"""
dirpath = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(dirpath, 'data/')

n = 20

id2word, lda_topics, docs, word_count, pair_count =\
    utility.ft_setup_pickle(path)
word_count, pair_count = utility.ft_count_pairs(docs)

df_visualisation = pd.DataFrame(
    data=lda_topics.transpose(),
    index=id2word.values(),
    columns=range(lda_topics.shape[0]),
)

for topic in range(lda_topics.shape[0]):

    df = df_visualisation.nlargest(n, topic)
    top_words = list(df.index.values)
    df_pmi = pd.DataFrame(
        index=top_words,
        columns=top_words,
    )

    word_pairs = list(combinations(top_words, 2))

    for word_pair in word_pairs:
        pmi_score = utility.ft_compute_pmi(
            word_pair[0], word_pair[1], word_count, pair_count)
        df_pmi[word_pair[0]][word_pair[1]] = pmi_score
        df_pmi[word_pair[1]][word_pair[0]] = pmi_score

    with open(path + 'pmi/' + 'df_pmi_' + str(topic) + '.pkl', 'wb') as f:
        pickle.dump(df_pmi, f)

    print('topic {} done'.format(topic))
