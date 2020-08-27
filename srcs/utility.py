import pickle
import squarify
import pyodbc
import statistics

import pandas as pd
import plotly.graph_objs as go

from math import log
from itertools import combinations
from collections import defaultdict, Counter

"""
includes:
----------

- methods used for setting-up the necessary files
- method used for pmi calculation of word pairs and topics
- methods solving minor calculations/issues
- method creating treemap figure
"""


def ft_setup_pickle(path):
    """Unpickles all the necessary files found in the specified path."""
    id2word = pd.read_pickle(path + 'lda/id2word.pkl')
    lda_topics = pd.read_pickle(path + 'lda/lda_topics.pkl')
    lda_docs = pd.read_pickle(path + 'lda/lda_docs.pkl')
    word_count = pd.read_pickle(path + 'pmi/word_count.pkl')
    pair_count = pd.read_pickle(path + 'pmi/pair_count.pkl')
    df = pd.read_pickle(path + 'df.pkl')
    return id2word, lda_topics, lda_docs, word_count, pair_count, df


def ft_compute_pmi(word1, word2, word_count, pair_count):
    """
    Computes the pmi score of a word pair.
    The order of words is not important when calling the function.

    Arguments
    ----------
    word1, word2: word pair for which pmi is to be calculated
    word_count: Counter object created with in pmi_calculation.py script
    pair_count: dict with word pairs as keys and their count as value

    Return
    ----------
    pmi score for given word pair
    """
    proba_word1 = word_count[word1] / sum(word_count.values())
    proba_word2 = word_count[word2] / sum(word_count.values())
    # we use setdefault to avoid KeyErrors if the key is not in the dictionary
    combined_proba = (
        pair_count.setdefault((word1, word2), 0) +
        pair_count.setdefault((word2, word1), 0)) /\
        sum(pair_count.values())
    try:
        return log(float(combined_proba) / float(proba_word1 * proba_word2), 2)
    except BaseException:
        return 0


def ft_compute_median_pmi(path, dropdown_topic_list, n_topic_pmi):
    """
    Computes the pmi score for topics in dropdown list.
    Takes the stored pmi scores and computes their median.
    Return a dict with {<topicnbr>: <pmiscore>...}
    """
    dict_pmi = dict()
    for i in dropdown_topic_list:
        df_pmi = pd.read_pickle(path + 'pmi/df_pmi_' + str(i) + '.pkl')
        list_pmi = list()
        line_start = 1
        for column in range(n_topic_pmi):
            for row in range(line_start, n_topic_pmi):
                list_pmi.append(df_pmi.iloc[row, column])
            line_start += 1
        dict_pmi[i] = statistics.median(list_pmi)
    return dict_pmi


def ft_count_pairs(docs):
    """Generates both a word and pair count using docs to generate pairs."""
    word_count = Counter()
    pair_count = dict()
    for doc in docs:
        # update the global word counter/store local count of words
        word_count.update(doc)
        docUnigrams = Counter(doc)
        # using unique tokens to form pairs
        # we don't count pairs where word 1 = word 2
        uniqueTokens = list(set(doc))
        pairs = list(combinations(uniqueTokens, 2))
        for pair in pairs:
            # simply counting occurence of pairs with a counter doesn't work
            # (word1, word2, word2) would be counted as two even though the
            # pair appears together one time
            count = min(docUnigrams[pair[0]], docUnigrams[pair[1]])
            try:
                pair_count[pair[0], pair[1]] += count
            except BaseException:
                pair_count[pair[0], pair[1]] = count
    print('done counting words/word pairs')
    return word_count, pair_count


def ft_generate_topic_dropdown(lda_topics):
    """
    Generates a list of dictionaries mapping their index to
    the name displayed in the drop dropdown.
    """
    topic_dropdown_labels = list()
    for i in range(0, lda_topics.shape[0]):
        label = dict()
        label['label'] = 'Topic ' + str(i)
        label['value'] = i
        topic_dropdown_labels.append(label)
    return topic_dropdown_labels


def ft_generate_center(topics_amount):
    """
    Generates a list of lists used to set the center for each layout.
    Displays topics in rows of 5.
    """
    x = 0
    y = 0
    center_list = [[x, y]]
    for i in range(topics_amount - 1):
        if ((i + 1) % 5 == 0):
            x = 0
            y -= 5
        else:
            x += 5
        center_list.append([x, y])
    return center_list


def ft_treemap_figure(percent_dict, percent_tuples):
    """
    This method creates the figure attribute for the treemap.
    Uses squarify to compute the necessary rectangles.

    Arguments
    ----------
    percent_dict: {topic nbr:sum(topic probability over all docs)}
    percent_tuples: same as percent_dict but under list of tuples

    Return
    ----------
    figure: plotly graph object which results in a treemap when ploted
    """
    x = 0.
    y = 0.
    width = 4.
    height = 4.

    values = list(percent_dict.values())
    values.sort(reverse=True)
    normed = squarify.normalize_sizes(values, width, height)
    rects = squarify.squarify(normed, x, y, width, height)

    # color_brewer = ['#800026', '#bd0026', '#e31a1c', '#fc4e2a',
    #                 '#fd8d3c', '#feb24c', '#fed976', '#ffeda0', '#ffffcc']

    shapes = []
    annotations = []
    # counter = 0
    topic_counter = 0

    for r in rects:
        shapes.append(
            dict(
                type='rect',
                x0=r['x'],
                y0=r['y'],
                x1=r['x'] + r['dx'],
                y1=r['y'] + r['dy'],
                line=dict(width=2),
                # fillcolor=color_brewer[counter]
            )
        )
        annotations.append(
            dict(
                x=r['x'] + (r['dx'] / 2),
                y=r['y'] + (r['dy'] / 2),
                text=percent_tuples[topic_counter][0],
                showarrow=False
            )
        )
        # counter += 1
        topic_counter += 1
        # if counter >= len(color_brewer):
        #     counter = 0

    figure = {
        'data': go.Scatter(
            x=[r['x'] + (r['dx'] / 2) for r in rects],
            y=[r['y'] + (r['dy'] / 2) for r in rects],
            text=[ptuple[0] for ptuple in percent_tuples],
            mode='text',
        ),

        'layout': go.Layout(
            height=700,
            width=700,
            xaxis={
                'fixedrange': True,
                'showgrid': False,
                'zeroline': False,
                'showticklabels': False
            },
            yaxis={
                'fixedrange': True,
                'showgrid': False,
                'zeroline': False,
                'showticklabels': False
            },
            shapes=shapes,
            annotations=annotations,
            hovermode='closest',)
    }

    return figure


def sql_query(query):
    '''Create a dataframe from a query to the Patstat database.'''

    # Connection
    server = 'server_ip'
    database = 'db_name'
    username = 'paozer'
    password = 'password'
    driver = '{ODBC Driver 17 for SQL Server}' # Ggf. anpassen, z.B. 17 statt 13
    cnxn = pyodbc.connect('DRIVER='+driver+';SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password)

    # Create dataframe from the query
    df = pd.read_sql(query, cnxn)
    return df
