import utility

import os
import itertools
import pandas as pd
from random import sample

from gensim.matutils import corpus2csc
from scipy.sparse import csc_matrix

import dash
import dash_table
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html

import plotly as py
import plotly.graph_objs as go
import networkx as nx

# MAIN SETUP #
dirpath = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(dirpath, 'data/')

id2word, lda_topics, lda_docs, word_count, pair_count, df\
    = utility.ft_setup_pickle(path)

# dataframe used for visualisation since it allows for easier access
# index: words from vocabulary columns: one per topic values: probabilities
df_visualisation = pd.DataFrame(
    data=lda_topics.transpose(),
    index=id2word.values(),
    columns=range(0, lda_topics.shape[0]),
)

# list of dict used for the dropdown element
list_topic_labels = utility.ft_generate_topic_dropdown(lda_topics)

# transform gensim object to csc matrix & create coordinates for it
# real bottleneck 90s
docs_csc = corpus2csc(lda_docs)
coo = docs_csc.tocoo(copy=False)

df_table = pd.DataFrame({'doc': coo.col, 'topic': coo.row, 'proba': coo.data})
df_table = pd.pivot_table(df_table, 'proba', 'doc', 'topic', fill_value=0)

# build dictionary for later building topic treemap
# not optimized; probably easy work
dict_percent = dict()

for topic in list(df_table.columns):
    dict_percent[topic] = sum(df_table[topic].values)

# use list of tuples to preserve combination
# of key:value while sorting in descending order
tuples_percent = sorted(dict_percent.items(), reverse=True, key=lambda x: x[1])

# VISUALISATION #

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([

    html.Div([

        html.Div([
            html.Label(children='Topic Multi-Select Dropdown'),
            dcc.Dropdown(
                id='dropdown_topic',
                options=list_topic_labels,
                value=sample(range(0, len(list_topic_labels)), 3),
                multi=True
            ),
        ]),

        html.Div([
            html.Label(id='slider_words_indicator'),
            dcc.Slider(
                id='slider_words',
                min=1,
                max=25,
                value=5,
                step=1,
                marks={i: str(i) for i in range(0, 26, 5)},
            ),
        ],
            style={'marginTop': 25},),

        html.Div([
            html.Label(id='range_pmi_indicator'),
            dcc.RangeSlider(
                id='range_pmi',
                min=-5,
                max=5,
                step=0.1,
                marks={i: str(i) for i in range(-6, 6, 1)},
                value=[0.5, 2.5],
            ),
        ],
            style={'marginTop': 25},),

        html.Div([
            html.Label(children='Nb Words for Topic PMI score:'),
            dcc.Input(
                id='top_nb',
                type='number',
                value=5,
                min=2,
                max=20,
                step=1,
            ),
        ],
            style={'marginTop': 25},),

        html.Div([
            html.Label(children='Nb of Documents:'),
            dcc.Input(
                id='doc_nb',
                type='number',
                value=3,
                min=1,
                max=100,
                step=1,
            ),
        ],
            style={'marginTop': 25},),

        html.Div(
            html.Button(
                id='button_display',
                n_clicks=0,
                children='Display visualization',
            ),
            style={'marginTop': 25},),

    ],
        style={'columnCount': 2},),

    html.Div([dcc.Graph(id='graph')]),

    html.Div([

        html.Div(
            dash_table.DataTable(
                id='table',
                sort_action='native',
                row_selectable='single',
                hidden_columns=[],
                style_header={
                    'backgroundColor': 'white',
                    'fontWeight': 'bold'
                },
                style_table={
                    'overflowY': 'scroll',
                    'border': 'thin lightgrey solid'
                },
                fixed_rows={'headers': True, 'data': 0},
            ),
        ),
        html.Div(dcc.Markdown(id='title_md')),
        html.Div(dcc.Markdown(id='abstract_md'))
        ],
        style={'columnCount': 2},),

    html.Div(
        dcc.Graph(
            id='treemap',
            figure=utility.ft_treemap_figure(dict_percent, tuples_percent),
            config={'displayModeBar': False},
        ),
    ),
])

# CALLBACK METHODS #


# main callback for building graph
@app.callback(
    Output('graph', 'figure'),
    [Input('button_display', 'n_clicks')],
    [State('range_pmi', 'value'),
     State('slider_words', 'value'),
     State('dropdown_topic', 'value'),
     State('top_nb', 'value'),])
def update_graph(n_clicks, range_pmi, n_slider_words, dropdown_topic_list, n_topic_pmi):
    global_node_trace, global_edge_trace = [], []
    dropdown_topic_list.sort()
    center_list = utility.ft_generate_center(len(dropdown_topic_list))

    G = nx.Graph()

    for topic in dropdown_topic_list:

        # create dataframe with n(_slider_words) most probable words (index)
        # for each topic (column)
        df_probabilities = df_visualisation.nlargest(n_slider_words, topic)

        # create nodes from words
        node_list = list(df_probabilities.index.values)
        G.add_nodes_from(node_list)

        edge_list = []
        possible_edge_list = list(itertools.combinations(node_list, 2))

        # PMI Retrieval/Calculations
        df_pmi = pd.read_pickle(path + 'pmi/df_pmi_' + str(topic) + '.pkl')

        if n_slider_words <= df_pmi.shape[0]:

            for edge in possible_edge_list:

                pmi_score = df_pmi[edge[0]][edge[1]]
                pmi_score = round(pmi_score, 2)

                index = possible_edge_list.index(edge)

                if pmi_score >= range_pmi[0] and pmi_score <= range_pmi[1]:
                    edge_list.append(
                        (possible_edge_list[index][0],
                         possible_edge_list[index][1],
                         pmi_score))

                else:
                    pass

        else:

            for edge in possible_edge_list:

                if (edge[0] and edge[1]) in df_pmi.index.values:

                    pmi_score = df_pmi[edge[0]][edge[1]]
                    pmi_score = round(pmi_score, 2)

                else:

                    pmi_score = utility.ft_compute_pmi(
                        edge[0], edge[1], word_count, pair_count)
                    pmi_score = round(pmi_score, 2)

                index = possible_edge_list.index(edge)

                if pmi_score >= range_pmi[0] and pmi_score <= range_pmi[1]:

                    edge_list.append(
                        (possible_edge_list[index][0],
                         possible_edge_list[index][1],
                         pmi_score))

                else:
                    pass

        G.add_weighted_edges_from(edge_list)

        # set node positions using spring layout
        c = center_list[dropdown_topic_list.index(topic)]
        pos = nx.spring_layout(node_list, center=c)

        for node in node_list:
            G.node[node]['pos'] = pos[node]

        # create edge trace:
        edge_trace = go.Scatter(
            x=[],
            y=[],
            text=[],
            line={
                'width': 2,
                'color': '#888'},
            mode='lines',
            showlegend=False,)

        for edge in edge_list:

            x0, y0 = G.node[edge[0]]['pos']
            x1, y1 = G.node[edge[1]]['pos']

            edge_trace['x'] += tuple([x0, x1, None])
            edge_trace['y'] += tuple([y0, y1, None])

        # create node trace
        node_trace = go.Scatter(
            x=[],
            y=[],
            text=[],
            name='Topic ' + str(topic),
            mode='markers+text',
            marker=dict(
                symbol='circle',
                size=35,),
            hoverinfo='text',
            hovertext=[])

        for node in node_list:

            node_trace['text'] += tuple([str(node)])
            x, y = G.node[node]['pos']
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])

        global_node_trace.append(node_trace)
        global_edge_trace.append(edge_trace)

    plot_weights = []
    for edge in G.edges.data('weight'):

        x0, y0 = G.node[edge[0]]['pos']
        x1, y1 = G.node[edge[1]]['pos']

        ax = (x0 + x1) / 2
        ay = (y0 + y1) / 2

        plot_weights.append((edge[2], ax, ay))

    annotations_list = [
        dict(
            x=plot_weight[1],
            y=plot_weight[2],
            xref='x',
            yref='y',
            text=plot_weight[0],
            ax=plot_weight[1],
            ay=plot_weight[2],)
        for plot_weight in plot_weights]

    # get dictionary with topic pmi score values
    dict_pmi = utility.ft_compute_median_pmi(path, dropdown_topic_list, n_topic_pmi)

    # add the topic pmi scores to the annotations list
    for topic in dropdown_topic_list:
        annotations_list.append(
            dict(
                x=(center_list[dropdown_topic_list.index(topic)][0] - 1.15),
                y=(center_list[dropdown_topic_list.index(topic)][1] + 1.15),
                xref='x',
                yref='y',
                text=round(dict_pmi[topic], 2),
                ax=(center_list[dropdown_topic_list.index(topic)][0] - 1.15),
                ay=(center_list[dropdown_topic_list.index(topic)][1] + 1.15),
            )
        )

    figure = {
        "data": [
            *global_edge_trace,
            *global_node_trace],
        "layout": go.Layout(
            title='',
            showlegend=True,
            hovermode='closest',
            annotations=annotations_list,
            margin={
                'b': 5,
                'l': 5,
                'r': 5,
                't': 40},
            xaxis={
                'showgrid': False,
                'zeroline': False,
                'showticklabels': False},
            yaxis={
                'showgrid': False,
                'zeroline': False,
                'showticklabels': False},)}

    return figure


# main callback for building table
@app.callback(
    [Output('table', 'columns'),
     Output('table', 'data')],
    [Input('button_display', 'n_clicks'),
     Input('doc_nb', 'value')],
    [State('dropdown_topic', 'value')])
def update_table(n_clicks, doc_nb, dropdown_topic_list):
    """
    This function is called everytime the display button is clicked.
    ----------
    For every topic that is currently in the dropdown we get the n documents
    which are the most composed of said topic. We then use the row index of
    every document to get a row full of information from the original
    dataframe (df). We append the information row to a temporary dataframe
    and use it to build our table.
    """
    docs_in_table = list()
    rows = list()
    n = doc_nb

    for topic in dropdown_topic_list:
        for index in df_table.nlargest(n, topic).index:
            if index in docs_in_table:
                # if doc is already in table associated to another topic
                # append topic nbr to topic column at appropriate row
                patent_id = df.iloc[index]['patent_id']
                for row in rows:
                    if row['patent_id'] == patent_id:
                        row['topic'].append(topic)
            else:
                # if doc is new to the table
                # get row from df with same index and append to rows list
                docs_in_table.append(index)
                row = df.iloc[index].to_dict()
                row['topic'] = [topic]
                rows.append(row)

    temp_df = pd.DataFrame(rows)
    del temp_df['detailed']
    del temp_df['docdb_family_id']
    del temp_df['patent_id']

    data = temp_df.to_dict('rows')
    columns = [{'name': i, 'id': i} for i in temp_df.columns]
    return columns, data


@app.callback(
    [Output('title_md', 'children'),
     Output('abstract_md', 'children')],
    [Input('table', 'selected_rows')],
    [State('table', 'data')])
def update_markdown(selected_row_index, table_data):
    if selected_row_index is None:
        return 'Select a row to display information.', ''
    id = selected_row_index[0]
    appln_id = table_data[id]['appln_id']

    query_title = '''
        SELECT      ti.appln_title
        FROM        dbo.tls202_appln_title ti
        WHERE       ti.appln_id = {}
    '''.format(appln_id)
    df_title = utility.sql_query(query_title)

    query_abstract = '''
        SELECT      ab.appln_abstract
        FROM        dbo.tls203_appln_abstr ab
        WHERE       ab.appln_id = {}
    '''.format(appln_id)
    df_abstract = utility.sql_query(query_abstract)

    return df_title.iloc[0]['appln_title'], df_abstract.iloc[0]['appln_abstract']


# minor callbacks managing settings box
@app.callback(
    Output('slider_words_indicator', 'children'),
    [Input('slider_words', 'value')],)
def setCurrentWords(n_slider_words):
    return u'Word Slider/Current Value: {}'.format(n_slider_words)


@app.callback(
    Output('range_pmi_indicator', 'children'),
    [Input('range_pmi', 'value')],)
def setCurrentWords(range_pmi):
    return u'PMI Slider/Current Range: {}'.format(range_pmi)


if __name__ == '__main__':
    app.run_server(debug=False)
