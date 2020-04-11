# -*- coding: utf-8 -*-
import io
import requests
import pickle
import gzip
import copy 
import math
from collections import defaultdict
import networkx as nx

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

def dump(fname, obj):
    with gzip.open(fname, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(url):
    response = requests.get(url)
    gzip_file = io.BytesIO(response.content)
    with gzip.open(gzip_file, 'rb') as f:
        return pickle.load(f)

def load_pickle_0(filename):
    with gzip.open(filename, 'rb') as f:
        return pickle.load(f)

references = load_pickle("https://storage.googleapis.com/mlstudy-phys/20200403_test/references_20200404.pkl.gz")
results = load_pickle("https://storage.googleapis.com/mlstudy-phys/20200403_test/results.pkl.gz")
df = load_pickle("https://storage.googleapis.com/mlstudy-phys/20200403_test/df_20200410.pkl.gz")

option_list = ["Candidate claim", "Without element A", "Without element B", "Without element C", "Without element D"]
preamble = "A health monitoring system for a plurality of satellites. "
text_A = "The system receives telemetry data from the plurality of satellites. The data is measured by the sensor which mounted on the satellite and received by the system on the ground. "
text_B = "The data is in a format specific to each satellite. The system converts the data to a standard format. The standard format of the data is independent of the hardware configuration of the satellite. "
text_C = "The system displays the data in the standard format. The system displays the data in a visualization format that is intuitive to understand for a human operator. The system displays the data via a web browser. "
text_D = "The health of the plurality of satellites is judged baesd on machine larning algorithms . When fault is detected, an alarm will be noticed to the operator. "
sentence_list = [preamble, text_A, text_B, text_C, text_D]

node_dict=defaultdict(int)
tmp_list = []
i = 1

for num in df["id"][:-5]:
    if not node_dict[num]:
        node_dict[num] = i
        dic = references[num]
        i += 1

    num_list = [url.split(sep="/")[-2] for url in dic["ForwardReferences"]]
    for n in num_list:
        if not node_dict[n]:
            if n in df["id"].values:
                node_dict[n] = i
                i += 1
                tmp_list.append( (node_dict[num], node_dict[n]) )

i += 1
target = i

num_to_id_dict = {v:k for k,v in node_dict.items()}

app.layout = html.Div([
    html.H1("Patent citation network"),
     html.Div([
         html.P("Enter the number of references you want the link to appear in."),
         html.P("If you change the number, the network diagram will be updated."),
         dcc.Input(
                id="top_N",
                type="number",
                value=10,
                placeholder="input number",
            ),
        dcc.Graph(id='graph')
        ],style={'width': '48%', 'display': 'inline-block'}),

    html.Div([
         html.P("Select the sentence you want to search."),
         dcc.Dropdown(
                id="target_sentence",
                options=[{'label': t, 'value': i} for i,t in enumerate(option_list)],
                value=0
            ),
         html.Blockquote(id='text_output', style={'backgroundColor':"#DCDCDC"}),
         html.H2("Selected references"),
         html.P(["Shift+click will accumulate the selected reference. You can also use ", html.I("Box Select "), "or ", html.I("Lasso Select "), "to select multiple references."]),
        #  html.Table([
        #      html.Tr([html.Th(["URL", "assignee", "priority_date", "abstract"])]),
        #      html.Tr([html.Td(id='table')])
        #      ]),
         html.Div([html.Div(id='table')])
        ],style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
])

@app.callback(
    Output(component_id='graph', component_property='figure'),
    [Input(component_id='top_N', component_property='value'),
    Input(component_id='target_sentence', component_property='value')])
def update_figure(top_N, target_sentence):
    edge_list = copy.copy(tmp_list)
    sorted_result = copy.copy(results[target_sentence])

    j = 0
    for m in range(50):
        n = node_dict[sorted_result[m]]
        if n > 0:
            edge_list.append( (i, n) )
            j += 1
        if j >= top_N:
            break

    G = nx.Graph()
    G.add_edges_from(edge_list)
    pos = nx.spring_layout(G, k=1.2/math.log2(len(edge_list)), pos={target:(0,0)}, fixed=[target] )

    for node in G.nodes:
        G.nodes[node]['pos'] = pos[node]

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='Portland',
            reversescale=True,
            color=[],
            size=[],
            colorbar=dict(
                thickness=15,
                title='Forward Citation Count',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    forward_citation_count = []
    node_text = []
    for node in G.nodes():
        if node == target:
            forward_citation_count.append(10)
            node_text.append(option_list[target_sentence])
            continue
        forward_citation_count.append(len(references[num_to_id_dict[node]]["ForwardReferences"] ))
        text = "{}".format(num_to_id_dict[node]) +", " + "{}".format(df[df["id"]==num_to_id_dict[node]]["assignee"].values[0]) 
        node_text.append(text)


    node_trace.marker.color = forward_citation_count
    node_trace.marker.size = [min(5 + (elm)**0.6, 30) for elm in forward_citation_count]
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                # title='Patent citation network',
                # titlefont_size=20,
                showlegend=True,
                hovermode='closest',
                clickmode='event+select',
                margin=dict(b=20,l=5,r=5,t=40),
                # xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                # yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                # width=1000, 
                height=800
                )
                )

    annotations = []
    X, Y = G.nodes[target]['pos']
    theta = math.atan2(Y, X)
    r = max([(X**2 + Y**2)**0.5, 0.1])
    d = dict(
        x=X,
        y=Y,
        text=option_list[target_sentence],
        showarrow=True,
        arrowhead=7,
        ax=10*math.cos(theta) * r**-1,
        ay=-10*math.sin(theta) * r**-1)
    annotations.append(d)

    fig.update_layout(
        showlegend=False,
        annotations=annotations)

    return fig

@app.callback(
    Output(component_id='text_output', component_property='children'),
    [Input(component_id='target_sentence', component_property='value')]
)
def update_output_div(input_value):
    if input_value == 0:
        return [html.P(s) for s in sentence_list]
    else:
        return [html.P(s) for i, s in enumerate(sentence_list) if i != input_value]

@app.callback(
    [Output(component_id='table', component_property='children')],
    [Input('graph', 'selectedData')])
def display_click_data(selectedData):
    if selectedData is not None:
        num_list = [elm["text"].split(sep=",")[0] for elm in selectedData["points"]]
        url_list = ["https://patents.google.com//patent/" + num + "/en" for num in num_list]
        # out_url = dcc.Link(num_list, href=url_list, title="If you don't get a response, try CTRL+click")
        # out_table = dash_table.DataTable(
        #      style_data={
        #         'whiteSpace': 'normal',
        #         'height': 'auto'},
        #      columns=[{"name": i, "id": i} for i in ["assignee", "priority_date", "abstract"]],
        #      data=df[df["id"] == num][["assignee", "priority_date", "abstract"]].to_dict('records')
        #     )

        tmp = [html.Tr([html.Th(elm) for elm in ["URL", "assignee", "priority_date", "abstract"]])]
        for num, url in zip(num_list, url_list):
            elm_list = [url] + df[df["id"] == num][["assignee", "priority_date", "abstract"]].values[0].tolist()
            tmp.append(html.Tr([html.Td(html.A(elm.split(sep="/")[-2], href=elm, target="_blank")) if i == 0 else html.Td(elm) for i, elm in enumerate(elm_list)]))
        out_table = html.Table(tmp)
        return [out_table]
    else:
        return [html.Tr([html.Td("") for _ in range(4)])]

if __name__ == '__main__':
    app.run_server(debug=True)
