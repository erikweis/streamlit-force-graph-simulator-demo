from networkx.classes import graph
from streamlit_force_graph_simulator import st_graph, ForceGraphSimulation
import streamlit as st

import networkx as nx
import numpy as np
from networkx.readwrite import json_graph

import random
from collections import Counter
import matplotlib.pyplot as plt


###################### functions ###########################

def initialize_avm_simulation():

    #initialize graph   
    G = nx.les_miserables_graph()
    for node in G.nodes:
        G.nodes[node]['color'] = random.choice(['red','blue','green'])

    #create simulation object
    return ForceGraphSimulation(G)

def run_avm_simulation(F):

    for _ in range(1000):

        #new event
        F.save_event()

        #choose node to act on
        node = random.choice(list(F.graph.nodes))

        #get neighbors
        neighbors = list(F.graph[node])

        #rewire or take a neighbor's state
        if random.random() < st.session_state.probability_rewire:

            #try to rewire a node with opposite opinion
            opp_neighbors = [n for n in neighbors if F.graph.nodes[n]['color'] != F.graph.nodes[node]['color'] ]
            if opp_neighbors:
                old_neighbor = random.choice(opp_neighbors)

                # choices = list(nx.ego_graph(G,node,radius=2,center=False).nodes)
                choices = list(F.graph.nodes)
                choices.remove(node)
                new_neighbor = random.choice(choices)
                F.add_edge(node,new_neighbor)
                F.remove_edge(node,old_neighbor)

        else:
            #take a neighbors state
            if neighbors:
                neighbor = random.choice(neighbors)
                neighbor_color = F.graph.nodes[neighbor]['color']
                if F.graph.nodes[node]['color'] != neighbor_color:
                    F.set_node_attributes(node, color=neighbor_color)
    return F

def callback_avm():
    F = initialize_avm_simulation()
    F = run_avm_simulation(F)
    st.session_state.graph = F.initial_graph_json
    st.session_state.events = F.events

############### Streamlit App ####################

st.title("Streamlit Force Graph Simulator Demo")

text = "".join("""
With the traditional voter model, a random node takes the state of one of its neighbors at each time step. With the adaptive voter model
nodes rewire an edge from a node with an opposite opinion, with probability p. Otherwise, nodes take the state of a neighbor, with probability 1-p.
""".split("\n"))

st.markdown(text)

######## Adaptive Voter Model
st.header("Adaptive Voter Model")

# adaptive voter model simulation
F = initialize_avm_simulation()
if 'graph' not in st.session_state:
    st.session_state.graph = F.initial_graph_json
if 'events' not in st.session_state:
    st.session_state.events = F.events

st.slider(
    'Probability Rewire',
    key='probability_rewire',
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    on_change=callback_avm,
    help='The probability that nodes choose to rewire a connection.'
)

F = run_avm_simulation(F)


graphprops = {'height':400}

st_graph(
    st.session_state.graph, 
    st.session_state.events,
    time_interval=20,
    graphprops = graphprops,
    key="graph"
)

##### Preferential attachment

st.header("Preferential Attachement")

text = "".join("""
With preferential attachement, nodes are added to the network one at a time. And each are 
connected to $n$ new nodes.
""".split("\n"))
st.markdown(text)


num_seed_nodes = 5
st.slider("Number of Attachement Nodes",min_value=1,max_value=num_seed_nodes,key='num_seed_nodes')

#seed graph

G = nx.complete_graph(num_seed_nodes)
F_pa = ForceGraphSimulation(G)

# add nodes to graph according to preferential attachment
for i in range(num_seed_nodes,100+num_seed_nodes):

    G = F_pa.graph
    
    #random choice according to degree
    nodes, degrees = map(lambda x: np.array(x),zip(*list(G.degree())))
    attachment_nodes = np.random.choice(
        nodes,
        p=degrees/np.sum(degrees),
        size=st.session_state.num_seed_nodes
    )

    # add node to network
    F_pa.add_node(i)
    for j in attachment_nodes:
        F_pa.add_edge(i, int(j))
    F_pa.save_event()

st_graph(
    F_pa.initial_graph_json, 
    F_pa.events,
    time_interval=500,
    graphprops = graphprops,
    key="graph_pa"
)


text_test_all_methods = "".join("""
Test all methods.""".split("\n"))

st.markdown(text_test_all_methods)

G = nx.erdos_renyi_graph(5,0.8,directed=True)
F = ForceGraphSimulation(G)

F.add_node(5)
F.add_edge(4,5)
F.add_edge(3,5)
F.save_event()

F.add_node(6)
F.add_edge(5,6)
F.save_event()

F.remove_node(5)
F.save_event()

props = {
    'height':300,
    'cooldownTicks':1000 ,
    'linkDirectionalArrowLength':3.5,
    'linkDirectionalArrowRelPos':1
}

w2 = st_graph(
    F.initial_graph_json,
    F.events,
    time_interval = 1000,
    graphprops=props,
    continuous_play = True,
    directed = True,
    key='my_graph'
)


text_test_new_graph = "".join("""
With the traditional voter model, a random node takes the state of one of its neighbors at each time step. With the adaptive voter model
nodes rewire an edge from a node with an opposite opinion, with probability p. Otherwise, nodes take the state of a neighbor, with probability 1-p.
""".split("\n"))

st.markdown(text_test_new_graph)

G1 = json_graph.node_link_data(nx.barbell_graph(5,1))
events2 = []

for i in range(2,5):
    G_t = json_graph.node_link_data(nx.barbell_graph(5,i))
    events2.append([{'event_type':'new_graph','graph':G_t}])

w2 = st_graph(
    G1,
    events2,
    time_interval = 2000,
    graphprops={ 'height':300 },
    key='graph2'
)

