from streamlit_force_graph_simulator import st_graph, ForceGraphSimulation
import streamlit as st

import networkx as nx
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib import cm
import ast

import random

# Use the non-interactive Agg backend, which is recommended as a
# thread-safe backend.
# See https://matplotlib.org/3.3.2/faq/howto_faq.html#working-with-threads.
import matplotlib as mpl
mpl.use("agg")

##############################################################################
# Workaround for the limited multi-threading support in matplotlib.
# Per the docs, we will avoid using `matplotlib.pyplot` for figures:
# https://matplotlib.org/3.3.2/faq/howto_faq.html#how-to-use-matplotlib-in-a-web-application-server.
# Moreover, we will guard all operations on the figure instances by the
# class-level lock in the Agg backend.
##############################################################################
from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock

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

st.markdown("""The D3 javascript package is amazing for creating dynamic force-directed graph layouts. Unfortunately,
doing so is difficult without knowledge of javascript, and these graphs aren't easy to generate from python. Fortunately,
streamlit handles the connection between python and javascript automatically, and the `streamlit-force-graph-simulator`
component makes it easy to create force-directed visualizations (and simulations) from python.
""".replace("\n",' '))

st.markdown("""Documentation can be found at [here](https://github.com/erikweis/streamlit-force-graph-simulator). Code for this
demo is available from the menu bar in the top right.""".replace("\n",' '))

st.header("Example Simulations")

######## Adaptive Voter Model
st.subheader("Adaptive Voter Model")

st.markdown("""
With the traditional voter model, a random node takes the state of one of its neighbors at each time step. With the adaptive voter model
nodes rewire an edge from a node with an opposite opinion, with probability p. Otherwise, nodes take the state of a neighbor, with probability 1-p.
""".replace("\n",' '))


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

st.subheader("Random Walk Model")


st.markdown("""
The random walk model can be thought of as an extension of preferential attachment model with higher clustering.
At each time step, a node is added to the network and attached to a randomly chosen node $j$. Then, the new node's friends
are also created by adding links to each neighbor of $j$ with probability $p$. Implicitly,
this process induces preferential attachment, because nodes with high degree are more likely to be
connected to the randomly chosen node $j$.
""".replace("\n",''))

col1,col2 = st.columns((2,3))

with col1:
    st.write('Controls')
    st.slider("Probability of Attaching to Friends of $j$",min_value=0.0,max_value=1.0,value=0.8,key='prob_link_rwm')
    st.slider("Number of Initial Friends",min_value=2,max_value=6,key='num_initial_friends')

#seed graph
num_seed_nodes = 5
G = nx.complete_graph(num_seed_nodes)
F_pa = ForceGraphSimulation(G)

# add nodes to graph according to preferential attachment
for i in range(num_seed_nodes,100+num_seed_nodes):

    G = F_pa.graph

    # add node to network
    edge_color = 'rgba(220,220,220,0.4)'

    i = len(G.nodes)
    j = random.choice(list(G.nodes))
    F_pa.add_node(i)
    F_pa.add_edge(i,j,color=edge_color)
    for neighbor in random.sample(list(G[j]),st.session_state.num_initial_friends):
        if random.random() < st.session_state.prob_link_rwm:
            F_pa.add_edge(i,random.choice(list(G[j])),color=edge_color)
        else:
            F_pa.add_edge(i,random.choice(list(G.nodes)),color=edge_color)
    
    # set node color
    degrees = list(G.degree)
    max_deg = max(degrees,key=lambda x:x[1])[1]
    cmap = cm.get_cmap('viridis',50)

    for node, degree in degrees:
        
        color = list(cmap(degree/max_deg))
        color = [int(np.floor(256*c)) if i < 3 else c for i,c in enumerate(color)]

        F_pa.set_node_attributes(node,color='rgba({},{},{},{})'.format(*color))

    F_pa.save_event()


with col2:
    degrees = [item[1] for item in F_pa.graph.degree()]
    fig, ax = plt.subplots()
    ax.hist(degrees,bins='auto')
    ax.set_title('Degree Distribution')
    st.pyplot(fig)


st_graph(
        F_pa.initial_graph_json, 
        F_pa.events,
        time_interval=100,
        graphprops = {
            'height':500,
            'linkColor':'color',
            'nodeColor':'color'
        },
        key="graph_pa"
    )

######### Static Graphs ###########

st.header('Visualizing Static Graphs')

st.markdown("""We can also use the streamlit force graph component to visualize elaborate networks as force-directed graphs.
This can be done by not passing an events list to the component.""".replace("\n"," "))

st.markdown("""This graph shows trust ratings between users on Bitcoin OTC. 
The edges are signed and weighted, so positive trust ratings are shown in blue, while negative trust ratings are 
shown in red. Edges are also weighted according to the strength of the rating, which took a value between $[-10,10]$.""".replace("\n",' '))

with open('bitcoinc.json','r') as f:
    jg = json.load(f)

st_graph(
    jg,
    graphprops = {
        'height':600,
        'linkWidth':'linkWidth',
        'linkColor':'color',
        'nodeRelSize':6,
        },
    key='graph_static'
)

###### Stochastic Block Model ##################

st.header('Stochastic Block Model')

def callback_sbm():

    try:
        sizes = ast.literal_eval(st.session_state.sbm_sizes)
        matrix = ast.literal_eval(st.session_state.connection_matrix)
        sbm_nx = nx.stochastic_block_model(sizes,matrix)
        sbm = nx.readwrite.json_graph.node_link_data(sbm_nx)
        sbm['graph'].pop('partition')
        st.session_state.sbm_graph = sbm
    except:
        pass


sizes = [25,25,25]
matrix =   [[0.5, 0.05, 0.05], [0.05, 0.5, 0.05], [0.05, 0.05, 0.5]]
sbm_nx = nx.stochastic_block_model(sizes,matrix)
sbm = nx.readwrite.json_graph.node_link_data(sbm_nx)

# label
sbm['graph'].pop('partition') #excluse partition data because it's not json serializable and not useful

if 'sbm_graph' not in st.session_state:
    st.session_state.sbm_graph = sbm

st.text_input('Sizes',sizes,key='sbm_sizes',on_change=callback_sbm)
st.text_area('Edge Density Matrix Matrix',str(matrix).replace('],','],\n'),key='connection_matrix',on_change=callback_sbm)

st_graph(
    st.session_state.sbm_graph,
    graphprops = {'height':500,'nodeAutoColorBy':'block'},
    key='sbm_graph'
)

############## Continuous Play ################

st.header('Continuous Play')

st.markdown("""We can also visualize directed graphs, or ask the simulation to play continuously.
The code for this simple visualization is shown below.""".replace("\n",' '))

with st.echo(code_location='below'):
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

    st_graph(
        F.initial_graph_json,
        F.events,
        time_interval = 1000,
        graphprops=props,
        continuous_play = True,
        directed = True,
        key='my_graph'
    )



