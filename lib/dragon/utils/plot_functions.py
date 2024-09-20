import pickle
import io
import torch
import graphviz

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)
        
def load_archi(path):
    with open(path, "rb") as f:
        model = CPU_Unpickler(f).load()
    return model

def str_operations(operations):
    n = []
    for op in operations:
        n_op = [op.combiner]
        n_op.append(str(op.name)[:-2].split('.')[-1])
        hp = op.hp
        for k,v in hp.items():
            n_op.append(v)
        n_op.append(op.activation._get_name())
        n.append(n_op)
    return n

def l_2_s(l):
    return ','.join([str(it) for it in l])


def draw_cell(graph, nodes, matrix, color, list_nodes, name_input=None, color_input=None):
    if name_input is not None:
        if isinstance(name_input, list):
            nodes[0] = name_input
        else:
            nodes[0] = [name_input]
    if color_input is None:
        color_input = color
    nodes = [l_2_s(l) for l in nodes]
    for i in range(len(nodes)):
        if nodes[i] in list_nodes:
            j = 1
            nodes[i] += " " + str(j)
            while nodes[i] in list_nodes:
                j += 1
                nodes[i] = nodes[i][:-2] + " " + str(j)
        list_nodes.append(nodes[i])

    graph.node(nodes[0], style="rounded,filled", color="black", fillcolor=color_input, fontcolor="#ECECEC")
    for i in range(1, len(nodes)):
        graph.node(nodes[i], style="rounded,filled", color="black", fillcolor=color)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i][j] == 1:
                graph.edge(nodes[i], nodes[j])
    return graph, list_nodes


def draw_graph(n_2d, m_2d, n_1d, m_1d, output_file, act="Identity()", name="Input Features", freq=48):
    G = graphviz.Digraph(output_file, format='pdf',
                            node_attr={'nodesep': '0.02', 'shape': 'box', 'rankstep': '0.02', 'fontsize': '20', "fontname": "sans-serif"})

    G, g_nodes = draw_cell(G, n_2d, m_2d, "#ffa600", [], name_input=name, color_input="#7a5195")
    G.node("Flatten", style="rounded,filled", color="black", fillcolor="#CE1C4E", fontcolor="#ECECEC")
    G.edge(g_nodes[-1], "Flatten")

    G, g_nodes = draw_cell(G, n_1d, m_1d, "#ffa600", g_nodes, name_input=["Flatten"],
                            color_input="#ef5675")
    G.node(','.join(["MLP", str(freq), act]), style="rounded,filled", color="black", fillcolor="#ef5675", fontcolor="#ECECEC")
    G.edge(g_nodes[-1], ','.join(["MLP", str(freq), act]))
    return G

def get_name_features(features, config):
    f = []
    for i, value in enumerate(features):
        if value>0:
            f.append(config['Features'][i])
    return f