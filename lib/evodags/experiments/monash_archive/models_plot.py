import graphviz
from lib.evodags.utils.plot_functions import draw_cell


def gluonts_nn_plot(n, m, act, output_file, out, name_dataset):
    G = graphviz.Digraph(output_file, format='png',
                         node_attr={'nodesep': '0.02', 'shape': 'box', 'rankstep': '0.02'})

    G, g_nodes = draw_cell(G, n, m, "sandybrown", [], name_input=name_dataset, color_input="darkorange")
    G.node(','.join(["MLP", str(out), act]), style="filled", color="yellowgreen")
    G.edge(g_nodes[-1], ','.join(["MLP", str(out), act]))
    G.view()