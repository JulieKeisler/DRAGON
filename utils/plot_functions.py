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
            nodes[i]+=" "+str(j)
            while nodes[i] in list_nodes:
                j+=1
                nodes[i] = nodes[i][:-2]+" "+str(j)
        list_nodes.append(nodes[i])

    graph.node(nodes[0], style = "filled", color=color_input)
    for i in range(1, len(nodes)):
        graph.node(nodes[i], style = "filled", color=color)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i][j] == 1:
                graph.edge(nodes[i], nodes[j])
    return graph, list_nodes