import pickle
import io
import torch
import graphviz
from sympy import Symbol, Add, Mul, Pow
from dragon.search_space.dag_encoding import AdjMatrix, SymbolicNode, fill_adj_matrix
from dragon.search_space.bricks.basics import Identity
from dragon.search_space.bricks.symbolic_regression import Negate, Inverse, SelectFeatures, ConstantBrick
import torch.nn as nn
import numpy as np
from sympy import Integer, Rational, Float


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

def op_tensors(inputs, combiner):
    if combiner == "add":
        op = "+"
        neutral = "0"
    elif combiner == "mul":
        op = "*"
        neutral = "1"
    elif combiner == "sub":
        op = "-"
        neutral = "0"
    elif combiner == "divide":
        op = "/"
        neutral = "1"
    elif combiner == "concat":
        # concat = concaténation des listes
        out = []
        for l in inputs:
            out.extend(l)
        return out
    else:
        raise ValueError(combiner)

    n = max(len(l) for l in inputs)

    # alignement à droite
    padded = []
    for l in inputs:
        pad = [neutral] * (n - len(l))
        padded.append(pad + l)

    result = []
    for j in range(n):
        expr = padded[0][j]
        for i in range(1, len(padded)):
            expr = f"({expr}) {op} ({padded[i][j]})"
        result.append(expr)

    return result

def apply_operation(out, node):
    op = node.operation
    name = op.__class__.__name__

    if name == "Identity":
        return out

    if name == "Negate":
        return [f"-({x})" for x in out]

    if name == "Inverse":
        return [f"1/({x})" for x in out]

    if name == "SumFeatures":
        if len(out) == 0:
            return ["0"]
        return [f"({'+'.join(out)})"]

    if name == "SelectFeatures":
        idx = node.hp.get("feature_indices", None)
        
        # Si pas d'indices spécifiés, retourner tel quel
        if idx is None:
            return out
        
        # Vérifier si un indice dépasse la taille de out
        if any(i >= len(out) for i in idx):
            # Si un indice est trop grand, retourner out tel quel
            # (comportement cohérent avec SelectFeatures.forward)
            return out
        
        # Sélectionner les features aux indices spécifiés
        try:
            return [out[i] for i in idx]
        except IndexError:
            # Sécurité supplémentaire : si erreur, retourner out
            return out

    if name == "ConstantBrick":
        value = getattr(op, "value", 0.0)
        return [str(value)]

    return out

from sympy import simplify, sympify

def graph_to_formula(adj_matrix, X, nodes):
    n = adj_matrix.shape[0]

    d = X.shape[-1]
    out_dict = {}
    out_dict[0] = [f"{i}" for i in X]

    for i in range(1, n):
        parents = [j for j in range(i) if adj_matrix[j, i] == 1]
        inputs = [out_dict[j] for j in parents]

        out = op_tensors(inputs, nodes[i].combiner)
        out = apply_operation(out, nodes[i])
        out_dict[i] = out

    expr_str = out_dict[n - 1][0]
    expr = sympify(expr_str)
    return simplify(expr)


def expr_to_mini_dag(expr, input_names):
    """
    Compile une expression SymPy en DAG Dragon minimal et valide.
    """
    from dragon.search_space.bricks.basics import Identity
    
    nodes = []
    edges = []
    cache = {}

    # ------------------------------------------------------------------
    # 0. Root node obligatoire
    # ------------------------------------------------------------------
    root = SymbolicNode(
        combiner="add",
        operation=Identity,
        hp={},
        activation=nn.Identity()
    )
    nodes.append(root)  # index 0

    # ------------------------------------------------------------------
    # 1. Fonction helper pour créer une copie d'un symbole
    # ------------------------------------------------------------------
    def create_symbol_copy(symbol_name):
        """Crée une nouvelle copie d'un symbole input"""
        i = input_names.index(symbol_name)
        node = SymbolicNode(
            combiner="add",
            operation=SelectFeatures,
            hp={"feature_indices": [i]},
            activation=nn.Identity()
        )
        idx = len(nodes)
        nodes.append(node)
        edges.append((0, idx))
        return idx

    # ------------------------------------------------------------------
    # 2. Fonction helper pour les puissances
    # ------------------------------------------------------------------
    def multiply_chain(base, exponent):
        """
        Crée une chaîne de multiplications pour base**exponent.
        """
        assert isinstance(exponent, int) and exponent > 0

        # Collecter tous les facteurs
        factors = []
        
        for _ in range(exponent):
            if isinstance(base, Symbol):
                # Créer une nouvelle copie pour chaque facteur
                factor_idx = create_symbol_copy(str(base))
                factors.append(factor_idx)
            else:
                # Pour les expressions complexes, compiler chaque instance
                factor_idx = compile_expr(base)
                factors.append(factor_idx)

        # Multiplier progressivement tous les facteurs
        current_idx = factors[0]
        for factor_idx in factors[1:]:
            mul_node = SymbolicNode("mul", Identity, {}, nn.Identity())
            mul_idx = len(nodes)
            nodes.append(mul_node)
            edges.append((current_idx, mul_idx))
            edges.append((factor_idx, mul_idx))
            current_idx = mul_idx

        return current_idx

    # ------------------------------------------------------------------
    # 3. Compilation récursive
    # ------------------------------------------------------------------
    def compile_expr(e):
        # Note: pas de cache pour les symboles car on veut des copies distinctes
        if not isinstance(e, Symbol) and e in cache:
            return cache[e]

        # ---- Constantes ----
        if isinstance(e, (int, float, Integer, Rational, Float)):
            value = float(e)
            const_node = SymbolicNode(
                combiner="add",
                operation=ConstantBrick,
                hp={"value": abs(value)},
                activation=nn.Identity()
            )
            const_idx = len(nodes)
            nodes.append(const_node)
            edges.append((0, const_idx))
            
            if value < 0:
                neg_node = SymbolicNode(
                    combiner="add",
                    operation=Negate,
                    hp={},
                    activation=nn.Identity()
                )
                neg_idx = len(nodes)
                nodes.append(neg_node)
                edges.append((const_idx, neg_idx))
                cache[e] = neg_idx
                return neg_idx
            else:
                cache[e] = const_idx
                return const_idx

        # ---- Symboles ----
        if isinstance(e, Symbol):
            # Toujours créer une nouvelle copie (pas de cache)
            idx = create_symbol_copy(str(e))
            return idx

        # ---- Addition ----
        if isinstance(e, Add):
            children = [compile_expr(a) for a in e.args]
            node = SymbolicNode("add", Identity, {}, nn.Identity())
            idx = len(nodes)
            nodes.append(node)
            for c in children:
                edges.append((c, idx))
            cache[e] = idx
            return idx

        # ---- Multiplication ----
        if isinstance(e, Mul):
            negate = False
            children_expr = []
            for a in e.args:
                if a == -1:
                    negate = True
                else:
                    children_expr.append(a)

            children = [compile_expr(a) for a in children_expr]
            node = SymbolicNode("mul", Identity, {}, nn.Identity())
            idx = len(nodes)
            nodes.append(node)
            for c in children:
                edges.append((c, idx))

            out = idx
            if negate:
                neg_node = SymbolicNode("add", Negate, {}, nn.Identity())
                neg_idx = len(nodes)
                nodes.append(neg_node)
                edges.append((out, neg_idx))
                out = neg_idx

            cache[e] = out
            return out

        # ---- Puissance ----
        if isinstance(e, Pow):
            base, exponent = e.args

            # Inverse
            if exponent == -1:
                child = compile_expr(base)
                node = SymbolicNode("add", Inverse, {}, nn.Identity())
                idx = len(nodes)
                nodes.append(node)
                edges.append((child, idx))
                cache[e] = idx
                return idx

            # Puissance entière positive
            elif isinstance(exponent, (int, Integer)) and exponent > 0:
                idx = multiply_chain(base, int(exponent))
                cache[e] = idx
                return idx

            # Puissance entière négative
            elif isinstance(exponent, (int, Integer)) and exponent < 0:
                pos_exp = -int(exponent)
                idx = multiply_chain(base, pos_exp)
                inv_node = SymbolicNode("add", Inverse, {}, nn.Identity())
                inv_idx = len(nodes)
                nodes.append(inv_node)
                edges.append((idx, inv_idx))
                cache[e] = inv_idx
                return inv_idx

            else:
                raise NotImplementedError(f"Unsupported exponent: {e}")

        raise NotImplementedError(f"Unsupported expr: {e}")

    # ------------------------------------------------------------------
    # 4. Compile l'expression et crée le nœud de sortie
    # ------------------------------------------------------------------
    output_expr_idx = compile_expr(expr)
    
    # Créer un nœud de sortie final (obligatoire pour le DAG)
    output_node = SymbolicNode(
        combiner="add",
        operation=Identity,
        hp={},
        activation=nn.Identity()
    )
    output_idx = len(nodes)
    nodes.append(output_node)
    edges.append((output_expr_idx, output_idx))

    # ------------------------------------------------------------------
    # 5. Adjacency matrix
    # ------------------------------------------------------------------
    n = len(nodes)
    matrix = np.zeros((n, n), dtype=int)
    for i, j in edges:
        assert i < j, f"Non topological edge {i}->{j}"
        matrix[i, j] = 1

    # ------------------------------------------------------------------
    # 6. Build DAG
    # ------------------------------------------------------------------
    dag = AdjMatrix(nodes, matrix)
    dag.set(input_shape=(len(input_names),))
    return dag