from operator import attrgetter
import random
import numpy as np

from dragon.search_space.addons import Operator, Mutator, Crossover
from dragon.search_space.cells import AdjMatrix, fill_adj_matrix


class DAGTwoPoint(Crossover):

    def __init__(self, search_space=None, size=10):
        self.size = size
        super(DAGTwoPoint, self).__init__(search_space)

    def _build(self, toolbox):
        toolbox.register("mate", self)

    def __call__(self, ind1, ind2):
        size = min(len(ind1), len(ind2))
        cxpoint1 = random.randint(1, size)
        cxpoint2 = random.randint(1, size - 1)
        if cxpoint2 >= cxpoint1:
            cxpoint2 += 1
        else:  # Swap the two cx points
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1

        ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]

        for i in range(cxpoint1, cxpoint2):
            if isinstance(ind1[i], AdjMatrix):
                ind1[i], ind2[i] = self.adj_matrix_crossover(ind1[i], ind2[i])

        return ind1, ind2

    def adj_matrix_crossover(self, p1, p2):
        crossed = False
        while not crossed:
            op1 = p1.operations
            op2 = p2.operations
            m1 = p1.matrix
            m2 = p2.matrix

            s1 = list(set(np.random.choice(range(1, len(op1)), size=len(op1) - 1)))
            s2 = list(set(np.random.choice(range(1, len(op2)), size=len(op2) - 1)))
            s1.sort()
            s2.sort()

            # remove subsets
            it = 0
            for i1 in s1:
                m1 = np.delete(m1, i1 - it, axis=0)
                m1 = np.delete(m1, i1 - it, axis=1)
                it+=1
            it = 0
            for i2 in s2:
                m2 = np.delete(m2, i2 - it, axis=0)
                m2 = np.delete(m2, i2 - it, axis=1)
                it+=1

            # Select index new nodes
            old_s1 = np.array(list(set(range(len(op1))) - set(s1)))
            old_s2 = np.array(list(set(range(len(op2))) - set(s2)))
            new_s1 = [np.argmin(np.abs(old_s2 - s1[0]))]
            if new_s1[0] == old_s2[new_s1[0]]:
                new_s1[0] += 1
            for i1 in range(1, len(s1)):
                new_s1.append(min(s1[i1] - s1[i1-1] + new_s1[i1-1], len(old_s2) + len(new_s1)))
            new_s2 = [np.argmin(np.abs(old_s1 - s2[0]))]
            if new_s2[0] == old_s1[new_s2[0]]:
                new_s2[0] += 1
            for i2 in range(1, len(s2)):
                new_s2.append(min(s2[i2] - s2[i2 - 1] + new_s2[i2-1], len(old_s1) + len(new_s2)))
            m1 = np.insert(m1, np.clip(new_s2, 0, m1.shape[0]), 0, axis=0)
            m1 = np.insert(m1, np.clip(new_s2, 0, m1.shape[1]), 0, axis=1)
            m2 = np.insert(m2, np.clip(new_s1, 0, m2.shape[0]), 0, axis=0)
            m2 = np.insert(m2, np.clip(new_s1, 0, m2.shape[1]), 0, axis=1)
            for i in range(len(s1)):
                diff = new_s1[i] - s1[i]
                if diff >= 0:
                    length = min(m2.shape[0] - diff, p1.matrix.shape[0])
                    m2[diff:diff+length, new_s1[i]] = p1.matrix[:length, s1[i]]
                    m2[new_s1[i], diff:diff+length] = p1.matrix[s1[i], :length]
                if diff < 0:
                    length = min(m2.shape[0], p1.matrix.shape[0]+diff)
                    m2[:length, new_s1[i]] = p1.matrix[-diff:-diff+length, s1[i]]
                    m2[new_s1[i], :length] = p1.matrix[s1[i], -diff:-diff+length]
            for i in range(len(s2)):
                diff = new_s2[i] - s2[i]
                if diff >= 0:
                    length = min(m1.shape[0] - diff, p2.matrix.shape[0])
                    m1[diff:diff+length, new_s2[i]] = p2.matrix[:length, s2[i]]
                    m1[new_s2[i], diff:diff+length] = p2.matrix[s2[i], :length]
                if diff < 0:
                    length = min(m1.shape[0], p2.matrix.shape[0]+diff)
                    m1[:length, new_s2[i]] = p2.matrix[-diff:-diff + length, s2[i]]
                    m1[new_s2[i], :length] = p2.matrix[s2[i], -diff:-diff + length]
            m1 = np.triu(m1, k=1)
            m1 = fill_adj_matrix(m1)
            m2 = np.triu(m2, k=1)
            m2 = fill_adj_matrix(m2)
            op1 = [op1[i] for i in range(len(op1)) if i not in s1]
            op2 = [op2[i] for i in range(len(op2)) if i not in s2]
            for i in range(len(new_s1)):
                op2 = op2[:new_s1[i]] + [p1.operations[s1[i]]] + op2[new_s1[i]:]
            for i in range(len(new_s2)):
                op1 = op1[:new_s2[i]] + [p2.operations[s2[i]]] + op1[new_s2[i]:]
            if max(len(op1), len(op2)) <= self.size:
                crossed = True
        for j in range(1, len(op1)):
            if hasattr(op1[j], "modification"):
                input_shapes = [op1[i].output_shape for i in range(j) if m1[i, j] == 1]
                op1[j].modification(input_shapes=input_shapes)
        for j in range(1, len(op2)):
            if hasattr(op2[j], "modification"):
                input_shapes = [op2[i].output_shape for i in range(j) if m2[i, j] == 1]
                op2[j].modification(input_shapes=input_shapes)
        return AdjMatrix(op1, m1), AdjMatrix(op2, m2)

    @Mutator.target.setter
    def target(self, search_space):
        self._target = search_space


class SelBestWoDuplicate(Operator):
    def __init__(self, search_space=None):
        super(SelBestWoDuplicate, self).__init__(search_space)

    def __call__(self, individuals, k, fit_attr="fitness"):
        """Select the *k* best individuals among the input *individuals*. The
            list returned contains references to the input *individuals*.

            :param individuals: A list of individuals to select from.
            :param k: The number of individuals to select.
            :param fit_attr: The attribute of individuals to use as selection criterion
            :returns: A list containing the k best individuals.
            """
        rpr = []
        wo_duplicates = []
        for ind in individuals:
            if ind.__repr__() not in rpr:
                rpr.append(ind.__repr__())
                wo_duplicates.append(ind)
        out = sorted(wo_duplicates, key=attrgetter(fit_attr), reverse=True)[:k]
        return out


class Random(Operator):
    def __init__(self, value, searchspace=None):
        super(Random, self).__init__(searchspace)
        self.value = value

    def __call__(self, *args, **kwargs):
        return self.value