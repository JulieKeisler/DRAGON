class OrphanError(Exception):
    def __init__(self, node, nodes):
        self.message = "Orphan error: Graph has orphans: node {} in {}".format(node.operation, [n for n in nodes])

    def __str__(self):
        return self.message


class InvalidArgumentError(Exception):
    def __init__(self, level, argument, set_arguments, exception=None):
        self.message = "Invalid arguments caused optimization to fail at level {}, with the argument {}, " \
                       "and the argument sets {}.\nException traceback: {}".format(level, argument, set_arguments,
                                                                                   exception)

    def __str__(self):
        return self.message


class EndOnNan(Exception):
    def __init__(self, epoch):
        self.message = "Stop running at epoch {} because loss is nan".format(epoch)

    def __str__(self):
        return self.message
