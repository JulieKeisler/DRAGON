import ast
import logging
import sys
import os
import datetime
import torch
import random
import numpy as np


def set_logs():
    if not os.path.exists("logs"):
        try:
            os.makedirs("logs")
        except FileExistsError:
            pass
    filename = os.path.join("logs", "evodag_" + str(datetime.date.today()) + ".log")
    logging.basicConfig(filename=filename, level='INFO')
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logging.getLogger('numexpr').setLevel(logging.WARNING)
    logger.info("The logger has been created.")
    return logger


def set_seed(s):
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    np.random.seed(s)
    random.seed(s)
    os.environ['PYTHONHASHSEED'] = str(s)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"


def read_nn(string):
    from dragon.search_space.dags import AdjMatrix
    nodes, matrix = string.split('|')
    nodes = ast.literal_eval(nodes.split(':')[1].strip())
    matrix = ast.literal_eval(matrix.split(':')[1].strip())
    return AdjMatrix(nodes, np.array(matrix))


logger = set_logs()
