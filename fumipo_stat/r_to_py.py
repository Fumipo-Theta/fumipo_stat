import rpy2
import rpy2.robjects as robjects
import numpy as np
from functools import reduce


def as_dict(vector):
    """Convert an RPy2 ListVector to a Python dict"""
    result = {}
    for i, name in enumerate(vector.names):
        if isinstance(vector[i], robjects.ListVector):
            result[name] = as_dict(vector[i])
        elif isinstance(vector[i], rpy2.rinterface.NULLType):
            result[name] = "Null"
        elif len(vector[i]) == 1:
            result[name] = vector[i][0]
        else:
            result[name] = vector[i]
    return result


def r_matrix_to_table(r_mat, td_hook=lambda v: v):
    """
    Convert R style matrix to markdown table.
    """
    row = r_mat.rownames
    col = r_mat.colnames
    values = np.array(r_mat)

    header = reduce(lambda acc, e: f"{acc}{e}|", ["/", *col], "|")
    align = reduce(lambda acc, e: f"{acc}---|", ["", *col], "|")
    body = ""
    for hr, tds in zip(row, values):
        body += f"|{hr}" + reduce(lambda acc,
                                  e: f"{acc}{td_hook(e)(e)}|", tds, "|") + "\n"

    return f"{header}\n{align}\n{body}"
