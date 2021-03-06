import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri

from rpy2.robjects.conversion import localconverter


def py2r(r_var_name: str, pydata):
    with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
        ro.r.assign(r_var_name, pydata)
