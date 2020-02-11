from scipy import stats as scipy_stats
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, numpy2ri
pandas2ri.activate()
numpy2ri.activate()
import dataframe_helper as dataframe
from .r_to_py import as_dict


def ANOVA(*array_likes):
    return scipy_stats.f_oneway(*array_likes)


def shapiro_test(array_like):

    # 正規性テスト
    shapiro1 = scipy_stats.shapiro(array_like)
    return shapiro1


def bartlett_test(df, matrix_selector, group_selector, block_selectors):
    """
    等分散性の検定
    """
    block_df = dataframe.create_complete_block_designed_df(
        matrix_selector, group_selector, block_selectors
    )(df).values
    robjects.r.assign("d", block_df)
    result = robjects.r(f"bartlett.test(list(d[,1],d[,2],d[,3]))")
    return as_dict(result)


def wilcoxon_signed_rank_test(df, group, y, paired=True, method="bonferroni"):
    """
    (対応のある)データのノンパラメトリックな群間比較


    Parameters
    ----------
    df: pandas.DataFrame
    group: str
    y: str
    """
    robjects.r.assign("d", pandas2ri.py2ri(df))
    result = robjects.r(
        f"pairwise.wilcox.test(d${y},d${group},paired={'T' if paired else 'F'},p.adjust.method='{method}')")
    result_dict = as_dict(result)
    return result_dict


def wilcoxon_rank_sum_test(x: np.ndarray, y: np.ndarray):
    robjects.r.assign("x", x)
    robjects.r.assign("y", y)
    result = robjects.r(
        f"wilcox.test(x, y)"
    )
    result_dict = as_dict(result)
    return result_dict
