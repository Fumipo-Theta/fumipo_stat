from scipy import stats as scipy_stats
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, numpy2ri
pandas2ri.activate()
import func_helper.func_helper.dataframe as dataframe
from .r_to_py import as_dict


def ANOVA(*array_likes):
    return scipy_stats.f_oneway(*array_likes)


def shapiro_test(array_like):

    # 正規性テスト
    shapiro1 = scipy_stats.shapiro(array_like)
    return shapiro1


def bartlett_test(df, matrix_selector, group_selector, block_selectors):
    block_df = dataframe.create_complete_block_designed_df(
        matrix_selector, group_selector, block_selectors
    )(df).values
    robjects.r.assign("d", block_df)
    result = robjects.r(f"bartlett.test(list(d[,1],d[,2],d[,3]))")
    return as_dict(result)


def wilcoxon_signed_rank_test(df, group, y, method="bonferroni"):

    robjects.r.assign("d", pandas2ri.py2ri(df))
    result = robjects.r(
        f"pairwise.wilcox.test(d${y},d${group},paired=T,p.adjust.method='{method}')")
    result_dict = as_dict(result)
    return result_dict
