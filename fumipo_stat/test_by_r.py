import dataclasses
from .r_to_py import as_dict
import dataframe_helper as dataframe
from scipy import stats as scipy_stats
import numpy as np
import scikit_posthocs as sp
import rpy2.robjects as robjects
from rpy2.robjects.packages import data, importr
from rpy2.robjects import pandas2ri, numpy2ri
pandas2ri.activate()
numpy2ri.activate()
importr("multcomp")


def ANOVA(*array_likes):
    return scipy_stats.f_oneway(*array_likes)


@dataclasses.dataclass()
class PairwiseTResult:
    pvalue: float


@dataclasses.dataclass()
class ShapiroResult:
    statistics: float
    pvalue: float


@dataclasses.dataclass()
class InvalidResult:
    statistics: float
    pvalue: float


def shapiro_test(array_like):
    """
    正規性テスト

    Returns
    -------
    (w: float, p: float)
    w: test statistics
    p: The p-value for the hypothesis test.
    """
    if len(array_like) < 3:
        return InvalidResult(None, None)

    return ShapiroResult(*scipy_stats.shapiro(array_like))


def t_test(array1, array2, equal_var=True):
    """
    If arrays are not equivarient, this do Welch's t test.
    """
    return scipy_stats.ttest_ind(array1, array2, equal_var=equal_var)


def pairwise_t_test(array1, array2, equal_var=True, method="bonf"):
    value = np.array([x for x in array1] + [x for x in array2])
    group = np.array([0 for _ in array1] + [1 for _ in array2])
    robjects.r.assign("d", value)
    robjects.r.assign("g", group)
    result = as_dict(robjects.r(
        f"pairwise.t.test(d, g, p.adj='{method}', pool.sd={'T' if equal_var else 'F'})"))
    return PairwiseTResult(pvalue=result["p.value"])


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


def bartlett(*array_likes):
    """
    Test of equality of varience.
    """
    return scipy_stats.bartlett(*array_likes)


def levene(*array_like, center="median", proportiontocut=0.05):
    """
    Robast test fof equality of varience.
    """
    return scipy_stats.levene(*array_like, center=center, proportiontocut=proportiontocut)


def wilcoxon_signed_rank_test(df, group, y, paired=True, method="bonferroni"):
    """
    (対応のある)データのノンパラメトリックな群間比較


    Parameters
    ----------
    df: pandas.DataFrame
    group: str
    y: str
    paired: bool
    """
    robjects.r.assign("d", pandas2ri.py2ri(df))
    result = robjects.r(
        f"pairwise.wilcox.test(d${y},d${group},paired={'T' if paired else 'F'},p.adjust.method='{method}')")
    result_dict = as_dict(result)
    return result_dict


def wilcoxon_rank_sum_test(x: np.ndarray, y: np.ndarray):
    """
    ノンパラメトリックな代表値比較の検定

    x,y 2つのベクトルの中央値が等しいという帰無仮説を検定する.
    """
    robjects.r.assign("x", x)
    robjects.r.assign("y", y)
    result = robjects.r(
        f"wilcox.test(x, y)"
    )
    result_dict = as_dict(result)
    return result_dict


def tukey_hsd(df, x, y):
    _df = df[[x, y]]
    _df[x] = _df[x].astype("category")
    robjects.r.assign("d", _df)
    robjects.r(f"aov_res <- aov({y}~{x}, d)")
    robjects.r(f"tuk <- glht(aov_res, linfct=mcp({x}='Tukey'))")
    return robjects.r("cld(tuk, decreasing=T)")


def steel_dwass(df, x, y, **kwargs):
    """
    TukeyHSD のノンパラメトリック版
    """
    return sp.posthoc_dscf(df, val_col=y, group_col=x, **kwargs)


def cld(significance, labels):
    """
    https://qiita.com/TomosurebaOrange/items/c28fc9cae922f3c21e08
    """
    import networkx as nx

    # 有意差のないものを 1 とする
    mat = (significance > 0.05)

    V = nx.from_numpy_matrix(mat)
    G = nx.Graph(V)

    # クリーク辺被覆問題を解き，同じ文字を与えるべきクリーク集合を得る
    cliques = nx.find_cliques(G)

    # 各クリーク集合の要素について，同一の文字列ラベルを与える
    cld_dict = {i: [] for i in labels}
    label_table = {i: l for i, l in enumerate(labels)}
    for i, clique in enumerate(cliques):
        letter = chr(i + 97)
        for j in clique:
            cld_dict[label_table[j]].append(letter)

    for i in cld_dict:
        cld_dict[i] = ''.join(cld_dict[i])

    return cld_dict
