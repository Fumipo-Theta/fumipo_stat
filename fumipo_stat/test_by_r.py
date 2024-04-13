from __future__ import annotations
import dataclasses
from .r_to_py import as_dict
from .util import py2r
from func_helper import pip
import dataframe_helper as dataframe
from scipy import stats as scipy_stats
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from typing import Literal, Sequence
import rpy2.robjects as ro
from rpy2.robjects.packages import data, importr

importr("multcomp")
base = importr("base")


def ANOVA(*array_likes):
    return scipy_stats.f_oneway(*array_likes)


@dataclasses.dataclass()
class TTestIndResult:
    statistics: float
    pvalue: float
    dof: float


@dataclasses.dataclass()
class PairwiseTResult:
    statistics: float
    pvalue: float
    dof: float


@dataclasses.dataclass()
class ShapiroResult:
    statistics: float
    pvalue: float


@dataclasses.dataclass()
class InvalidResult:
    statistics: float
    pvalue: float


from sklearn.linear_model import LinearRegression
from statsmodels.stats.diagnostic import normal_ad


@dataclasses.dataclass()
class AndersonDarlingTestResult:
    p_value: float
    statistics: float

    def is_significant(self, p_threshold):
        return self.p_value >= p_threshold


class TestNormalityOfError:
    """
    X should be a multi-dimensional vector.
    If you want to pass pd.Series, call `.values.reshape(-1, 1)` and pass the returned array.
    """

    def __init__(self, X: pd.Series, y: pd.Series):
        self.model = LinearRegression()
        self.model.fit(X, y)
        self.X = X
        self.y = y

    def __calculate_residuals(self):
        """
        Creates predictions on the features with the model and calculates residuals
        """
        predictions = self.model.predict(self.X)
        df_results = pd.DataFrame({'Actual': self.y, 'Predicted': predictions})
        df_results['Residuals'] = abs(
            df_results['Actual']) - abs(df_results['Predicted'])

        return df_results

    def test(self) -> AndersonDarlingTestResult:
        df_results = self.__calculate_residuals()
        statistics, p_value = normal_ad(df_results['Residuals'])

        return AndersonDarlingTestResult(p_value, statistics)


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
    py2r("a1", array1)
    py2r("a2", array2)
    result = as_dict(ro.r(
        f"t.test(a1, a2, var.equal={'T' if equal_var else 'F'})"
    ))
    return TTestIndResult(result["statistic"], result["p.value"], result["parameter"])


def pairwise_t_test(array1, array2, equal_var=True, method="bonf"):
    """
    two-sided.
    """
    value = np.array([x for x in array1] + [x for x in array2])
    group = np.array([1 for _ in array1] + [2 for _ in array2])
    py2r("d", value)
    py2r("g", group)
    result = as_dict(ro.r(
        f"pairwise.t.test(d, g, p.adj='{method}', pool.sd={'T' if equal_var else 'F'})"))

    py2r("a1", array1)
    py2r("a2", array2)
    meta = as_dict(ro.r(
        f"t.test(a1, a2, var.equal={'T' if equal_var else 'F'})"
    ))

    dof = len(value) / 2 - 1

    return PairwiseTResult(meta["statistic"], result["p.value"], dof)


def bartlett_test(df, matrix_selector, group_selector, block_selectors):
    """
    等分散性の検定
    """
    block_df = dataframe.create_complete_block_designed_df(
        matrix_selector, group_selector, block_selectors
    )(df).values
    py2r("d", block_df)
    result = ro.r(f"bartlett.test(list(d[,1],d[,2],d[,3]))")
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


@dataclasses.dataclass()
class WilcoxonSignedRankTestResult:
    statistics: float
    pvalue: float


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
    py2r("d", df)
    result = ro.r(
        f"pairwise.wilcox.test(d${y},d${group},paired={'T' if paired else 'F'},p.adjust.method='{method}')")
    result_dict = as_dict(result)
    return result_dict


@dataclasses.dataclass()
class WilcoxonRankSumTestResult:
    statistics: float
    pvalue: float


def wilcoxon_rank_sum_test(x: np.ndarray, y: np.ndarray, raw=False):
    """
    ノンパラメトリックな代表値比較の検定

    x,y 2つのベクトルの中央値が等しいという帰無仮説を検定する.
    """
    py2r("x", x)
    py2r("y", y)
    result = ro.r(
        f"wilcox.test(x, y)"
    )
    result_dict = as_dict(result)
    if raw:
        return result_dict
    else:
        return WilcoxonRankSumTestResult(result_dict["statistic"], result_dict["p.value"])


def tukey_hsd(df, x, y) -> tuple:
    _df = df[[x, y]]
    _df[x] = _df[x].astype("category")
    py2r("d", _df)
    ro.r(f"aov_res <- aov({y}~{x}, d)")
    ro.r(f"tuk <- glht(aov_res, linfct=mcp({x}='Tukey'))")
    tuk = base.summary(ro.r("tuk"))
    cld = ro.r("cld(tuk, decreasing=T, reversed=T)")
    return (tuk, cld)


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

    # TODO: nx.from_numpy_matrix removed from networkx >= 3
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


def multicomp(df: pd.DataFrame, test_col, group_col, groups: Sequence = [], presenter=print, p_threshold=0.05):
    """
    Premise
        Data is R compatible.
        groups is factors which is in df[group_col]
    """

    _groups = df[group_col].astype(
        "category").cat.categories if len(groups) == 0 else groups
    data = df[df[group_col].isin(_groups)]

    def check_normal(df, test_col, group_col, groups, threshold):
        def is_normal(series):
            return shapiro_test(series).pvalue > threshold
        return all(map(is_normal, [df[df[group_col] == group][test_col] for group in groups]))

    def check_equivariant(df, test_col, group_col, groups, threshold):
        bartlett_res = bartlett(*[
            df[df[group_col] == group][test_col] for group in groups
        ])
        return bartlett_res.pvalue <= threshold

    def parametric_compat(df, test_col, group_col, groups, threshold):
        are_all_normal = check_normal(
            df, test_col, group_col, groups, threshold)
        are_all_equi_var = check_equivariant(
            df, test_col, group_col, groups, threshold)
        if are_all_normal and are_all_equi_var:
            return (True, None)
        elif are_all_normal:
            return (False, "not equivariant")
        elif are_all_equi_var:
            # 今回はデータサイズ同じなのでパラメトリックでも良い
            return (True, "not normal")
        else:
            return (False, "complex distribution")

    (is_parametric_compat, reason) = parametric_compat(
        data, test_col, group_col, _groups, p_threshold)
    if is_parametric_compat:
        res = ("TukeyHSD", reason, tukey_hsd(data, group_col, test_col)[1])
    else:
        _df = data[[test_col, group_col]]
        _df[test_col] = _df[test_col].apply(lambda v: np.power(v, 0.25))
        (is_parametric_compat, reason) = parametric_compat(
            _df, test_col, group_col, _groups, p_threshold)
        if is_parametric_compat:
            res = ("TukeyHSD_biquadroot", reason,
                   tukey_hsd(_df, group_col, test_col)[1])
        else:
            __df = _df[[test_col, group_col]]
            __df[test_col] = __df[test_col].apply(lambda v: np.log(v))
            (is_parametric_compat, reason) = parametric_compat(
                __df, test_col, group_col, _groups, p_threshold)
            if is_parametric_compat:
                res = ("TukeyHSD_log", reason, tukey_hsd(
                    __df, group_col, test_col)[1])
            else:
                significance = steel_dwass(data, group_col, test_col)
                res = ("SteelDwass", reason, cld(
                    significance.values, list(significance.columns)))
    return res


Larger = Literal["left", "right"]


def _len(i):
    if hasattr(i, "count"):
        return i.count()
    return len(i[np.isfinite(i)])


def compare_test_suite(x, y, paired, presenter=print) -> tuple[bool, Larger | None, dict]:
    empty_result = {
        "test": None,
        "statistics": None,
        "pvalue": None,
        "dof": None,
        "significant": False,
        "note": None,
    }
    if _len(x) < 3 or _len(y) < 3:
        note = f"Data size must be larger than 3. Actural x: {_len(x)}, y: {_len(y)}"
        presenter(note)
        return (False, None, empty_result | {"note": note})

    def which_is_larger(left, right, method) -> Larger | None:
        l = method(left)
        r = method(right)
        if l == r:
            return None
        elif l > r:
            return "left"
        else:
            return "right"

    x_shapiro = shapiro_test(x)
    y_shapiro = shapiro_test(y)
    are_normal_dist = x_shapiro.pvalue > 0.05 and y_shapiro.pvalue > 0.05

    equal_var_test = bartlett(
        x, y) if are_normal_dist else levene(x.values, y.values)
    are_equal_var = equal_var_test.pvalue <= 0.05

    if paired:
        presenter("Paired test")

        if are_normal_dist and are_equal_var:
            note = "are normal distribution and equal variance"
            exam_result = pairwise_t_test(x, y, True)
            larger = which_is_larger(x, y, np.mean)
        elif are_normal_dist:
            note = "are normal distribution but not equal variance"
            exam_result = pairwise_t_test(x, y, False)
            larger = which_is_larger(x, y, np.mean)
        else:
            note = "are not normal distribution and not equal variance"
            exam_result = WilcoxonSignedRankTestResult(*scipy_stats.wilcoxon(
                x, y))
            larger = which_is_larger(x, y, np.nanmedian)
    else:
        presenter("Individual test")

        if are_normal_dist and are_equal_var:
            note = "are normal distribution and equal variance"
            exam_result = t_test(x, y, True)
            larger = which_is_larger(x, y, np.mean)
        elif are_normal_dist:
            note = "are normal distribution but not equal variance"
            exam_result = t_test(x, y, False)
            larger = which_is_larger(x, y, np.mean)
        else:
            note = "are not normal distribution and not equal variance"
            exam_result = wilcoxon_rank_sum_test(x, y)
            larger = which_is_larger(x, y, np.nanmedian)

    if exam_result.pvalue <= 0.05:
        label = '**Maybe significant**'
        is_significant = True
    else:
        label = 'Not significant'
        is_significant = False

    result = {
        "test": exam_result.__class__.__name__,
        "statistics": exam_result.statistics,
        "pvalue": exam_result.pvalue,
        "dof": exam_result.dof if hasattr(exam_result, "dof") else None,
        "significant": is_significant,
        "note": f"{label} ({note})",
    }

    presenter(note)
    presenter(f"{label} [{exam_result}]")

    return (is_significant, larger, result)


def basic_stat_map(s: pd.Series) -> dict:
    return {
        "mean": s.mean(),
        "median": s.median(),
        "std": s.std(),
        "count": s.count(),
    }


def basic_stat_df(xs: list[pd.Series], names: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "mean": map(lambda s: s.mean(), xs),
        "median": map(lambda s: s.median(), xs),
        "std": map(lambda s: s.std(), xs),
        "count": map(lambda s: s.count(), xs)
    }, index=names)


Nanpolicy = Literal["propagate", "raise", "omit"]


def kruskal(df, group, target, nan_policy: Nanpolicy | None = None):
    factors = pip(
        np.unique,
        np.sort
    )(df[group])

    groups = list(map(
        lambda factor: df[df[group] == factor][target],
        factors
    ))

    return scipy_stats.kruskal(*groups, nan_policy=nan_policy)
