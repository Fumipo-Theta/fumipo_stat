import pandas as pd
import numpy as np
from func_helper import pip


def glm_with(model_generator, all_variables):
    def apply_map(explains):
        """
        explains: List[str]
        """
        res = model_generator(filter(lambda v: v is not None, explains))
        coeff = res.coeff().as_dict()
        var_set = set(res.get_variables())
        explains_as_flag = map(
            lambda v: 1 if v in var_set else 0, all_variables)

        return (*explains_as_flag, res.AIC(), res.psuedo_r_squared(), coeff)
    return apply_map


def set_dAIC(df):
    best_AIC = df["AIC"].min()
    dAIC = df["AIC"] - best_AIC
    return df.assign(dAIC=dAIC)


def set_Akaike_weight(df):
    """
    [Model Selection Using the Akaike Information Criterion (AIC)](http://brianomeara.info/aic.html)
    """
    rel_liklihood = np.exp(-0.5 * df["dAIC"])
    w_Akaike = rel_liklihood / sum(rel_liklihood)
    return df.assign(wi=w_Akaike)


def select_dAIC_lt(v):
    def apply(df):
        return df[df["dAIC"] < v]
    return apply


def se2(series):
    return series.var() / len(series)


def get_coeffs(df, var_name):
    return df[var_name] * df["coeff"].apply(lambda c: c["Estimate"].get(var_name, 0))


def get_selection_probability(df, var_name):
    return (df["wi"] * df[var_name]).sum()


def get_estimate(df, var_name):
    return (df["wi"] * get_coeffs(df, var_name)).sum()


def get_se_of_estimate(df, var_name):
    coeffs = get_coeffs(df, var_name)
    weighted_coeff = get_estimate(df, var_name)
    coeff_se2 = se2(coeffs)
    return (df["wi"] * np.sqrt((coeff_se2 + (weighted_coeff - coeffs)**2))).sum()


def get_vias_of_estimate(df, var_name):
    abs_estimate = np.abs(get_estimate(df, var_name))
    se = get_se_of_estimate(df, var_name)
    return se / abs_estimate


def get_estimate_statistics(df, var_name, presenter):
    estimate = get_estimate(df, var_name)
    se = get_se_of_estimate(df, var_name)
    p = get_selection_probability(df, var_name)
    vias = get_vias_of_estimate(df, var_name)
    return presenter(var_name, estimate, se, p, vias)


class ExhaustiveResult:
    def __init__(self, result_df, variables):
        self.models = result_df
        self.variables = variables

    def get_confidential_set(self, dAIC_threshold):
        return pip(
            set_dAIC,
            select_dAIC_lt(dAIC_threshold),
            set_Akaike_weight
        )(self.models.sort_values("AIC"))

    def get_estimate(self, dAIC_threshold, presenter=print):
        df = self.get_confidential_set(dAIC_threshold)
        model = pd.DataFrame.from_records(
            map(lambda v: get_estimate_statistics(
                df, v, lambda *a: a), self.variables),
            columns=["variable", "estimate", "se", "probability", "vias"]
        )
        return model


class ExhaustiveRegression:
    """
    Class for perform exhaustive combination of variables.
    """

    def __init__(self, stat_model, SubsetCls):
        """
        stat_model: IRegressionModel
        SubsetCls:  Subset
        """
        self.regression = stat_model
        self.SubsetBuilder = SubsetCls

    def get_confidential_set(self, df, objective, variables):
        var_subset = self.SubsetBuilder(variables, drop=False)
        full_model = self.regression.fit(df, objective, *variables)
        all_vars = full_model.get_variables()

        models = pd.DataFrame.from_records(
            map(glm_with(lambda explains: self.regression.fit(
                df, objective, *explains), all_vars), var_subset),
            columns=[*all_vars, "AIC", "R2", "coeff"]
        )
        return ExhaustiveResult(models, all_vars)
