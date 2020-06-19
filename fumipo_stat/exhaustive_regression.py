import pandas as pd
import numpy as np
from func_helper import pip
import re
from functools import reduce


def has_scaled(term: str):
    return term[0].islower()


def before_scaled(scaled_term: str):
    return scaled_term[0].upper() + scaled_term[1:]


def quote_poly_term(term: str):
    """
    variable^2 -> (variable, 2)
    """
    match = re.match(r"^(\w+)\^(\d+)$", term)
    if match:
        return (match[1], int(match[2]))
    else:
        return (term, 1)


def quote_interact(term):
    match = re.match(r"(\w+):(\w+)", term)
    if match:
        return (True, match)
    else:
        return (False, term)


def quote_scaled(term: str):
    """
    If variable name start with upper case, the term is determined as original.
    Else, the term is determined to should be scaled.

    Example
    -------
    quote_scaled("Original") -> (False, ("Original", 1))
    quote_scaled("scaled")   -> (True,  ("Scaled", 1))
    quote_scaled("Var^2)     -> (False, ("Var", 2))

    """
    if has_scaled(term):
        return (True, quote_poly_term(before_scaled(term)))
    else:
        return (False, quote_poly_term(term))


def quote_term(term: str):
    (interact, interact_term) = quote_interact(term)
    if interact:
        term_1 = quote_scaled(interact_term[1])
        term_2 = quote_scaled(interact_term[2])
        return [term_1, term_2]
    else:
        return [quote_scaled(term)]


def single_term_to_value(maybe_scaled, coeff, scale_funcs):
    is_scaled, (key, power) = maybe_scaled[0]
    if is_scaled:
        if key not in scale_funcs:
            raise KeyError(f"{key} is not in scale_funcs")
        return lambda **kwargs: coeff * (scale_funcs.get(key)(kwargs.get(key)))**power
    else:
        return lambda **kwargs: coeff * kwargs.get(key) ** power


def double_term_to_value(maybe_scaled, coeff, scale_funcs):
    [(term1_is_scaled, (key1, power1)),
     (term2_is_scaled, (key2, power2))] = maybe_scaled

    if term1_is_scaled & term2_is_scaled:
        return lambda **kwargs: coeff * (
            ((scale_funcs.get(key1)(kwargs.get(key1))) ** power1)
            * ((scale_funcs.get(key2)(kwargs.get(key2))) ** power2)
        )
    elif term1_is_scaled:
        return lambda **kwargs: coeff * (
            ((scale_funcs.get(kwargs.get(key1))(kwargs.get(key1)))**power1)
            * (kwargs.get(key2)**power2)
        )
    elif term2_is_scaled:
        return lambda **kwargs: coeff * (
            (kwargs.get(key1)**power1)
            * (scale_funcs.get(kwargs.get(key2))(kwargs.get(key2))**power2)
        )
    else:
        return lambda **kwargs: coeff * (
            (kwargs.get(key1)**power1)
            * (kwargs.get(key2)**power2)
        )


def intercept_to_value(just_intercept, coeff, _):
    return lambda **kwargs: coeff


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


class Predictor:
    def __init__(self, terms, expressions):
        self.terms = terms
        self.expressions = expressions

    def __repr__(self):
        return "y ~ " + reduce(
            lambda acc, e: acc + self.__construct_term_expression(e) + "\n",
            self.expressions,
            ""
        )

    def predict(self, **kwargs):
        return reduce(lambda acc, f: acc + f(**kwargs), self.terms, 0)

    def __construct_term_expression(self, expression):
        name, coeff = expression

        if coeff > 0:
            return f"+ {coeff} {name} "
        else:
            return f"- {coeff*-1} {name} "


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

    @staticmethod
    def Predictor(estimated, probability_limit, vias_limit, scalers):
        """
        Generate predicting function automatically from exhaustive GLM.


        Usage
        -----
        estimated = ExhaustiveModel(...).generate_confidential_set(...).get_estimate(...)
        predictor = ExhaustiveModel.Predictor(estimated, probability_limit, vias_limit, scale_func_dict)
        predictor.predict(Var1=0.5, Var2=1, ...)


        Parameters
        ----------
        estimated: pd.DataFrame
            columns: [variable, estimate, se, probability, vias]
        probability_limit: float
            probability >= limit variables are selected.
        vias_limit: float
            vias < limit valiables are selected
        scalers: dict
            keys: Original_variable_name (must start from upper case)
            values: scaling function


        Returns
        -------
        When the valid model is y ~ a VarA + b VarB + c varC + de VarD:VarE, 

        Predictor.predict() function equivalent to: 

        def f(VarA, VarB, VarC, VarD, VarE):
            return (
                Intercept
                + a * VarA
                + b * VarB
                + c * scale_C(VarC)
                + de * scale_D(VarD) * scale_E(VarE)
            )

        """
        valid = estimated[(estimated["probability"] >= probability_limit) & (
            estimated["vias"] < vias_limit)]

        terms = []
        raw_terms = []
        for i, row in valid.iterrows():
            term = quote_term(row["variable"])
            coeff = row["estimate"]
            raw_terms.append((row["variable"], coeff))

            if term[0][1][0] == "(Intercept)":
                terms.append(intercept_to_value(term, coeff, scalers))
            elif len(term) == 1:
                terms.append(single_term_to_value(term, coeff, scalers))
            elif len(term) == 2:
                terms.append(double_term_to_value(term, coeff, scalers))

        return Predictor(terms, raw_terms)
