from .exhaustive_regression import quote_term, Predictor, intercept_to_value,\
    single_term_to_value, double_term_to_value
import re
from functools import reduce
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from .util import py2r
from dataclasses import dataclass
base = importr('base')
importr("multcomp")
importr("glm2")


def presentable(func):
    def wrapper(self, presenter=lambda d: d):
        return presenter(func(self))
    return wrapper


def power_to_R_poly(express: str):
    """
    variable^2 -> poly(variable, degree=2)
    """
    match = re.match(r"^(\w+)\^([\d\.]+)$", express)
    if match:
        var = match[1]
        power = match[2]
        return f"poly({var}, degree={power})"
    else:
        return express


def R_poly_to_power(express: str):
    """
    get variable names of polynominal form expressed as R syntax

    Example
    -------
    variable                  -> variable
    variable_2                -> variable_2
    poly(variable, degree=2)1 -> variable
    poly(variable, degree=2)2 -> variable^2
    """

    match = re.match(r"^poly\((\w+)\,.*\)(\d+)$", express)

    if match:
        var = match[1]
        power = int(match[2])
        return f"{var}^{power}" if power != 1 else var
    else:
        return express


def count_degree(express):
    import re
    result = re.search(r"poly.*degree=(\d+)", express)
    return int(result.groups()[0]) if result else 1


@dataclass
class FStatistics:
    f_value: float
    dofs: list[float]

    def __repr__(self):
        """
        returns a string as follow:

        F-statistic: {f} on {dof1} and {dof2} and ... and {dofn} DoF
        """
        dofs_string = reduce(
            lambda acc, e: f"{acc} and {e}" if acc != "" else e, self.dofs, "")
        return f"F-statistic: {self.f_value} on {dofs_string} DoF"

    def __str__(self):
        return self.__repr__()

    def get_f_value(self):
        return self.f_value

    def get_dofs(self):
        return self.dofs


class IRegressionModelResult:
    def __init__(self,
                 model_result,
                 objective_variable_name,
                 model_discription
                 ):
        self.model_result = model_result
        self.objective_name = objective_variable_name
        self.model_discription = model_discription

    def __repr__(self):
        return f"{self.model_discription}\n"\
            + f"Objective: {self.objective_name}\n"\
            + f"Variables: {self.get_variables()}\n"

    def result(self):
        return self.model_result

    def is_significant(self, threshould: float) -> bool:
        pass

    def get_summary_section(self, index, default=None):
        try:
            return self.summary().rx(index + 1)
        except:
            return default

    def summary(self, default=None):
        return base.summary(self.result())

    def y(self):
        return self.objective_name

    def get_variables(self):
        pass

    def colnames(self, index):
        return list(self.summary()[:][index].colnames)

    def rownames(self, index):
        return list(self.summary()[:][index].names)


class IRegressionModel:
    def __init__(self):
        pass

    def __repr__(self):
        return self.discript_model()

    def __new__(cls, *arg, **kwargs):
        return super().__new__(cls)

    def fit(self) -> IRegressionModelResult:
        pass

    def discript_model(self) -> str:
        pass


class FailedResult(IRegressionModelResult):
    def __init__(
        self,
        model_result,
        objective_variable_name,
        model_discription,
    ):
        super(FailedResult, self).__init__(model_result,
                                           objective_variable_name,
                                           model_discription,)
        self.model_type = "Model fit failed"
        self.variable_names = [""]

    def summary(self):
        return self.result()


class LMResult(IRegressionModelResult):
    def __init__(self,
                 model_result,
                 objective_variable_name,
                 model_discription,
                 ):
        super(LMResult, self).__init__(model_result,
                                       objective_variable_name,
                                       model_discription,)
        self.model_type = "R built in lm"

    def get_variables(self):
        coeff_matrix = self.get_summary_section(3, None)
        variables = list(map(R_poly_to_power,
                             coeff_matrix[0].names[0])) if coeff_matrix is not None else []
        return variables

    def is_significant(self, threshold: float = 0.05, strict: bool = True) -> bool:
        if strict:
            return all(map(
                lambda p: p < threshold,
                self.coeff().as_dict()["Pr(>|t|)"].values()
            ))
        else:
            return all(
                map(
                    lambda kp: kp[1] < threshold,
                    filter(
                        lambda kv: kv[0] != '(Intercept)',
                        self.coeff().as_dict()["Pr(>|t|)"].items())))

    def f_statistics(self) -> FStatistics:
        """
        F-statistic: {f} on {df1} and {df2} DF
        """
        f, df1, df2 = self.summary().rx2(10)
        return FStatistics(f, [df1, df2])

    @presentable
    def coeff(self):

        coeff_matrix = self.get_summary_section(3, None)

        if coeff_matrix is None:
            return (PrettyNamedMatrix([[None for __ in range(4)] for _ in range(1)], colnames=range(4), rownames=range(1)))
        else:
            variables = self.get_variables()
            colnames = ["Estimate", "Std. Error", "t value", "Pr(>|t|)"]
            return (PrettyNamedMatrix(coeff_matrix, colnames=colnames, rownames=variables))

    @presentable
    def r_squared(self):
        r2 = self.get_summary_section(8, None)
        return r2[0] if r2 is not None else None

    @presentable
    def r_sqared(self):
        return self.r_squared()

    def Predictor(self, p_limit: float, scalers, disc="") -> Predictor:
        """
        Generate predicting function automatically from GLM.


        Usage
        -----


        Returns
        -------
        When the model is y ~ a VarA + b VarB + c varC + de VarD:VarE, 

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

        estimated = pd.DataFrame(self.coeff().as_dict()).reset_index()

        valid = estimated[estimated["Pr(>|t|)"] < p_limit]

        terms = []
        raw_terms = []
        for i, row in valid.iterrows():
            term = quote_term(row["index"])
            coeff = row["Estimate"]
            raw_terms.append((row["index"], coeff))

            if term[0][1][0] == "(Intercept)":
                terms.append(intercept_to_value(term, coeff, scalers))
            elif len(term) == 1:
                terms.append(single_term_to_value(term, coeff, scalers))
            elif len(term) == 2:
                terms.append(double_term_to_value(term, coeff, scalers))

        _disc = (
            disc + "\n" +
            f"Model parameters selected by\n" +
            f"p-value < {p_limit}\n"
        )

        return Predictor(terms, raw_terms, disc=repr(self) + "\n" + _disc)


class GLM2Result(IRegressionModelResult):
    def __init__(self,
                 model_result,
                 objective_variable_name,
                 model_discription,
                 ):
        super(GLM2Result, self).__init__(model_result,
                                         objective_variable_name,
                                         model_discription,)
        self.model_type = "R package glm2"

    def get_variables(self):
        # This gets invalid attribute when the data set has any NA values.
        # In the case, self.get_summary_section(10, None) will be (omit,)
        # , and self.get_summary_section(12, None) is coeff_matrix
        coeff_matrix = self.result()[0]
        names = list(map(R_poly_to_power,
                         coeff_matrix.names)) if coeff_matrix is not None else []
        return names

    def is_significant(self, threshold: float = 0.05, strict: bool = True) -> bool:
        return all(map(
            lambda p: p < threshold,
            self.coeff().as_dict()["Pr(>|t|)"].values()
        ))

    @presentable
    def coeff(self):
        coeff_matrix = self.get_summary_section(12, None)
        try:
            if coeff_matrix is None:
                return (PrettyNamedMatrix([[None for __ in range(4)] for _ in range(1)], colnames=range(4), rownames=range(1)))
            else:
                variables = self.result()[0].names
                colnames = ["Estimate", "Std. Error", "t value", "Pr(>|t|)"]
                return (PrettyNamedMatrix(coeff_matrix[0], colnames=colnames, rownames=variables))
        except Exception as e:

            coeff_matrix = self.get_summary_section(11, None)
            try:
                if coeff_matrix is None:
                    return (PrettyNamedMatrix([[None for __ in range(4)] for _ in range(1)], colnames=range(4), rownames=range(1)))
                else:
                    variables = self.result()[0].names
                    colnames = ["Estimate", "Std. Error",
                                "t value", "Pr(>|t|)"]
                    return (PrettyNamedMatrix(coeff_matrix[0], colnames=colnames, rownames=variables))
            except Exception as e:
                print(self)
                raise e

    @presentable
    def AIC(self):
        result = self.result()
        if type(result) is FailedResult:
            return (None)
        sample_size = result[16][0]
        parameter_size = sample_size - result[15][0]
        aic = self.get_summary_section(4)[0]
        return aic[0]
        # if (sample_size/parameter_size) > 40:
        #    return aic
        # else:
        #    return aic * sample_size / (sample_size - parameter_size - 1)

    @presentable
    def residual_deviance(self):
        res_dev = self.get_summary_section(3)
        return res_dev[0][0] if res_dev is not None else None

    @presentable
    def null_deviance(self):
        null_dev = self.get_summary_section(7)
        return null_dev[0][0] if null_dev is not None else None

    @presentable
    def psuedo_r_squared(self):
        return (1 - self.residual_deviance() / self.null_deviance())

    def Predictor(self, p_limit: float, scalers, disc="") -> Predictor:
        """
        Generate predicting function automatically from GLM.


        Usage
        -----


        Returns
        -------
        When the model is y ~ a VarA + b VarB + c varC + de VarD:VarE, 

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

        estimated = pd.DataFrame(self.coeff().as_dict()).reset_index()

        valid = estimated[estimated["Pr(>|t|)"] < p_limit]

        terms = []
        raw_terms = []
        for i, row in valid.iterrows():
            term = quote_term(row["index"])
            coeff = row["Estimate"]
            raw_terms.append((row["index"], coeff))

            if term[0][1][0] == "(Intercept)":
                terms.append(intercept_to_value(term, coeff, scalers))
            elif len(term) == 1:
                terms.append(single_term_to_value(term, coeff, scalers))
            elif len(term) == 2:
                terms.append(double_term_to_value(term, coeff, scalers))

        _disc = (
            disc + "\n" +
            f"Model parameters selected by\n" +
            f"p-value < {p_limit}\n"
        )

        return Predictor(terms, raw_terms, disc=repr(self) + "\n" + _disc)


class PrettyNamedMatrix:
    def __init__(self, matrix, colnames=None, rownames=None):
        _colnames = list(matrix.colnames) if colnames is None else colnames
        _rownames = list(matrix.rownames) if rownames is None else rownames
        self._matrix = np.array(matrix).reshape(
            (len(_rownames), len(_colnames))).transpose()
        self._cols = _colnames
        self._rows = _rownames

    def __repr__(self):
        return f"{self.as_list()}\n{self.get_rownames()}"

    def as_list(self):
        return {k: v for k, v in zip(self._cols, self._matrix)}

    def as_dict(self):
        return {k: {k: v for k, v in zip(self._rows, _vec)} for k, _vec in zip(self._cols, self._matrix)}

    def get_rownames(self):
        return self._rows

    def get_colnames(self):
        return self._cols


class LM(IRegressionModel):
    """
    Usage
    -----
    lm = LM()
    lm.fit(data, y_column, x_column1, x_column_2)
    coeff = lm.coeff().as_dict()

    # Example of coeff
    coeff = {
        'Estimate': {
            '(Intercept)': -12.66863903440659,
            'photon': 4.092616876300162
        },
        'Std. Error': {
            '(Intercept)': 6.503325010299246,
            'photon': 0.7534521569237514
        },
        't value': {
            '(Intercept)': -1.9480248971631284,
            'photon': 5.431820506042204
        },
        'Pr(>|t|)': {
                '(Intercept)': 0.05752895762315295,
                'photon': 2.037628873365147e-06
        }
    }
    """

    def __init__(self):
        self.Result = LMResult
        self.model_name = "lm"

    def discript_model(self):
        return f"{self.model_name} model: {self.create_R_call_string()}"

    def set_formula(self, y, xs):
        expression = reduce(lambda acc, e: acc + "+" +
                            e if acc != "" else e, map(power_to_R_poly, xs), "")
        formula = f"{y}~{expression}"
        self.formula = formula

    def set_fit_kwargs(self, kwargs):
        self.call_kwargs = reduce(
            lambda acc, e: acc + "," + e if acc != "" else e,
            [f"{key}={value}" for key, value in kwargs.items()],
            "")

    def create_R_call_string(self):
        return f"{self.model_name}({self.formula},data=d,{self.call_kwargs})"

    def fit(self, df, y, *xs, **lm_kwargs):

        if len(xs) == 0:
            raise Exception("At least 1 predicative variable required.")

        self.set_formula(y, xs)
        self.set_fit_kwargs(lm_kwargs)

        py2r("d", df)
        call_string = self.create_R_call_string()

        try:
            result = robjects.r(call_string)

            return self.Result(
                result, y, self.discript_model())

        except Exception as e:

            return self.Result(
                None, y, self.discript_model() + f"\n{e}")


class GLM2(IRegressionModel):
    def __init__(self, family="gaussian", link="identity"):
        self.Result = GLM2Result
        self.distribution_family = family
        self.link_function = link
        self.model_name = "glm2"

    def discript_model(self):
        return f"GLM model {self.distribution_family} family "\
            + f"{self.link_function} link: "\
            + f"{self.formula}"

    def set_formula(self, y, xs):
        expression = reduce(lambda acc, e: acc + "+" +
                            e if acc != "" else e, map(power_to_R_poly, xs), "")
        formula = f"{y}~{expression}"
        self.formula = formula

    def set_fit_kwargs(self, kwargs):
        self.call_kwargs = reduce(
            lambda acc, e: acc + "," + e if acc != "" else e,
            [f"{key}={value}" for key, value in kwargs.items()],
            "")

    def create_R_call_string(self):
        return f"{self.model_name}({self.formula},data=d,{self.call_kwargs},family={self.distribution_family}(link={self.link_function}))"

    def fit(self, df, y, *xs, **glm_kwargs):

        if len(xs) == 0:
            raise Exception("At least 1 predicative variable required.")

        self.set_formula(y, xs)
        self.set_fit_kwargs(glm_kwargs)

        py2r("d", df)
        call_string = self.create_R_call_string()

        try:
            result = robjects.r(call_string)

            return self.Result(
                result, y, self.discript_model())

        except Exception as e:

            return self.Result(
                None, y, self.discript_model() + f"\n{e}")
