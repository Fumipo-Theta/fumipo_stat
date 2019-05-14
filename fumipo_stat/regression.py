import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
base = importr('base')
importr("multcomp")
importr("glm2")
from functools import reduce


def presentable(func):
    def wrapper(self, presenter=lambda d: d):
        return presenter(func(self))
    return wrapper


def get_variable_names_of_R_poly(express: str):
    """
    get variable names of polynominal form expressed as R syntax

    Example
    -------
    get_variable_names("poly(param, degree=2)")
    > ["param_1","param_2"]

    get_variable_names("param")
    > ["param"]
    """
    import re
    variable = re.search(r"poly\(([\w\d]+),.*\)", express)
    _degree = re.search(r"poly\(.*\)(\d+)", express)
    degree = int(_degree.groups()[0]) if _degree else 1

    return f"{variable.groups()[0]}_{degree}" if variable else express


def count_degree(express):
    import re
    result = re.search(r"poly.*degree=(\d+)", express)
    return int(result.groups()[0]) if result else 1


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

    def get_summary_section(self, index, default=None):
        try:
            return self.summary()[:][index]
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

    def fit(self)->IRegressionModelResult:
        pass

    def discript_model(self)->str:
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
        variables = list(map(get_variable_names_of_R_poly,
                             coeff_matrix.names[0])) if coeff_matrix is not None else []
        return variables

    @presentable
    def coeff(self):

        coeff_matrix = self.get_summary_section(3, None)

        if coeff_matrix is None:
            return (PrettyNamedMatrix([[None for __ in range(4)] for _ in range(1)], colnames=range(4), rownames=range(1)))
        else:
            variables = self.get_variables()
            return (PrettyNamedMatrix(coeff_matrix, rownames=variables))

    @presentable
    def r_sqared(self):
        r2 = self.get_summary_section(8, None)
        return r2[0] if r2 is not None else None


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
        coeff_matrix = self.get_summary_section(11, None)
        names = list(map(get_variable_names_of_R_poly,
                         coeff_matrix.names[0])) if coeff_matrix is not None else []
        return names

    @presentable
    def coeff(self):
        coeff_matrix = self.get_summary_section(11, None)

        if coeff_matrix is None:
            return (PrettyNamedMatrix([[None for __ in range(4)] for _ in range(1)], colnames=range(4), rownames=range(1)))
        else:
            variables = self.get_variables()
            return (PrettyNamedMatrix(coeff_matrix, rownames=variables))

    @presentable
    def AIC(self):
        if type(self.result()) is FailedResult:
            return (None)
        aic = self.get_summary_section(4)[0]
        return (aic)

    @presentable
    def residual_deviance(self):
        res_dev = self.get_summary_section(3)
        return res_dev[0] if res_dev is not None else None

    @presentable
    def null_deviance(self):
        null_dev = self.get_summary_section(7)
        return null_dev[0] if null_dev is not None else None

    @presentable
    def psuedo_r_sqared(self):
        return (1-self.residual_deviance()/self.null_deviance())


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
    coeff = lm.coeff()
    """

    def __init__(self):
        self.Result = LMResult
        self.model_name = "lm"

    def discript_model(self):
        return f"{self.model_name} model: {self.create_R_call_string()}"

    def set_formula(self, y, xs):
        expression = reduce(lambda acc, e: acc+"+" +
                            e if acc != "" else e, xs, "")
        formula = f"{y}~{expression}"
        self.formula = formula

    def set_fit_kwargs(self, kwargs):
        self.call_kwargs = reduce(
            lambda acc, e: acc+","+e if acc is not "" else e,
            [f"{key}={value}" for key, value in kwargs.items()],
            "")

    def create_R_call_string(self):
        return f"{self.model_name}({self.formula},data=d,{self.call_kwargs})"

    def fit(self, df, y, *xs, **lm_kwargs):

        if len(xs) is 0:
            raise Exception("At least 1 predicative variable required.")

        self.set_formula(y, xs)
        self.set_fit_kwargs(lm_kwargs)

        robjects.r.assign("d", df)
        call_string = self.create_R_call_string()

        try:
            result = robjects.r(call_string)

            return self.Result(
                result, y, self.discript_model())

        except Exception as e:

            return self.Result(
                None, y, self.discript_model()+f"\n{e}")


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
        expression = reduce(lambda acc, e: acc+"+" +
                            e if acc != "" else e, xs, "")
        formula = f"{y}~{expression}"
        self.formula = formula

    def set_fit_kwargs(self, kwargs):
        self.call_kwargs = reduce(
            lambda acc, e: acc+","+e if acc is not "" else e,
            [f"{key}={value}" for key, value in kwargs.items()],
            "")

    def create_R_call_string(self):
        return f"{self.model_name}({self.formula},data=d,{self.call_kwargs},family={self.distribution_family}(link={self.link_function}))"

    def fit(self, df, y, *xs, **glm_kwargs):

        if len(xs) is 0:
            raise Exception("At least 1 predicative variable required.")

        self.set_formula(y, xs)
        self.set_fit_kwargs(glm_kwargs)

        robjects.r.assign("d", df)
        call_string = self.create_R_call_string()

        try:
            result = robjects.r(call_string)

            return self.Result(
                result, y, self.discript_model())

        except Exception as e:

            return self.Result(
                None, y, self.discript_model()+f"\n{e}")
