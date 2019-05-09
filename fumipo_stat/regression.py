import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
base = importr('base')
importr("multcomp")
importr("glm2")
from functools import reduce


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
    _degree = re.search(r"poly.*degree=(\d+)", express)
    degree = int(_degree.groups()[0]) if _degree else 1

    return [f"{variable.groups()[0]}_{i+1}" for i in range(int(degree))] if variable else [express]


def count_degree(express):
    import re
    result = re.search(r"poly.*degree=(\d+)", express)
    return int(result.groups()[0]) if result else 1


class IRegressionModelResult:
    def __init__(self,
                 model_result,
                 objective_variable_name,
                 express_variable_names,
                 express_variables_number,
                 model_discription
                 ):
        self.model_result = model_result
        self.objective_name = objective_variable_name
        self.variable_names = express_variable_names
        self.variables_number = express_variables_number
        self.model_discription = model_discription
        self.model_type = ""

    def __repr__(self):
        return f"{self.model_type}\n"\
            + f"{self.model_discription}\n"\
            + f"Objective: {self.objective_name}\n"\
            + f"Variables: {self.variable_names}\n"

    def result(self):
        return self.model_result

    def summary(self):
        return base.summary(self.result())

    def y(self):
        return self.objective_name

    def variables(self):
        return self.variable_names


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

    def result(self):
        return self._result


class FailedResult(IRegressionModelResult):
    def __init__(
        self,
        model_result,
        objective_variable_name,
        express_variable_names,
        express_variables_number,
        model_discription,
    ):
        super(FailedResult, self).__init__(model_result,
                                           objective_variable_name,
                                           express_variable_names,
                                           express_variables_number,
                                           model_discription,)
        self.model_type = "Model fit failed"

    def summary(self):
        return self.result()


class LMResult(IRegressionModelResult):
    def __init__(self,
                 model_result,
                 objective_variable_name,
                 express_variable_names,
                 express_variables_number,
                 model_discription,
                 ):
        super(LMResult, self).__init__(model_result,
                                       objective_variable_name,
                                       express_variable_names,
                                       express_variables_number,
                                       model_discription,)
        self.model_type = "R built in lm"


class GLMResult(IRegressionModelResult):
    def __init__(self,
                 model_result,
                 objective_variable_name,
                 express_variable_names,
                 express_variables_number,
                 model_discription,
                 ):
        super(GLMResult, self).__init__(model_result,
                                        objective_variable_name,
                                        express_variable_names,
                                        express_variables_number,
                                        model_discription,)
        self.model_type = "R package glm2"


class LM(IRegressionModel):
    def __init__(self):
        self.Result = LMResult
        self.Failed = FailedResult

    def discript_model(self):
        return f"LM model: {self.formula}"

    def fit(self, df, y, *xs, **lm_kwargs):

        if len(xs) is 0:
            raise Exception("At least 1 predicative variable required.")

        lm_kwargs_string = reduce(
            lambda acc, e: acc+","+e if acc is not "" else e,
            [f"{key}={value}" for key, value in lm_kwargs.items()],
            "")

        variable_names = reduce(
            lambda acc, e: acc+e, map(get_variable_names_of_R_poly, xs), ["Intercept"])
        num_variable = 1 + reduce(lambda acc, e: acc+e, map(count_degree, xs))

        expression = reduce(lambda acc, e: acc+"+" +
                            e if acc != "" else e, xs, "")
        formula = f"{y}~{expression}"
        self.formula = formula

        robjects.r.assign("d", df)
        call_string = f"lm({formula},data=d,{lm_kwargs_string})"
        try:
            result = robjects.r(call_string)
            self._result = self.Result(
                result, y, variable_names, num_variable, self.discript_model())
        except Exception as e:
            self._result = self.Failed(
                None, y, variable_names, num_variable, self.discript_model()+f"\n{e}")

        return self

    def coeff(self, presenter=lambda d: d):
        result = self.result()

        if type(result) is self.Failed:
            coeff = [[None for __ in result.variables()] for _ in range(4)]
        else:
            coeff_vector = result.summary()[:][3]
            coeff = np.array(coeff_vector).reshape(
                (len(result.variables()), 4)).transpose()

        coeff_array = {
            "estimated": coeff[0],
            "std-error": coeff[1],
            "t-value": coeff[2],
            "p-value": coeff[3]
        }

        coeff_dict = {
            "estimated": {k: v for k, v in zip(result.variables(), coeff[0])},
            "std-error": {k: v for k, v in zip(result.variables(), coeff[1])},
            "t-value": {k: v for k, v in zip(result.variables(), coeff[2])},
            "p-value": {k: v for k, v in zip(result.variables(), coeff[3])}
        }
        result_dict = {
            "coeff": coeff_array,
            "coeff_dict": coeff_dict,
            "variables": result.variables()
        }
        return presenter(result_dict)


class GLM(IRegressionModel):
    def __init__(self, family="gaussian", link="identity"):
        self.Result = GLMResult
        self.Failed = FailedResult
        self.distribution_family = family
        self.link_function = link

    def discript_model(self):
        return f"GLM model {self.distribution_family} family "\
            + f"{self.link_function} link: "\
            + f"{self.formula}"

    def fit(self, df, y, *xs, **glm_kwargs):

        if len(xs) is 0:
            raise Exception("At least 1 predicative variable required.")

        glm_kwargs_string = reduce(
            lambda acc, e: acc+","+e if acc is not "" else e,
            [f"{key}={value}" for key, value in glm_kwargs.items()],
            "")

        variable_names = reduce(
            lambda acc, e: acc+e, map(get_variable_names_of_R_poly, xs), ["Intercept"])
        num_variable = 1 + reduce(lambda acc, e: acc+e, map(count_degree, xs))

        expression = reduce(lambda acc, e: acc+"+" +
                            e if acc != "" else e, xs, "")
        formula = f"{y}~{expression}"
        self.formula = formula

        robjects.r.assign("d", df)

        call_string = f"glm2({formula},family={self.distribution_family}(link={self.link_function}),data=d,{glm_kwargs_string})"
        try:
            result = robjects.r(call_string)
            self._result = self.Result(
                result, y, variable_names, num_variable, self.discript_model())
        except Exception as e:
            self._result = self.Failed(
                None, y, variable_names, num_variable, self.discript_model()+f"\n{e}")

        return self

    def coeff(self, presenter=lambda d: d):
        result = self.result()

        if type(result) is self.Failed:
            coeff = [[None for __ in result.variables()] for _ in range(4)]
        else:
            coeff_vector = result.summary()[:][11]
            coeff = np.array(coeff_vector).reshape(
                (len(result.variables()), 4)).transpose()

        coeff_array = {
            "estimated": coeff[0],
            "std-error": coeff[1],
            "t-value": coeff[2],
            "p-value": coeff[3]
        }
        coeff_dict = {
            "estimated": {k: v for k, v in zip(result.variables(), coeff[0])},
            "std-error": {k: v for k, v in zip(result.variables(), coeff[1])},
            "t-value": {k: v for k, v in zip(result.variables(), coeff[2])},
            "p-value": {k: v for k, v in zip(result.variables(), coeff[3])}
        }
        result_dict = {
            "coeff": coeff_array,
            "coeff_dict": coeff_dict,
            "variables": result.variables()
        }
        return presenter(result_dict)

    def AIC(self, presenter=lambda d: d):
        if type(self.result()) is self.Failed:
            return presenter(None)
        aic = self.result().summary()[4][0]
        return presenter(aic)
