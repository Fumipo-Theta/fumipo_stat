# %%
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import sys
sys.path.append("../")
from fumipo_stat.util import py2r
base = importr('base')

ro.r('print("Hello R")')


ro.r("x <- c(0, 1, 2, 3, 4)")
ro.r("y <- c(0, 1, 0, 2, 3)")
result = ro.r("lm(y~x)")
print(result)
print(base.summary(result))
assert list(base.summary(result).rx(4)[0].names.rx2(1)) == ["(Intercept)", "x"]
assert list(base.summary(result).rx(4)[0].names.rx2(2)) == [
    "Estimate", "Std. Error", "t value", "Pr(>|t|)"]
assert list(base.summary(result).rx(4)[0].rownames) == ["(Intercept)", "x"]

# %%
import numpy as np
import pandas as pd
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.conversion import localconverter

ro.r("x <- c(0, 1, 2, 3, 4)")
ro.r("y <- c(0, 1, 0, 2, 3)")
result = ro.r("lm(y~x)")

assert list(base.summary(result).rx(4)[0].names.rx2(1)) == ["(Intercept)", "x"]
assert list(base.summary(result).rx(4)[0].names.rx2(2)) == [
    "Estimate", "Std. Error", "t value", "Pr(>|t|)"]
assert list(base.summary(result).rx(4)[0].rownames) == ["(Intercept)", "x"]

# %%
d = pd.DataFrame({
    "x": [0, 1, 2, 3, 4],
    "y": [0, 1, 0, 2, 3]
})

with localconverter(ro.default_converter + pandas2ri.converter):
    r_from_pd_df = ro.conversion.py2rpy(d)
ro.r.assign("d", r_from_pd_df)
result = ro.r("lm(data=d, y~x)")

assert list(base.summary(result).rx(4)[0].names.rx2(1)) == ["(Intercept)", "x"]
assert list(base.summary(result).rx(4)[0].names.rx2(2)) == [
    "Estimate", "Std. Error", "t value", "Pr(>|t|)"]
assert list(base.summary(result).rx(4)[0].rownames) == ["(Intercept)", "x"]

# %%
with localconverter(ro.default_converter + numpy2ri.converter):
    ro.r.assign("x", np.array([0, 1, 2, 3, 4]))
    ro.r.assign("y", np.array([0, 1, 0, 2, 3]))

result = ro.r("lm(data=d, y~x)")

assert list(base.summary(result).rx(4)[0].names.rx2(1)) == ["(Intercept)", "x"]
assert list(base.summary(result).rx(4)[0].names.rx2(2)) == [
    "Estimate", "Std. Error", "t value", "Pr(>|t|)"]
assert list(base.summary(result).rx(4)[0].rownames) == ["(Intercept)", "x"]
# %%
py2r("xx", np.array([0, 1, 2, 3, 4]))
py2r("yy", np.array([0, 1, 0, 2, 3]))

result = ro.r("lm(yy~poly(xx, degree=2, raw=T))")

assert list(base.summary(result).rx(4)[0].names.rx2(1)) == [
    '(Intercept)', 'poly(xx, degree = 2, raw = T)1', 'poly(xx, degree = 2, raw = T)2']
assert list(base.summary(result).rx(4)[0].names.rx2(2)) == [
    "Estimate", "Std. Error", "t value", "Pr(>|t|)"]
assert list(base.summary(result).rx(4)[0].rownames) == [
    '(Intercept)', 'poly(xx, degree = 2, raw = T)1', 'poly(xx, degree = 2, raw = T)2']

# %%
