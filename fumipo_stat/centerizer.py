import pandas as pd


def centerizer(series: pd.Series):
    mean = series.mean()

    def rescale(v): return v + mean

    def scale(v): return v - mean
    return (scale, rescale)
