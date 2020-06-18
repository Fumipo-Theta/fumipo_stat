import pandas as pd


def normalizer(series: pd.Series):
    std = series.std()
    mean = series.mean()

    def rescale(v): return v * std + mean

    def scale(v): return (v - mean)/std
    return (scale, rescale)
