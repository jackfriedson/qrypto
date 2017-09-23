import numpy as np
import pandas as pd

def ema(data, window):
    series = pd.Series(data)
    return series.ewm(span=window).mean()
