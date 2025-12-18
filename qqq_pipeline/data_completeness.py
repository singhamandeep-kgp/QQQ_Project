import numpy as np
import pandas as pd
import gc

def check_data_completeness(QQQ: pd.DataFrame) -> pd.DataFrame:
    liq_bad = (QQQ["callVolume"].isna() | QQQ["putVolume"].isna() |
            QQQ["callOpenInterest"].isna() | QQQ["putOpenInterest"].isna())

    print("Percent observations with NA open interest or volume %:", 100 * liq_bad.mean())

    greeks_bad = (~np.isfinite(QQQ["callDelta"])|
                  ~np.isfinite(QQQ["putDelta"])|
                 ~np.isfinite(QQQ["gamma"]) |
                 ~np.isfinite(QQQ["vega"])  |
                 ~np.isfinite(QQQ["callTheta"]) |
                 ~np.isfinite(QQQ["callRho"]))

    print("Percent observations with NA or inf greeks %:", 100 * greeks_bad.mean())

    price_bad = (QQQ["callValue"].isna() | (QQQ["callValue"] < 0) |
                QQQ["putValue"].isna()  | (QQQ["putValue"] < 0)  |
                QQQ["callBidPrice"].isna() | (QQQ["callBidPrice"] < 0) |
                QQQ["callAskPrice"].isna() | (QQQ["callAskPrice"] < 0) |
                QQQ["putBidPrice"].isna()  | (QQQ["putBidPrice"] < 0)  |
                QQQ["putAskPrice"].isna()  | (QQQ["putAskPrice"] < 0)  )

    print("Percent observations with NA or negative option prices or values %:", 100 * price_bad.mean())

    underlying_bad = ((QQQ["spotPrice"].isna()) | (QQQ["spotPrice"] <= 0))
    print("Percent observations with NA or negative underlying prices %:", 100 * underlying_bad.mean())

    print("Percent observations with weird delta not falling in expected bounds %:", 100 * ((QQQ['callDelta'].between(0, 1)) | (QQQ['putDelta'].between(-1, 0))).mean())
    
    print("Percent observations with negative DTE %:", 100 * (QQQ['dte'] < 0).mean())

    del liq_bad, greeks_bad, price_bad, underlying_bad
    gc.collect()

    return QQQ