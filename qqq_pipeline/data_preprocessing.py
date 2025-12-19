import numpy as np
import pandas as pd
import gc

divYield = 0.0047

def preprocessor(QQQ: pd.DataFrame) -> pd.DataFrame:

    QQQ.rename(columns={'delta': 'callDelta', 'theta' : 'callTheta', 'rho' : 'callRho'}, inplace=True)
    call_delta_idx = list(QQQ.columns).index('callDelta')
    QQQ.insert(call_delta_idx + 1, 'putDelta', QQQ['callDelta'] - np.exp(-divYield * QQQ['dte']/365))

    QQQ["tradeDate"] = pd.to_datetime(QQQ["tradeDate"])
    QQQ["expirDate"] = pd.to_datetime(QQQ["expirDate"])
    QQQ.drop(columns=["stockPrice"], inplace=True)

    del call_delta_idx
    gc.collect()

    return QQQ