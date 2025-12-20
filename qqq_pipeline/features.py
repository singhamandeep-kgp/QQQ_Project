import numpy as np
import pandas as pd
import gc

# Default option expiry dates 
DEFAULT_EXPIRY_DATES = [
    pd.Timestamp('2020-01-17'),
    pd.Timestamp('2020-02-21'),
    pd.Timestamp('2020-03-20'),
    pd.Timestamp('2020-04-09'),
    pd.Timestamp('2020-04-17'),
    pd.Timestamp('2020-05-15'),
    pd.Timestamp('2020-06-19'),
    pd.Timestamp('2020-07-02'),
    pd.Timestamp('2020-07-17'),
    pd.Timestamp('2020-08-21'),
    pd.Timestamp('2020-09-18'),
    pd.Timestamp('2020-10-16'),
    pd.Timestamp('2020-11-20'),
    pd.Timestamp('2020-12-18'),
    pd.Timestamp('2020-12-24'),
    pd.Timestamp('2020-12-31'),
    pd.Timestamp('2021-01-15'),
    pd.Timestamp('2021-02-19'),
    pd.Timestamp('2021-03-19'),
    pd.Timestamp('2021-04-01'),
    pd.Timestamp('2021-04-16'),
    pd.Timestamp('2021-05-21'),
    pd.Timestamp('2021-06-18'),
    pd.Timestamp('2021-07-16'),
    pd.Timestamp('2021-08-20'),
    pd.Timestamp('2021-09-17'),
    pd.Timestamp('2021-10-15'),
    pd.Timestamp('2021-11-19'),
    pd.Timestamp('2021-12-17'),
    pd.Timestamp('2021-12-23'),
    pd.Timestamp('2022-01-21'),
    pd.Timestamp('2022-02-18'),
    pd.Timestamp('2022-03-18'),
    pd.Timestamp('2022-04-14'),
    pd.Timestamp('2022-05-20'),
    pd.Timestamp('2022-06-17'),
    pd.Timestamp('2022-07-15'),
    pd.Timestamp('2022-08-19'),
    pd.Timestamp('2022-09-16'),
    pd.Timestamp('2022-10-21'),
    pd.Timestamp('2022-11-18'),
    pd.Timestamp('2022-12-16'),
    pd.Timestamp('2023-01-20'),
    pd.Timestamp('2023-02-17'),
    pd.Timestamp('2023-03-17'),
    pd.Timestamp('2023-04-06'),
    pd.Timestamp('2023-04-21'),
    pd.Timestamp('2023-05-19'),
    pd.Timestamp('2023-06-16'),
    pd.Timestamp('2023-07-21'),
    pd.Timestamp('2023-08-18'),
    pd.Timestamp('2023-09-15'),
    pd.Timestamp('2023-10-20'),
    pd.Timestamp('2023-11-17'),
    pd.Timestamp('2023-12-15'),
    pd.Timestamp('2024-01-19'),
    pd.Timestamp('2024-02-16'),
    pd.Timestamp('2024-03-15'),
    pd.Timestamp('2024-03-28'),
    pd.Timestamp('2024-04-19'),
    pd.Timestamp('2024-05-17'),
    pd.Timestamp('2024-06-21'),
    pd.Timestamp('2024-07-19'),
    pd.Timestamp('2024-08-16'),
    pd.Timestamp('2024-09-20'),
    pd.Timestamp('2024-10-18'),
    pd.Timestamp('2024-11-15'),
    pd.Timestamp('2024-12-20'),
    pd.Timestamp('2025-01-17'),
    pd.Timestamp('2025-02-21'),
    pd.Timestamp('2025-03-21'),
    pd.Timestamp('2025-04-17'),
    pd.Timestamp('2025-05-16'),
    pd.Timestamp('2025-06-20'),
    pd.Timestamp('2025-07-03'),
    pd.Timestamp('2025-07-18'),
    pd.Timestamp('2025-08-15'),
    pd.Timestamp('2025-09-19'),
    pd.Timestamp('2025-10-17'),
    pd.Timestamp('2025-11-21'),
    pd.Timestamp('2025-12-19')
]

DEFAULT_EXPIRY_DATES_SET = set(DEFAULT_EXPIRY_DATES)


class time_features:
    """Object-oriented wrapper for time-based feature engineering on QQQ daily data.

    Usage:
        tf = TimeFeatures()  # can pass custom expiry dates or vol windows
        daily = tf.build_features(QQQ)

    The class mirrors the previous module-level functions but groups configuration
    and behavior into an instance (expiry dates, vol windows, annualisation).
    """

    def __init__(self,
                expiry_dates: list | None = None,
                vol_windows: tuple = (5, 21, 63),
                annualisation: float = 252.0 ) -> None:
        
        self.expiry_dates = (expiry_dates if expiry_dates is not None else DEFAULT_EXPIRY_DATES)
        self.expiry_dates_set = set(self.expiry_dates)
        self.vol_windows = tuple(vol_windows)
        self.annualisation = float(annualisation)

    def prepare_daily_dataframe(self, QQQ: pd.DataFrame) -> pd.DataFrame:
        """Aggregate raw QQQ intraday/row data to a daily spot series.

        Expects `QQQ` to have columns `tradeDate` and `spotPrice`.
        """
        df = QQQ.copy()
        df["tradeDate"] = pd.to_datetime(df["tradeDate"])  # be robust
        daily = (
            df.sort_values("tradeDate")
            .groupby("tradeDate", as_index=False)
            .agg(spot=("spotPrice", "first"))
        )
        
        del df
        gc.collect()

        return daily

    def add_nextDayReturn(self, daily: pd.DataFrame) -> pd.DataFrame:
        daily = daily.copy()
        daily["ret_1d"] = daily["spot"].pct_change()
        daily["nextDayRet"] = daily["ret_1d"].shift(-1)
        return daily

    def add_realisedVolFeatures(self, daily: pd.DataFrame) -> pd.DataFrame:
        daily = daily.copy()
        for w in self.vol_windows:
            col = f"vol_{w}d"
            daily[col] = daily["ret_1d"].rolling(w).std() * np.sqrt(self.annualisation)
        return daily

    def add_calendarEffectFeatures(self, daily: pd.DataFrame) -> pd.DataFrame:
        daily = daily.copy()
        if not pd.api.types.is_datetime64_any_dtype(daily["tradeDate"]):
            daily["tradeDate"] = pd.to_datetime(daily["tradeDate"])

        daily["is_monday"] = (daily["tradeDate"].dt.weekday == 0).astype(int)
        daily["is_friday"] = (daily["tradeDate"].dt.weekday == 4).astype(int)
        daily["is_option_expiry"] = daily["tradeDate"].isin(self.expiry_dates_set).astype(int)
        daily["month"] = daily["tradeDate"].dt.month

        # month-end (last trading day in each calendar month)
        last_trading_day_in_month = daily.groupby(daily["tradeDate"].dt.to_period("M"))["tradeDate"].transform("max")
        daily["is_month_end_trading"] = (daily["tradeDate"] == last_trading_day_in_month).astype(int)
        del last_trading_day_in_month

        # quarter-end (last trading day in each calendar quarter)
        last_trading_day_in_quarter = daily.groupby(daily["tradeDate"].dt.to_period("Q"))["tradeDate"].transform("max")
        daily["is_quarter_end_trading"] = (daily["tradeDate"] == last_trading_day_in_quarter).astype(int)
        del last_trading_day_in_quarter

        # year-end (last trading day in calendar year)
        last_trading_day_in_year = daily.groupby(daily["tradeDate"].dt.to_period("Y"))["tradeDate"].transform("max")
        daily["is_year_end_trading"] = (daily["tradeDate"] == last_trading_day_in_year).astype(int)
        del last_trading_day_in_year

        gc.collect()
        return daily

    def build_features(
        self,
        QQQ: pd.DataFrame,
        compute_realised: bool = True,
        compute_calendar: bool = True) -> pd.DataFrame:
        """High level method to go from raw `QQQ` to a daily dataframe with features.

        Parameters:
            QQQ: raw dataframe with `tradeDate` and `spotPrice` columns
            compute_realised: whether to add realised volatility features
            compute_calendar: whether to add calendar effect features
        """
        daily = self.prepare_daily_dataframe(QQQ)
        daily = self.add_nextDayReturn(daily)
        if compute_realised:
            daily = self.add_realisedVolFeatures(daily)
        if compute_calendar:
            daily = self.add_calendarEffectFeatures(daily)
        return daily

def build_time_features(
    QQQ: pd.DataFrame,
    expiry_dates: list | None = None,
    vol_windows: tuple = (5, 21, 63),
    annualisation: float = 252.0,
    compute_realised: bool = True,
    compute_calendar: bool = True) -> pd.DataFrame:

    tf = time_features(expiry_dates=expiry_dates, vol_windows=vol_windows, annualisation=annualisation)
    return tf.build_features(QQQ, compute_realised=compute_realised, compute_calendar=compute_calendar)

class volume_OI_features():

    def __init__(self,
                 dte_buckets: list = [-np.inf, 7, 30, 90, 365, np.inf], 
                 dte_buckets_labels: list = ["7d", "1mo", "3mo", "12mo", "far"]) -> None:
        
        self.dte_buckets = dte_buckets
        self.dte_buckets_labels = dte_buckets_labels

    
    def calc_volume_oi(self, QQQ: pd.DataFrame) -> pd.DataFrame: 
        
        QQQ = QQQ.copy()
        QQQ["mid_call"] = (QQQ["callBidPrice"] + QQQ["callAskPrice"]) / 2
        QQQ["mid_put"] = (QQQ["putBidPrice"] + QQQ["putAskPrice"]) / 2

        QQQ["call_notvol"] = QQQ["callVolume"] * QQQ["mid_call"]
        QQQ["put_notvol"] = QQQ["putVolume"] * QQQ["mid_put"]
        QQQ["call_notoi"] = QQQ["callOpenInterest"] * QQQ["mid_call"]
        QQQ["put_notoi"] = QQQ["putOpenInterest"] * QQQ["mid_put"]

        return QQQ

    def create_delta_dte_buckets(self, QQQ: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
        
        QQQ = QQQ.copy()
        QQQ["dte_bucket"] = pd.cut(QQQ["dte"],
                                   bins= self.dte_buckets,
                                   labels=self.dte_buckets_labels)
        
        callDelta = QQQ["callDelta"]
        putDelta = QQQ["putDelta"].abs()
        QQQ["call_delta_bucket"] = "itm"
        QQQ["put_delta_bucket"] = "itm"
        QQQ.loc[callDelta.between(0.4, 0.6, inclusive="both"), "call_delta_bucket"] = "atm"
        QQQ.loc[putDelta.between(0.4, 0.6, inclusive="both"), "put_delta_bucket"] = "atm"
        QQQ.loc[callDelta < 0.4, "call_delta_bucket"] = "otm"
        QQQ.loc[putDelta < 0.4, "put_delta_bucket"] = "otm"

        call_agg = (
            QQQ.groupby(["tradeDate", "dte_bucket", "call_delta_bucket"], observed=True).agg(
            call_notvol=("call_notvol", "sum"),
            call_notoi=("call_notoi", "sum")).reset_index())

        put_agg = (QQQ.groupby(["tradeDate", "dte_bucket", "put_delta_bucket"], observed=True).agg(
            call_notvol=("call_notvol", "sum"),
            put_notvol=("put_notvol", "sum"),
            call_notoi=("call_notoi", "sum"),
            put_notoi=("put_notoi", "sum")).reset_index())

        daily_call_agg = call_agg.pivot_table(
            index="tradeDate",
            columns=["dte_bucket", "call_delta_bucket"],
            values=["call_notvol", "call_notoi"],
            aggfunc="sum",
            fill_value=0)

        daily_put_agg = put_agg.pivot_table(
            index="tradeDate",
            columns=["dte_bucket", "put_delta_bucket"],
            values=["put_notvol", "put_notoi"],
            aggfunc="sum",
            fill_value=0)
        
        daily_call_agg.columns = [
            f"{metric}_{dte}_{delta}"
            for metric, dte, delta in daily_call_agg.columns]
        
        daily_call_agg = daily_call_agg.reset_index()

        daily_put_agg.columns = [
            f"{metric}_{dte}_{delta}"
            for metric, dte, delta in daily_put_agg.columns]
        
        daily_put_agg = daily_put_agg.reset_index()

        intermediate = pd.merge(daily_put_agg, daily_call_agg, on='tradeDate', how='inner')
        daily = pd.merge(daily, intermediate, on='tradeDate', how='inner')

        del intermediate, daily_put_agg, daily_call_agg, put_agg, call_agg, callDelta, putDelta
        gc.collect()
        
        return daily
    
    def build_features(self,
                       QQQ: pd.DataFrame,
                       daily: pd.DataFrame) -> pd.DataFrame:

        QQQ = self.calc_volume_oi(QQQ)
        daily = self.create_delta_dte_buckets(QQQ, daily)

        return daily

def build_volume_oi_features(QQQ: pd.DataFrame,
                            daily: pd.DataFrame, 
                            dte_buckets: list = [-np.inf, 7, 30, 90, 365, np.inf],
                            dte_buckets_labels: list = ["7d", "1mo", "3mo", "12mo", "far"]) -> pd.DataFrame:

    voi = volume_OI_features(dte_buckets = dte_buckets, dte_buckets_labels = dte_buckets_labels)
    return voi.build_features(QQQ = QQQ, daily = daily)
    
                                            

