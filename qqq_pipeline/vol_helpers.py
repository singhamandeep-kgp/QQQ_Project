import numpy as np
import pandas as pd
import gc
import logging
from scipy.stats import norm
from scipy.optimize import brentq

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ============================================================================
# Black-Scholes Helper Functions for IV Calculation
# ============================================================================

def black_scholes_call(S, K, T, r, q, sigma):
    """Calculate Black-Scholes call option price"""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = S * norm.cdf(d1) - K * np.exp(-(r-q) * T) * norm.cdf(d2)
    return call_price


def black_scholes_put(S, K, T, r, q, sigma):
    """Calculate Black-Scholes put option price"""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    put_price = K * np.exp(-(r-q) * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price


def implied_volatility_call(price, S, K, T, r=0.025, q = 0.0047):
    """
    Calculate implied volatility for a call option using Brent's method
    Returns np.nan if solution not found
    """
    if T <= 0 or price <= 0:
        return np.nan

    # Check intrinsic value
    intrinsic = max(S - K, 0)
    if price <= intrinsic * 1.001:  # Allow tiny time value
        return np.nan

    try:
        def objective(sigma):
            return black_scholes_call(S, K, T, r, q, sigma) - price

        # Brent's method between 1% and 500% vol
        iv = brentq(objective, 0.01, 5.0, xtol=1e-6, maxiter=100)
        return iv
    except (ValueError, RuntimeError):
        return np.nan
	

def implied_volatility_put(price, S, K, T, r=0.025, q = 0.0047):
    """
    Calculate implied volatility for a put option using Brent's method
    Returns np.nan if solution not found
    """
    if T <= 0 or price <= 0:
        return np.nan

    # Check intrinsic value
    intrinsic = max(K - S, 0)
    if price <= intrinsic * 1.001:  # Allow tiny time value
        return np.nan

    try:
        def objective(sigma):
            return black_scholes_put(S, K, T, r, q, sigma) - price

        # Brent's method between 1% and 500% vol
        iv = brentq(objective, 0.01, 5.0, xtol=1e-6, maxiter=100)
        return iv
    except (ValueError, RuntimeError):
        return np.nan
	

def interpolate_iv_by_variance(target_d: int, iv_lower: float, iv_upper: float, dte_lower: float, dte_upper: float) -> float:
	"""Interpolate implied volatility at target DTE using linear variance interpolation.
	
	Returns:
		interpolated implied vol (float) or np.nan on error
	"""
	try:
		if any(pd.isna(x) for x in (iv_lower, iv_upper, dte_lower, dte_upper)):
			return np.nan

		T_target = float(target_d) / 365.0
		T_l = float(dte_lower) / 365.0
		T_u = float(dte_upper) / 365.0
		if T_u <= T_l or T_target <= 0:
			return np.nan

		V_l = (float(iv_lower) ** 2) * T_l
		V_u = (float(iv_upper) ** 2) * T_u
		# linear interpolation of total variance
		V_t = V_l + (T_target - T_l) / (T_u - T_l) * (V_u - V_l)
		if V_t < 0:
			return np.nan

		iv_t = np.sqrt(max(V_t, 0.0) / T_target)
		return float(iv_t)
	except Exception:
		return np.nan

def pick_itm(df_subset, spot):
    """Return a dict with call/put info for nearest ITM within df_subset."""
    res = {
        'call_price': np.nan,
        'call_strike': np.nan,
        'call_dte': np.nan,
        'put_price': np.nan,
        'put_strike': np.nan,
        'put_dte': np.nan,
    }
    if df_subset is None or df_subset.empty:
        return res

    # Calls: strike < spot -> pick max(strike)
    calls = df_subset[df_subset['strike'] < spot]
    if not calls.empty:
        call_row = calls.loc[calls['strike'].idxmax()]
        res['call_strike'] = call_row['strike']
        res['call_price'] = call_row['mid_call']
        res['call_dte'] = call_row['dte']

    # Puts: strike > spot -> pick min(strike)
    puts = df_subset[df_subset['strike'] > spot]
    if not puts.empty:
        put_row = puts.loc[puts['strike'].idxmin()]
        res['put_strike'] = put_row['strike']
        res['put_price'] = put_row['mid_put']
        res['put_dte'] = put_row['dte']
        
    return res

# ============================================================================
# Main Functions for IV Calculation
# ============================================================================

def find_atm_straddle_iv(
	QQQ: pd.DataFrame,
	daily: pd.DataFrame,
	target_dte: int = None,
	r: float = 0.025,
	q: float = 0.0047,
) -> pd.DataFrame:
	"""
	Find nearest ITM call and put (ATM straddle) for each date and given target DTE.

	Iterates over all rows in daily. For each date, if options with DTE == target_dte 
	exist, pick the nearest ITM call (strike < spot, largest strike) and nearest ITM put 
	(strike > spot, smallest strike). If none exist for the exact DTE, find the nearest 
	available DTE below and above the target and store both for interpolation.

	Returns the updated daily with all rows filled.
	"""

	daily = daily.copy()

	d = int(target_dte)
	logging.info(f"Processing DTE: {d} days")
	
	# Initialize all IV columns for this target DTE upfront
	daily[f'call_iv_{d}d'] = np.nan
	daily[f'put_iv_{d}d'] = np.nan
	daily[f'call_iv_{d}d_lower'] = np.nan
	daily[f'put_iv_{d}d_lower'] = np.nan
	daily[f'call_iv_{d}d_upper'] = np.nan
	daily[f'put_iv_{d}d_upper'] = np.nan

	for idx, row in daily.iterrows():
		trade_date = row['tradeDate']
		logging.info(f"Processing date: {trade_date}")
		spot = daily.at[idx, "spot"]
		date_options = QQQ[QQQ['tradeDate'] == trade_date]
		
		if date_options.empty:
			logging.warning(f"No options available for date {trade_date}")
			continue

		available_dtes = sorted(date_options['dte'])

		# exact matches (allow small tolerance)
		exact = date_options[date_options['dte'] == d]

		if not exact.empty:
			logging.info(f"Found exact DTE match ({d}d) for {trade_date}")
			picked = pick_itm(exact, spot)
			# assign into daily

			# --- compute implied vols for exact DTE ---
			call_iv = np.nan
			put_iv = np.nan
			try:
				if not pd.isna(picked['call_price']) and not pd.isna(picked['call_dte']) and picked['call_dte'] > 0:
					T_call = float(picked['call_dte']) / 365.0
					call_iv = implied_volatility_call(float(picked['call_price']), float(spot), float(picked['call_strike']), T_call, r=r, q=q)
				if not pd.isna(picked['put_price']) and not pd.isna(picked['put_dte']) and picked['put_dte'] > 0:
					T_put = float(picked['put_dte']) / 365.0
					put_iv = implied_volatility_put(float(picked['put_price']), float(spot), float(picked['put_strike']), T_put, r=r, q=q)
			except Exception:
				call_iv = np.nan
				put_iv = np.nan

			logging.info(f"Exact DTE {d}d for {trade_date}: call_iv={call_iv:.4f}, put_iv={put_iv:.4f}")
			daily.at[idx, f'call_iv_{d}d'] = call_iv
			daily.at[idx, f'put_iv_{d}d'] = put_iv
		else:
			logging.info(f"No exact DTE match ({d}d) for {trade_date}, will interpolate from lower/upper")
			# need lower and upper DTE candidates to allow interpolation later
			lower = None
			upper = None
			call_iv_lower = np.nan
			put_iv_lower = np.nan
			call_iv_upper = np.nan
			put_iv_upper = np.nan
			
			for ad in available_dtes:
				if ad < d:
					lower = ad
				if ad > d and upper is None:
					upper = ad
					break
			# attempt to pick from lower and upper
			if lower is not None:
				lower_subset = date_options[date_options['dte']==lower]
				picked_lower = pick_itm(lower_subset, spot)
				
				# compute IVs for lower dte
				call_iv_lower = np.nan
				put_iv_lower = np.nan
				try:
					if not pd.isna(picked_lower['call_price']) and not pd.isna(picked_lower['call_dte']) and picked_lower['call_dte'] > 0:
						T_l = float(picked_lower['call_dte']) / 365.0
						call_iv_lower = implied_volatility_call(float(picked_lower['call_price']), float(spot), float(picked_lower['call_strike']), T_l, r=r, q=q)
					if not pd.isna(picked_lower['put_price']) and not pd.isna(picked_lower['put_dte']) and picked_lower['put_dte'] > 0:
						T_l_put = float(picked_lower['put_dte']) / 365.0
						put_iv_lower = implied_volatility_put(float(picked_lower['put_price']), float(spot), float(picked_lower['put_strike']), T_l_put, r=r, q=q)
				except Exception:
					call_iv_lower = np.nan
					put_iv_lower = np.nan

				daily.at[idx, f'call_iv_{d}d_lower'] = call_iv_lower
				daily.at[idx, f'put_iv_{d}d_lower'] = put_iv_lower

			if upper is not None:
				upper_subset = date_options[date_options['dte'] == upper]
				picked_upper = pick_itm(upper_subset, spot)
				
				# compute IVs for upper dte
				call_iv_upper = np.nan
				put_iv_upper = np.nan
				try:
					if not pd.isna(picked_upper['call_price']) and not pd.isna(picked_upper['call_dte']) and picked_upper['call_dte'] > 0:
						T_u = float(picked_upper['call_dte']) / 365.0
						call_iv_upper = implied_volatility_call(float(picked_upper['call_price']), float(spot), float(picked_upper['call_strike']), T_u, r=r, q=q)
					if not pd.isna(picked_upper['put_price']) and not pd.isna(picked_upper['put_dte']) and picked_upper['put_dte'] > 0:
						T_u_put = float(picked_upper['put_dte']) / 365.0
						put_iv_upper = implied_volatility_put(float(picked_upper['put_price']), float(spot), float(picked_upper['put_strike']), T_u_put, r=r, q=q)
				except Exception:
					call_iv_upper = np.nan
					put_iv_upper = np.nan

				daily.at[idx, f'call_iv_{d}d_upper'] = call_iv_upper
				daily.at[idx, f'put_iv_{d}d_upper'] = put_iv_upper

			# if both lower and upper IVs exist, do linear variance interpolation to target d
			if lower is not None and upper is not None:
				try:
					# interpolated call IV
					if (not pd.isna(call_iv_lower) and not pd.isna(call_iv_upper)):
						call_iv_t = interpolate_iv_by_variance(d, call_iv_lower, call_iv_upper, picked_lower['call_dte'], picked_upper['call_dte'])
						daily.at[idx, f'call_iv_{d}d'] = call_iv_t
						logging.info(f"Interpolated {d}d call_iv for {trade_date} from {lower}d to {upper}d: {call_iv_t:.4f}")
					# interpolated put IV
					if (not pd.isna(put_iv_lower) and not pd.isna(put_iv_upper)):
						put_iv_t = interpolate_iv_by_variance(d, put_iv_lower, put_iv_upper, picked_lower['put_dte'], picked_upper['put_dte'])
						daily.at[idx, f'put_iv_{d}d'] = put_iv_t
						logging.info(f"Interpolated {d}d put_iv for {trade_date} from {lower}d to {upper}d: {put_iv_t:.4f}")
				except Exception:
					pass
	
	daily = daily.drop(columns=[f'call_iv_{d}d_lower', f'put_iv_{d}d_lower', f'call_iv_{d}d_upper', f'put_iv_{d}d_upper'], errors='ignore')
	
	gc.collect()

	return daily
    

