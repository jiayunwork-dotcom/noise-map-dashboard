import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from datetime import datetime, time as dtime


DAY_START_HOUR = 6
DAY_END_HOUR = 22


def _is_daytime(hour: int) -> bool:
    return DAY_START_HOUR <= hour < DAY_END_HOUR


def calculate_ldn(leq_values: np.ndarray, hours: np.ndarray) -> Optional[float]:
    if len(leq_values) == 0 or len(hours) == 0:
        return None
    
    day_mask = np.array([_is_daytime(int(h)) for h in hours])
    night_mask = ~day_mask
    
    day_values = leq_values[day_mask]
    night_values = leq_values[night_mask]
    
    if len(day_values) == 0 or len(night_values) == 0:
        return None
    
    ld = 10 * np.log10(np.mean(10 ** (day_values / 10)))
    ln = 10 * np.log10(np.mean(10 ** (night_values / 10)))
    
    ldn = 10 * np.log10(
        (15 * 10 ** (ld / 10) + 9 * 10 ** ((ln + 10) / 10)) / 24
    )
    
    return float(ldn)


def get_ldn_components(leq_values: np.ndarray, hours: np.ndarray) -> Dict:
    result = {
        'ld': None,
        'ln': None,
        'ldn': None,
        'day_count': 0,
        'night_count': 0
    }
    
    if len(leq_values) == 0:
        return result
    
    day_mask = np.array([_is_daytime(int(h)) for h in hours])
    night_mask = ~day_mask
    
    day_values = leq_values[day_mask]
    night_values = leq_values[night_mask]
    
    result['day_count'] = len(day_values)
    result['night_count'] = len(night_values)
    
    if len(day_values) > 0:
        result['ld'] = float(10 * np.log10(np.mean(10 ** (day_values / 10))))
    if len(night_values) > 0:
        result['ln'] = float(10 * np.log10(np.mean(10 ** (night_values / 10))))
    
    if result['ld'] is not None and result['ln'] is not None:
        result['ldn'] = float(10 * np.log10(
            (15 * 10 ** (result['ld'] / 10) + 9 * 10 ** ((result['ln'] + 10) / 10)) / 24
        ))
    
    return result


def calculate_weekly_pattern(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    
    df_copy = df.copy()
    df_copy['weekday'] = df_copy['measurement_time'].dt.dayofweek
    df_copy['weekday_name'] = df_copy['weekday'].map({
        0: '周一', 1: '周二', 2: '周三', 3: '周四',
        4: '周五', 5: '周六', 6: '周日'
    })
    
    result = df_copy.groupby(['weekday', 'weekday_name'])['leq'].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('median', 'median'),
        ('min', 'min'),
        ('max', 'max'),
        ('std', 'std')
    ]).reset_index()
    
    result = result.sort_values('weekday')
    
    return result


def calculate_hourly_pattern(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[int]]:
    if df.empty:
        return pd.DataFrame(), []
    
    df_copy = df.copy()
    df_copy['hour'] = df_copy['measurement_time'].dt.hour
    
    result = df_copy.groupby('hour')['leq'].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('median', 'median'),
        ('min', 'min'),
        ('max', 'max'),
        ('std', 'std')
    ]).reset_index()
    
    all_hours = pd.DataFrame({'hour': range(24)})
    result = all_hours.merge(result, on='hour', how='left')
    result = result.fillna(0)
    
    daily_mean = np.mean(df_copy['leq'])
    threshold = daily_mean + 5
    
    hourly_means = result['mean'].values
    peak_hours = []
    
    in_peak = False
    peak_start = None
    
    for h in range(24):
        if hourly_means[h] > threshold and hourly_means[h] > 0:
            if not in_peak:
                in_peak = True
                peak_start = h
        else:
            if in_peak:
                peak_hours.extend(range(peak_start, h))
                in_peak = False
                peak_start = None
    
    if in_peak and peak_start is not None:
        peak_hours.extend(range(peak_start, 24))
    
    result['is_peak'] = result['hour'].isin(peak_hours)
    result['threshold'] = threshold
    result['daily_mean'] = daily_mean
    
    return result, peak_hours


def calculate_monthly_trend(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    if df.empty:
        return pd.DataFrame(), {}
    
    df_copy = df.copy()
    df_copy['year_month'] = df_copy['measurement_time'].dt.to_period('M')
    df_copy['year'] = df_copy['measurement_time'].dt.year
    df_copy['month'] = df_copy['measurement_time'].dt.month
    
    result = df_copy.groupby(['year_month', 'year', 'month'])['leq'].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('median', 'median'),
        ('min', 'min'),
        ('max', 'max'),
        ('std', 'std')
    ]).reset_index()
    
    result['year_month_str'] = result['year_month'].astype(str)
    result = result.sort_values('year_month')
    
    trend_info = {}
    
    if len(result) >= 3:
        x = np.arange(len(result))
        y = result['mean'].values
        
        valid_mask = ~np.isnan(y)
        if np.sum(valid_mask) >= 3:
            x_valid = x[valid_mask]
            y_valid = y[valid_mask]
            
            slope, intercept = np.polyfit(x_valid, y_valid, 1)
            
            trend_info['slope'] = float(slope)
            trend_info['intercept'] = float(intercept)
            
            if abs(slope) < 0.05:
                trend_info['trend'] = '稳定'
                trend_info['trend_color'] = '#2196F3'
            elif slope > 0:
                trend_info['trend'] = '恶化'
                trend_info['trend_color'] = '#F44336'
            else:
                trend_info['trend'] = '改善'
                trend_info['trend_color'] = '#4CAF50'
            
            trend_info['monthly_change_dB'] = float(slope)
            
            y_pred = slope * x + intercept
            ss_res = np.sum((y_valid - (slope * x_valid + intercept)) ** 2)
            ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            trend_info['r_squared'] = float(r_squared)
            
            trend_info['predicted'] = y_pred.tolist()
    
    return result, trend_info


def calculate_quarterly_trend(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    if df.empty:
        return pd.DataFrame(), {}
    
    df_copy = df.copy()
    df_copy['quarter'] = df_copy['measurement_time'].dt.to_period('Q')
    df_copy['year'] = df_copy['measurement_time'].dt.year
    df_copy['q'] = df_copy['measurement_time'].dt.quarter
    
    result = df_copy.groupby(['quarter', 'year', 'q'])['leq'].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('median', 'median'),
        ('min', 'min'),
        ('max', 'max'),
        ('std', 'std')
    ]).reset_index()
    
    result['quarter_str'] = result['quarter'].astype(str)
    result = result.sort_values('quarter')
    
    trend_info = {}
    
    if len(result) >= 3:
        x = np.arange(len(result))
        y = result['mean'].values
        
        valid_mask = ~np.isnan(y)
        if np.sum(valid_mask) >= 3:
            x_valid = x[valid_mask]
            y_valid = y[valid_mask]
            
            slope, intercept = np.polyfit(x_valid, y_valid, 1)
            
            trend_info['slope'] = float(slope)
            trend_info['intercept'] = float(intercept)
            
            if abs(slope) < 0.1:
                trend_info['trend'] = '稳定'
                trend_info['trend_color'] = '#2196F3'
            elif slope > 0:
                trend_info['trend'] = '恶化'
                trend_info['trend_color'] = '#F44336'
            else:
                trend_info['trend'] = '改善'
                trend_info['trend_color'] = '#4CAF50'
            
            trend_info['quarterly_change_dB'] = float(slope)
            
            y_pred = slope * x + intercept
            ss_res = np.sum((y_valid - (slope * x_valid + intercept)) ** 2)
            ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            trend_info['r_squared'] = float(r_squared)
            trend_info['predicted'] = y_pred.tolist()
    
    return result, trend_info


def get_station_time_analysis(df: pd.DataFrame, station_name: str = '') -> Dict:
    results = {
        'station_name': station_name,
        'basic_stats': {},
        'ldn': {},
        'weekly': None,
        'hourly': None,
        'peak_hours': [],
        'monthly': None,
        'monthly_trend': {},
        'quarterly': None,
        'quarterly_trend': {}
    }
    
    if df.empty:
        return results
    
    results['basic_stats'] = {
        'total_records': len(df),
        'date_range': f"{df['measurement_time'].min().strftime('%Y-%m-%d')} 至 {df['measurement_time'].max().strftime('%Y-%m-%d')}",
        'leq_mean': float(df['leq'].mean()),
        'leq_median': float(df['leq'].median()),
        'leq_min': float(df['leq'].min()),
        'leq_max': float(df['leq'].max()),
        'leq_std': float(df['leq'].std())
    }
    
    leq_vals = df['leq'].values
    hours = df['measurement_time'].dt.hour.values
    results['ldn'] = get_ldn_components(leq_vals, hours)
    
    results['weekly'] = calculate_weekly_pattern(df)
    results['hourly'], results['peak_hours'] = calculate_hourly_pattern(df)
    results['monthly'], results['monthly_trend'] = calculate_monthly_trend(df)
    results['quarterly'], results['quarterly_trend'] = calculate_quarterly_trend(df)
    
    return results
