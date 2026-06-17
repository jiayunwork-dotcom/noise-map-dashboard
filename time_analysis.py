import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from datetime import datetime, time as dtime, timedelta
import ast


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


def _infer_event_source(event: Dict, measurements_df: pd.DataFrame) -> Dict:
    start_hour = event['start_time'].hour
    duration_min = event['duration_minutes']
    amplitude = event['amplitude_db']

    spectrum_strs = measurements_df[
        (measurements_df['measurement_time'] >= event['start_time']) &
        (measurements_df['measurement_time'] <= event['end_time'])
    ]['spectrum'].dropna().tolist()

    low_mid_energy = 0
    mid_energy = 0
    mid_high_energy = 0
    has_spectrum = False

    from data_models import FREQUENCY_BANDS
    for spec_str in spectrum_strs[:5]:
        try:
            spec_dict = ast.literal_eval(spec_str) if isinstance(spec_str, str) else spec_dict
            if isinstance(spec_dict, dict):
                for freq, val in spec_dict.items():
                    freq_int = int(freq)
                    energy = 10 ** (float(val) / 10)
                    if 63 <= freq_int <= 500:
                        low_mid_energy += energy
                    elif 250 <= freq_int <= 1000:
                        mid_energy += energy
                    elif 1000 <= freq_int <= 4000:
                        mid_high_energy += energy
                has_spectrum = True
        except Exception:
            continue

    is_traffic_peak = (7 <= start_hour <= 9) or (17 <= start_hour <= 20)
    is_night = (start_hour >= 22) or (start_hour <= 5)

    if has_spectrum:
        total = low_mid_energy + mid_energy + mid_high_energy
        if total > 0:
            low_mid_ratio = low_mid_energy / total
            mid_ratio = mid_energy / total
            mid_high_ratio = mid_high_energy / total
        else:
            low_mid_ratio = mid_ratio = mid_high_ratio = 0.33
    else:
        low_mid_ratio = mid_ratio = mid_high_ratio = 0.33

    if is_night and amplitude > 15 and duration_min <= 60:
        return {
            'source': 'construction',
            'source_name': '施工/突发噪声',
            'icon': '🏗️',
            'color': '#9C27B0',
            'reason': '夜间短时高声级脉冲，符合施工或突发噪声特征'
        }
    elif is_traffic_peak and low_mid_ratio > 0.4:
        return {
            'source': 'traffic',
            'source_name': '交通噪声',
            'icon': '🚗',
            'color': '#FF9800',
            'reason': '发生在交通高峰时段，低频成分突出'
        }
    elif duration_min >= 120 and amplitude < 20 and mid_ratio > 0.35:
        return {
            'source': 'industrial',
            'source_name': '工业噪声',
            'icon': '🏭',
            'color': '#607D8B',
            'reason': '持续时间较长且声级稳定，符合工业噪声特征'
        }
    elif mid_high_ratio > 0.4:
        return {
            'source': 'life',
            'source_name': '生活噪声',
            'icon': '👥',
            'color': '#4CAF50',
            'reason': '中高频成分突出，推测为社会生活噪声'
        }
    else:
        return {
            'source': 'unknown',
            'source_name': '待判定',
            'icon': '❓',
            'color': '#9E9E9E',
            'reason': '特征不明显，建议结合现场踏勘确认'
        }


def detect_noise_events(df: pd.DataFrame, threshold_db: float = 10.0,
                        window_hours: int = 3) -> List[Dict]:
    if df.empty or len(df) < 10:
        return []

    df_sorted = df.sort_values('measurement_time').copy()
    df_sorted = df_sorted.reset_index(drop=True)

    times = df_sorted['measurement_time'].values
    leq_vals = df_sorted['leq'].values.astype(float)
    n = len(df_sorted)
    if n < 7:
        return []

    time_deltas = []
    for i in range(1, min(n, 20)):
        dt = (pd.Timestamp(times[i]) - pd.Timestamp(times[i - 1])).total_seconds() / 60.0
        if dt > 0:
            time_deltas.append(dt)
    avg_interval = float(np.median(time_deltas)) if time_deltas else 60.0

    valid_leq = leq_vals[~np.isnan(leq_vals)]
    global_median = float(np.median(valid_leq)) if len(valid_leq) > 0 else 50.0
    global_p25 = float(np.percentile(valid_leq, 25)) if len(valid_leq) > 0 else 50.0

    background = np.full(n, np.nan)

    for i in range(n):
        if np.isnan(leq_vals[i]):
            continue

        t_center = pd.Timestamp(times[i])
        t_far_before = t_center - pd.Timedelta(hours=window_hours * 3)
        t_before = t_center - pd.Timedelta(hours=window_hours)
        t_after = t_center + pd.Timedelta(hours=window_hours)
        t_far_after = t_center + pd.Timedelta(hours=window_hours * 3)

        far_before_vals = []
        far_after_vals = []
        near_vals = []

        for j in range(n):
            if j == i:
                continue
            tj = pd.Timestamp(times[j])
            if np.isnan(leq_vals[j]):
                continue
            if t_far_before <= tj < t_before:
                far_before_vals.append(leq_vals[j])
            elif t_after < tj <= t_far_after:
                far_after_vals.append(leq_vals[j])
            elif t_before <= tj <= t_after:
                near_vals.append(leq_vals[j])

        candidates = []
        if len(far_before_vals) >= 2:
            candidates.append(float(np.percentile(far_before_vals, 35)))
        if len(far_after_vals) >= 2:
            candidates.append(float(np.percentile(far_after_vals, 35)))

        if len(candidates) == 0:
            if len(near_vals) >= 5:
                near_sorted = sorted(near_vals)
                lower_half = near_sorted[:len(near_sorted) // 2]
                candidates.append(float(np.mean(lower_half)))

        if len(candidates) > 0:
            background[i] = float(np.mean(candidates))

    fallback_bg = min(global_median, global_p25 + 3)
    for i in range(n):
        if np.isnan(background[i]):
            background[i] = fallback_bg

    exceed_mask = np.zeros(n, dtype=bool)
    for i in range(n):
        if not np.isnan(leq_vals[i]) and not np.isnan(background[i]):
            if leq_vals[i] - background[i] >= threshold_db:
                exceed_mask[i] = True

    gap_tolerance = max(1, int(round(60.0 / avg_interval)))
    events = []
    in_event = False
    event_start_idx = None
    event_indices = []
    gap_counter = 0

    for i in range(n):
        if exceed_mask[i]:
            gap_counter = 0
            if not in_event:
                in_event = True
                event_start_idx = i
                event_indices = []
            event_indices.append(i)
        else:
            if in_event:
                gap_counter += 1
                if gap_counter <= gap_tolerance and not np.isnan(leq_vals[i]):
                    if leq_vals[i] - background[i] >= threshold_db * 0.6:
                        event_indices.append(i)
                    else:
                        pass
                else:
                    if event_start_idx is not None and len(event_indices) >= 1:
                        start_idx = event_indices[0]
                        end_idx = event_indices[-1]
                        start_time = pd.Timestamp(times[start_idx])
                        end_time = pd.Timestamp(times[end_idx])
                        duration = (end_time - start_time).total_seconds() / 60.0
                        if duration <= 0:
                            duration = avg_interval

                        event_leq_vals = leq_vals[event_indices]
                        event_bg_vals = background[event_indices]
                        valid_leq_in = event_leq_vals[~np.isnan(event_leq_vals)]
                        valid_bg_in = event_bg_vals[~np.isnan(event_bg_vals)]

                        if len(valid_leq_in) > 0:
                            peak_leq = float(np.max(valid_leq_in))
                        else:
                            peak_leq = float(leq_vals[start_idx])
                        if len(valid_bg_in) > 0:
                            mean_bg = float(np.mean(valid_bg_in))
                        else:
                            mean_bg = float(background[start_idx])
                        amplitude = float(peak_leq - mean_bg)

                        if amplitude >= threshold_db * 0.85:
                            events.append({
                                'start_time': start_time,
                                'end_time': end_time,
                                'peak_leq': round(peak_leq, 1),
                                'duration_minutes': round(duration, 1),
                                'background_db': round(mean_bg, 1),
                                'amplitude_db': round(amplitude, 1),
                                'start_idx': start_idx,
                                'end_idx': end_idx
                            })
                    in_event = False
                    event_start_idx = None
                    event_indices = []
                    gap_counter = 0

    if in_event and event_start_idx is not None and len(event_indices) >= 1:
        start_idx = event_indices[0]
        end_idx = event_indices[-1]
        start_time = pd.Timestamp(times[start_idx])
        end_time = pd.Timestamp(times[end_idx])
        duration = (end_time - start_time).total_seconds() / 60.0
        if duration <= 0:
            duration = avg_interval

        event_leq_vals = leq_vals[event_indices]
        event_bg_vals = background[event_indices]
        valid_leq_in = event_leq_vals[~np.isnan(event_leq_vals)]
        valid_bg_in = event_bg_vals[~np.isnan(event_bg_vals)]

        if len(valid_leq_in) > 0:
            peak_leq = float(np.max(valid_leq_in))
        else:
            peak_leq = float(leq_vals[start_idx])
        if len(valid_bg_in) > 0:
            mean_bg = float(np.mean(valid_bg_in))
        else:
            mean_bg = float(background[start_idx])
        amplitude = float(peak_leq - mean_bg)

        if amplitude >= threshold_db * 0.85:
            events.append({
                'start_time': start_time,
                'end_time': end_time,
                'peak_leq': round(peak_leq, 1),
                'duration_minutes': round(duration, 1),
                'background_db': round(mean_bg, 1),
                'amplitude_db': round(amplitude, 1),
                'start_idx': start_idx,
                'end_idx': end_idx
            })

    merged_events = []
    for event in events:
        if not merged_events:
            merged_events.append(event)
        else:
            last = merged_events[-1]
            gap_between = (event['start_time'] - last['end_time']).total_seconds() / 60.0
            if gap_between <= avg_interval * 3:
                all_peaks = [last['peak_leq'], event['peak_leq']]
                all_bgs = [last['background_db'], event['background_db']]
                merged_events[-1] = {
                    'start_time': last['start_time'],
                    'end_time': event['end_time'],
                    'peak_leq': round(float(max(all_peaks)), 1),
                    'duration_minutes': round((event['end_time'] - last['start_time']).total_seconds() / 60.0, 1),
                    'background_db': round(float(np.mean(all_bgs)), 1),
                    'amplitude_db': round(float(max(all_peaks) - np.mean(all_bgs)), 1),
                    'start_idx': last['start_idx'],
                    'end_idx': event['end_idx']
                }
            else:
                merged_events.append(event)

    for event in merged_events:
        source_info = _infer_event_source(event, df_sorted)
        event.update(source_info)

    return merged_events


def get_event_statistics(events: List[Dict]) -> Dict:
    if not events:
        return {
            'total_events': 0,
            'avg_duration_minutes': 0,
            'max_peak_leq': 0,
            'most_frequent_hour': None,
            'hour_distribution': {}
        }

    durations = [e['duration_minutes'] for e in events]
    peaks = [e['peak_leq'] for e in events]

    hour_counts = {}
    for e in events:
        h = e['start_time'].hour
        hour_counts[h] = hour_counts.get(h, 0) + 1

    most_freq_hour = max(hour_counts, key=hour_counts.get) if hour_counts else None

    return {
        'total_events': len(events),
        'avg_duration_minutes': round(np.mean(durations), 1),
        'max_peak_leq': round(np.max(peaks), 1),
        'most_frequent_hour': most_freq_hour,
        'hour_distribution': hour_counts
    }
