import json
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from data_models import get_station_measurements

ALERT_RULES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'alerts_rules.json')
ALERT_HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'alerts_history.json')

ALERT_LEVELS = {
    'info': {'name': '提示', 'color': '#2196F3', 'bg_color': '#E3F2FD'},
    'warning': {'name': '警告', 'color': '#FF9800', 'bg_color': '#FFF3E0'},
    'critical': {'name': '严重', 'color': '#F44336', 'bg_color': '#FFEBEE'}
}

METRIC_TYPES = ['leq_mean', 'leq_peak', 'event_frequency']
METRIC_NAMES = {
    'leq_mean': 'Leq均值',
    'leq_peak': 'Leq峰值',
    'event_frequency': '事件频次'
}

COMPARE_TYPES = ['greater_than', 'greater_equal', 'continuous_minutes']
COMPARE_NAMES = {
    'greater_than': '大于',
    'greater_equal': '大于等于',
    'continuous_minutes': '连续N分钟超过'
}


def _load_json_file(filepath: str, default_value) -> any:
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return default_value
    return default_value


def _save_json_file(filepath: str, data: any) -> bool:
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        return True
    except Exception as e:
        print(f"保存文件失败 {filepath}: {e}")
        return False


def load_alert_rules() -> List[Dict]:
    rules = _load_json_file(ALERT_RULES_FILE, [])
    return rules if isinstance(rules, list) else []


def save_alert_rules(rules: List[Dict]) -> bool:
    return _save_json_file(ALERT_RULES_FILE, rules)


def load_alert_history() -> List[Dict]:
    history = _load_json_file(ALERT_HISTORY_FILE, [])
    return history if isinstance(history, list) else []


def save_alert_history(history: List[Dict]) -> bool:
    return _save_json_file(ALERT_HISTORY_FILE, history)


def add_alert_rule(rule_data: Dict) -> Optional[Dict]:
    rules = load_alert_rules()
    rule_id = rule_data.get('rule_id') or f"RULE-{uuid.uuid4().hex[:8].upper()}"
    new_rule = {
        'rule_id': rule_id,
        'rule_name': rule_data.get('rule_name', ''),
        'monitor_target': rule_data.get('monitor_target', 'all'),
        'station_id': rule_data.get('station_id'),
        'metric_type': rule_data.get('metric_type', 'leq_mean'),
        'compare_type': rule_data.get('compare_type', 'greater_than'),
        'threshold': float(rule_data.get('threshold', 0)),
        'continuous_minutes': int(rule_data.get('continuous_minutes', 5)),
        'alert_level': rule_data.get('alert_level', 'warning'),
        'time_period': rule_data.get('time_period', 'all_day'),
        'start_time': rule_data.get('start_time', '08:00'),
        'end_time': rule_data.get('end_time', '20:00'),
        'enabled': rule_data.get('enabled', True),
        'created_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat()
    }
    existing = False
    for i, r in enumerate(rules):
        if r['rule_id'] == rule_id:
            new_rule['created_at'] = r.get('created_at', datetime.now().isoformat())
            rules[i] = new_rule
            existing = True
            break
    if not existing:
        rules.append(new_rule)
    if save_alert_rules(rules):
        return new_rule
    return None


def delete_alert_rule(rule_id: str) -> bool:
    rules = load_alert_rules()
    rules = [r for r in rules if r['rule_id'] != rule_id]
    return save_alert_rules(rules)


def _is_in_time_period(rule: Dict, check_time: datetime) -> bool:
    if rule.get('time_period') == 'all_day':
        return True
    start_str = rule.get('start_time', '08:00')
    end_str = rule.get('end_time', '20:00')
    try:
        start_h, start_m = map(int, start_str.split(':'))
        end_h, end_m = map(int, end_str.split(':'))
        current_minutes = check_time.hour * 60 + check_time.minute
        start_minutes = start_h * 60 + start_m
        end_minutes = end_h * 60 + end_m
        if start_minutes <= end_minutes:
            return start_minutes <= current_minutes <= end_minutes
        else:
            return current_minutes >= start_minutes or current_minutes <= end_minutes
    except Exception:
        return True


def _calculate_background_level(df: pd.DataFrame) -> float:
    if df.empty or len(df) < 5:
        return 50.0
    leq_vals = df['leq'].dropna().values
    if len(leq_vals) < 5:
        return 50.0
    return float(np.percentile(leq_vals, 25))


def _count_noise_events(df: pd.DataFrame, threshold_db: float = 5.0) -> int:
    if df.empty or len(df) < 5:
        return 0
    df_sorted = df.sort_values('measurement_time').copy()
    leq_vals = df_sorted['leq'].dropna().values
    if len(leq_vals) < 5:
        return 0
    background = float(np.percentile(leq_vals, 25))
    exceed_mask = leq_vals - background >= threshold_db
    event_count = 0
    in_event = False
    gap_count = 0
    for exceed in exceed_mask:
        if exceed:
            gap_count = 0
            if not in_event:
                in_event = True
                event_count += 1
        else:
            if in_event:
                gap_count += 1
                if gap_count > 2:
                    in_event = False
                    gap_count = 0
    return event_count


def _check_continuous_minutes(df: pd.DataFrame, threshold: float, n_minutes: int) -> Tuple[bool, Optional[datetime]]:
    if df.empty:
        return False, None
    df_sorted = df.sort_values('measurement_time').copy()
    df_sorted = df_sorted.reset_index(drop=True)
    if len(df_sorted) < n_minutes:
        return False, None
    times = df_sorted['measurement_time'].values
    leq_vals = df_sorted['leq'].values
    current_streak = 0
    streak_start_idx = None
    for i in range(len(leq_vals)):
        if np.isnan(leq_vals[i]):
            current_streak = 0
            streak_start_idx = None
            continue
        if leq_vals[i] > threshold:
            if current_streak == 0:
                streak_start_idx = i
            current_streak += 1
            if current_streak >= n_minutes:
                return True, pd.Timestamp(times[streak_start_idx])
        else:
            current_streak = 0
            streak_start_idx = None
    return False, None


def evaluate_alert_rule(rule: Dict, station_id: str, start_time: datetime, end_time: datetime) -> Optional[Dict]:
    if not rule.get('enabled', True):
        return None
    if rule.get('monitor_target') == 'single' and rule.get('station_id') != station_id:
        return None
    check_time = start_time
    if not _is_in_time_period(rule, check_time):
        return None
    df = get_station_measurements(station_id, start_time.strftime('%Y-%m-%d %H:%M:%S'), end_time.strftime('%Y-%m-%d %H:%M:%S'))
    if df.empty:
        return None
    metric_type = rule.get('metric_type', 'leq_mean')
    compare_type = rule.get('compare_type', 'greater_than')
    threshold = rule.get('threshold', 0)
    triggered = False
    measured_value = None
    trigger_time = None
    if metric_type in ['leq_mean', 'leq_peak']:
        if metric_type == 'leq_mean':
            measured_value = float(df['leq'].mean()) if not df['leq'].dropna().empty else None
        else:
            measured_value = float(df['leq'].max()) if not df['leq'].dropna().empty else None
        if measured_value is None:
            return None
        if compare_type == 'greater_than':
            triggered = measured_value > threshold
            trigger_time = start_time
        elif compare_type == 'greater_equal':
            triggered = measured_value >= threshold
            trigger_time = start_time
        elif compare_type == 'continuous_minutes':
            n = rule.get('continuous_minutes', 5)
            triggered, trig_time = _check_continuous_minutes(df, threshold, n)
            if trig_time:
                trigger_time = trig_time
            else:
                trigger_time = start_time
    elif metric_type == 'event_frequency':
        event_count = _count_noise_events(df, threshold_db=5.0)
        measured_value = float(event_count)
        if compare_type == 'greater_than':
            triggered = event_count > threshold
        elif compare_type == 'greater_equal':
            triggered = event_count >= threshold
        else:
            triggered = event_count > threshold
        trigger_time = start_time
    if triggered and measured_value is not None:
        return {
            'rule_id': rule['rule_id'],
            'rule_name': rule.get('rule_name', ''),
            'station_id': station_id,
            'alert_level': rule.get('alert_level', 'warning'),
            'metric_type': metric_type,
            'measured_value': round(measured_value, 2),
            'threshold': threshold,
            'trigger_time': trigger_time.isoformat() if isinstance(trigger_time, (datetime, pd.Timestamp)) else str(trigger_time),
            'window_start': start_time.isoformat(),
            'window_end': end_time.isoformat()
        }
    return None


def run_alert_engine(station_id: str, start_time: datetime, end_time: datetime) -> List[Dict]:
    rules = load_alert_rules()
    triggered_alerts = []
    for rule in rules:
        result = evaluate_alert_rule(rule, station_id, start_time, end_time)
        if result:
            triggered_alerts.append(result)
    if triggered_alerts:
        history = load_alert_history()
        for alert in triggered_alerts:
            alert_record = {
                'alert_id': f"ALERT-{uuid.uuid4().hex[:8].upper()}",
                'alert_time': alert['trigger_time'],
                'rule_name': alert['rule_name'],
                'rule_id': alert['rule_id'],
                'station_id': alert['station_id'],
                'measured_value': alert['measured_value'],
                'threshold': alert['threshold'],
                'alert_level': alert['alert_level'],
                'metric_type': alert['metric_type'],
                'window_start': alert['window_start'],
                'window_end': alert['window_end'],
                'acknowledged': False,
                'created_at': datetime.now().isoformat()
            }
            history.append(alert_record)
        save_alert_history(history)
    return triggered_alerts


def run_alert_engine_all_stations(station_ids: List[str], start_time: datetime, end_time: datetime) -> List[Dict]:
    all_alerts = []
    for sid in station_ids:
        alerts = run_alert_engine(sid, start_time, end_time)
        all_alerts.extend(alerts)
    return all_alerts


def get_alert_statistics() -> Dict:
    history = load_alert_history()
    now = datetime.now()
    today_start = datetime(now.year, now.month, now.day)
    week_start = today_start - timedelta(days=now.weekday())
    today_alerts = [a for a in history if datetime.fromisoformat(a['alert_time'].replace('Z', '+00:00')) >= today_start]
    week_alerts = [a for a in history if datetime.fromisoformat(a['alert_time'].replace('Z', '+00:00')) >= week_start]
    level_counts = {'info': 0, 'warning': 0, 'critical': 0}
    for a in history:
        level = a.get('alert_level', 'info')
        if level in level_counts:
            level_counts[level] += 1
    daily_counts = {}
    for a in history:
        try:
            alert_dt = datetime.fromisoformat(a['alert_time'].replace('Z', '+00:00'))
            date_key = alert_dt.strftime('%Y-%m-%d')
            daily_counts[date_key] = daily_counts.get(date_key, 0) + 1
        except Exception:
            pass
    last_7_days = []
    for i in range(6, -1, -1):
        day = now - timedelta(days=i)
        date_key = day.strftime('%Y-%m-%d')
        last_7_days.append({'date': date_key, 'count': daily_counts.get(date_key, 0)})
    station_counts = {}
    for a in history:
        sid = a.get('station_id', 'unknown')
        station_counts[sid] = station_counts.get(sid, 0) + 1
    top_stations = sorted(station_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    return {
        'today_count': len(today_alerts),
        'week_count': len(week_alerts),
        'total_count': len(history),
        'level_counts': level_counts,
        'daily_trend': last_7_days,
        'top_stations': top_stations
    }


def get_alerts_by_station(station_id: str) -> List[Dict]:
    history = load_alert_history()
    return [a for a in history if a.get('station_id') == station_id]


def has_active_alerts(station_id: str, lookback_hours: int = 24) -> bool:
    history = load_alert_history()
    cutoff = datetime.now() - timedelta(hours=lookback_hours)
    for a in history:
        if a.get('station_id') == station_id:
            try:
                alert_time = datetime.fromisoformat(a['alert_time'].replace('Z', '+00:00'))
                if alert_time >= cutoff:
                    return True
            except Exception:
                pass
    return False
