import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from data_models import FREQUENCY_BANDS
import ast

NOISE_SOURCE_TYPES = {
    'traffic': {
        'name': '交通噪声',
        'description': '道路交通噪声',
        'color': '#FF9800',
        'icon': '🚗'
    },
    'construction': {
        'name': '施工噪声',
        'description': '建筑施工噪声',
        'color': '#9C27B0',
        'icon': '🏗️'
    },
    'industrial': {
        'name': '工业噪声',
        'description': '工业生产噪声',
        'color': '#607D8B',
        'icon': '🏭'
    },
    'life': {
        'name': '生活噪声',
        'description': '社会生活噪声',
        'color': '#4CAF50',
        'icon': '👥'
    },
    'unknown': {
        'name': '未知类型',
        'description': '无法确定的噪声源',
        'color': '#9E9E9E',
        'icon': '❓'
    }
}


def parse_spectrum_string(spectrum_str: str) -> Optional[Dict[int, float]]:
    if not spectrum_str or pd.isna(spectrum_str):
        return None
    try:
        if isinstance(spectrum_str, dict):
            return {int(k): float(v) for k, v in spectrum_str.items()}
        parsed = ast.literal_eval(spectrum_str)
        if isinstance(parsed, dict):
            return {int(k): float(v) for k, v in parsed.items()}
    except Exception:
        pass
    return None


def spectrum_to_array(spectrum_dict: Dict[int, float]) -> Optional[np.ndarray]:
    if not spectrum_dict:
        return None
    
    arr = np.full(len(FREQUENCY_BANDS), np.nan)
    for i, freq in enumerate(FREQUENCY_BANDS):
        if freq in spectrum_dict:
            arr[i] = spectrum_dict[freq]
    
    return arr


def get_frequency_band_range(bands: List[int], freq_bands: List[int] = FREQUENCY_BANDS) -> List[int]:
    indices = []
    for i, f in enumerate(freq_bands):
        if bands[0] <= f <= bands[-1]:
            indices.append(i)
    return indices


def calculate_band_energy(spectrum_arr: np.ndarray, band_indices: List[int]) -> float:
    if len(band_indices) == 0:
        return 0.0
    
    valid_values = spectrum_arr[band_indices]
    valid_values = valid_values[~np.isnan(valid_values)]
    
    if len(valid_values) == 0:
        return 0.0
    
    return float(np.mean(10 ** (valid_values / 10)))


def detect_spectral_peaks(spectrum_arr: np.ndarray, threshold: float = 3.0) -> List[Dict]:
    peaks = []
    valid_mask = ~np.isnan(spectrum_arr)
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) < 3:
        return peaks
    
    for i in range(1, len(valid_indices) - 1):
        idx = valid_indices[i]
        prev_idx = valid_indices[i - 1]
        next_idx = valid_indices[i + 1]
        
        val = spectrum_arr[idx]
        prev_val = spectrum_arr[prev_idx]
        next_val = spectrum_arr[next_idx]
        
        if val > prev_val + threshold and val > next_val + threshold:
            local_vals = spectrum_arr[valid_indices[max(0, i-3):min(len(valid_indices), i+4)]]
            local_mean = np.nanmean(local_vals)
            
            if val > local_mean + threshold:
                peaks.append({
                    'frequency': FREQUENCY_BANDS[idx],
                    'index': idx,
                    'value': float(val),
                    'excess': float(val - local_mean)
                })
    
    if len(peaks) > 5:
        peaks = sorted(peaks, key=lambda p: p['excess'], reverse=True)[:5]
        peaks = sorted(peaks, key=lambda p: p['index'])
    
    return peaks


def calculate_spectrum_rolloff(spectrum_arr: np.ndarray, percentage: float = 0.85) -> Optional[int]:
    valid_mask = ~np.isnan(spectrum_arr)
    if np.sum(valid_mask) < 5:
        return None
    
    energies = 10 ** (spectrum_arr[valid_mask] / 10)
    total_energy = np.sum(energies)
    
    if total_energy <= 0:
        return None
    
    cumulative = np.cumsum(energies)
    threshold = total_energy * percentage
    
    rolloff_idx = np.argmax(cumulative >= threshold)
    valid_freqs = np.array(FREQUENCY_BANDS)[valid_mask]
    
    if rolloff_idx < len(valid_freqs):
        return int(valid_freqs[rolloff_idx])
    return int(valid_freqs[-1])


def calculate_spectrum_centroid(spectrum_arr: np.ndarray) -> Optional[float]:
    valid_mask = ~np.isnan(spectrum_arr)
    if np.sum(valid_mask) < 3:
        return None
    
    freqs = np.array(FREQUENCY_BANDS)[valid_mask]
    energies = 10 ** (spectrum_arr[valid_mask] / 10)
    total_energy = np.sum(energies)
    
    if total_energy <= 0:
        return None
    
    centroid = np.sum(freqs * energies) / total_energy
    return float(centroid)


def identify_noise_source(leq: float, lmax: Optional[float],
                          l10: Optional[float], l50: Optional[float], l90: Optional[float],
                          spectrum_arr: np.ndarray,
                          hourly_data: Optional[pd.DataFrame] = None) -> Dict:
    scores = {
        'traffic': 0.0,
        'construction': 0.0,
        'industrial': 0.0,
        'life': 0.0
    }
    
    confidence_factors = []
    
    low_mid_bands = get_frequency_band_range([63, 500])
    mid_bands = get_frequency_band_range([250, 1000])
    mid_high_bands = get_frequency_band_range([1000, 4000])
    
    low_mid_energy = calculate_band_energy(spectrum_arr, low_mid_bands) if spectrum_arr is not None else 0
    mid_energy = calculate_band_energy(spectrum_arr, mid_bands) if spectrum_arr is not None else 0
    mid_high_energy = calculate_band_energy(spectrum_arr, mid_high_bands) if spectrum_arr is not None else 0
    
    total_energy = low_mid_energy + mid_energy + mid_high_energy
    
    if total_energy > 0:
        low_mid_ratio = low_mid_energy / total_energy
        mid_ratio = mid_energy / total_energy
        mid_high_ratio = mid_high_energy / total_energy
        confidence_factors.append(0.3)
    else:
        low_mid_ratio = 0.33
        mid_ratio = 0.33
        mid_high_ratio = 0.33
        confidence_factors.append(0.1)
    
    if low_mid_ratio > 0.5:
        scores['traffic'] += 30 * low_mid_ratio
    if mid_ratio > 0.5:
        scores['construction'] += 25 * mid_ratio
    if mid_high_ratio > 0.4:
        scores['life'] += 30 * mid_high_ratio
    
    if l10 is not None and l90 is not None:
        l10_l90_diff = l10 - l90
        confidence_factors.append(0.2)
        
        if l10_l90_diff > 10:
            scores['traffic'] += 25
            scores['construction'] += 15
        elif l10_l90_diff < 5:
            scores['industrial'] += 20
        else:
            scores['life'] += 10
    
    if lmax is not None:
        lmax_leq_diff = lmax - leq
        confidence_factors.append(0.2)
        
        if lmax_leq_diff > 15:
            scores['construction'] += 30
        elif lmax_leq_diff > 10:
            scores['traffic'] += 15
            scores['life'] += 10
        elif lmax_leq_diff < 5:
            scores['industrial'] += 20
    
    peaks = []
    if spectrum_arr is not None:
        peaks = detect_spectral_peaks(spectrum_arr, threshold=4.0)
        confidence_factors.append(0.15)
        
        if len(peaks) >= 2:
            sharp_peaks = [p for p in peaks if p['excess'] > 6]
            if len(sharp_peaks) >= 1:
                scores['industrial'] += 25
            else:
                scores['construction'] += 10
        elif len(peaks) == 1:
            if peaks[0]['excess'] > 8:
                scores['industrial'] += 20
            elif 250 <= peaks[0]['frequency'] <= 1000:
                scores['construction'] += 15
    
    if hourly_data is not None and len(hourly_data) > 12:
        confidence_factors.append(0.15)
        
        hours = hourly_data['hour'].values
        means = hourly_data['mean'].values
        
        morning_mask = (hours >= 7) & (hours <= 9)
        evening_mask = (hours >= 18) & (hours <= 21)
        night_mask = (hours >= 22) | (hours <= 5)
        
        if np.any(morning_mask) and np.any(evening_mask):
            morning_mean = np.mean(means[morning_mask])
            evening_mean = np.mean(means[evening_mask])
            overall_mean = np.mean(means[means > 0])
            
            if overall_mean > 0:
                morning_ratio = morning_mean / overall_mean
                evening_ratio = evening_mean / overall_mean
                
                if morning_ratio > 1.1 and evening_ratio > 1.1:
                    scores['traffic'] += 20
                    scores['life'] += 15
                elif evening_ratio > 1.15:
                    scores['life'] += 25
    
    max_score = max(scores.values()) if scores else 0
    
    if max_score <= 5:
        source_type = 'unknown'
        confidence = 0.0
    else:
        source_type = max(scores, key=scores.get)
        
        total_scores = sum(scores.values())
        if total_scores > 0:
            confidence = (scores[source_type] / total_scores) * 100
        else:
            confidence = 0.0
        
        base_confidence = 50 if len(confidence_factors) > 0 else 20
        factor_quality = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
        confidence = min(95, base_confidence + confidence * factor_quality * 0.5)
    
    sorted_sources = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    result = {
        'primary_source': source_type,
        'primary_source_name': NOISE_SOURCE_TYPES[source_type]['name'],
        'confidence': round(confidence, 1),
        'scores': {k: round(v, 1) for k, v in scores.items()},
        'ranked_sources': [
            {
                'type': src,
                'name': NOISE_SOURCE_TYPES[src]['name'],
                'score': round(score, 1),
                'color': NOISE_SOURCE_TYPES[src]['color'],
                'icon': NOISE_SOURCE_TYPES[src]['icon']
            }
            for src, score in sorted_sources
        ],
        'features': {
            'low_mid_ratio': round(low_mid_ratio * 100, 1),
            'mid_ratio': round(mid_ratio * 100, 1),
            'mid_high_ratio': round(mid_high_ratio * 100, 1),
            'l10_l90_diff': round(l10 - l90, 1) if l10 is not None and l90 is not None else None,
            'lmax_leq_diff': round(lmax - leq, 1) if lmax is not None else None,
            'peak_count': len(peaks),
            'peaks': peaks
        }
    }
    
    return result


def get_station_source_analysis(measurements_df: pd.DataFrame,
                                hourly_data: Optional[pd.DataFrame] = None) -> Dict:
    if measurements_df.empty:
        return {}
    
    spectrum_values = []
    for _, row in measurements_df.iterrows():
        spec_dict = parse_spectrum_string(row.get('spectrum'))
        if spec_dict:
            arr = spectrum_to_array(spec_dict)
            if arr is not None:
                spectrum_values.append(arr)
    
    avg_spectrum = None
    if spectrum_values:
        stacked = np.vstack(spectrum_values)
        avg_spectrum = np.nanmean(stacked, axis=0)
    
    avg_leq = float(measurements_df['leq'].mean())
    avg_lmax = float(measurements_df['lmax'].mean()) if 'lmax' in measurements_df.columns and measurements_df['lmax'].notna().any() else None
    avg_l10 = float(measurements_df['l10'].mean()) if 'l10' in measurements_df.columns and measurements_df['l10'].notna().any() else None
    avg_l50 = float(measurements_df['l50'].mean()) if 'l50' in measurements_df.columns and measurements_df['l50'].notna().any() else None
    avg_l90 = float(measurements_df['l90'].mean()) if 'l90' in measurements_df.columns and measurements_df['l90'].notna().any() else None
    
    source_result = identify_noise_source(
        avg_leq, avg_lmax, avg_l10, avg_l50, avg_l90,
        avg_spectrum, hourly_data
    )
    
    centroid = calculate_spectrum_centroid(avg_spectrum) if avg_spectrum is not None else None
    rolloff = calculate_spectrum_rolloff(avg_spectrum) if avg_spectrum is not None else None
    
    result = {
        'source_identification': source_result,
        'spectrum': {
            'frequencies': FREQUENCY_BANDS,
            'values': [round(float(v), 2) if not np.isnan(v) else None for v in avg_spectrum] if avg_spectrum is not None else [None] * len(FREQUENCY_BANDS),
            'centroid': round(centroid, 1) if centroid else None,
            'rolloff_85': rolloff
        },
        'temporal_features': {
            'avg_leq': round(avg_leq, 2),
            'avg_lmax': round(avg_lmax, 2) if avg_lmax else None,
            'avg_l10': round(avg_l10, 2) if avg_l10 else None,
            'avg_l50': round(avg_l50, 2) if avg_l50 else None,
            'avg_l90': round(avg_l90, 2) if avg_l90 else None
        }
    }
    
    return result


def generate_source_recommendations(source_type: str, zone_type: Optional[int] = None,
                                    exceedance: float = 0.0) -> List[str]:
    recommendations = []
    
    if exceedance > 0:
        if exceedance > 10:
            recommendations.append(f"当前噪声超标{exceedance:.1f}dB，属于严重超标，建议立即采取综合治理措施。")
        elif exceedance > 5:
            recommendations.append(f"当前噪声超标{exceedance:.1f}dB，超标较明显，建议采取针对性降噪措施。")
        else:
            recommendations.append(f"当前噪声超标{exceedance:.1f}dB，超标幅度较小，可采取适度的控制措施。")
    
    source_recs = {
        'traffic': [
            "优化道路规划，合理分流重型车辆",
            "在道路两侧设置声屏障或绿化带",
            "采用低噪声路面材料（如多孔沥青路面）",
            "实施交通管制，限制夜间重型车辆通行",
            "对沿线建筑物加装隔声窗"
        ],
        'construction': [
            "合理安排施工时间，避免夜间（22:00-6:00）施工",
            "选用低噪声施工设备并加装消声装置",
            "在施工现场设置移动式声屏障",
            "对混凝土搅拌、材料加工等高噪声工序采用封闭作业",
            "加强施工管理，文明施工，减少人为噪声"
        ],
        'industrial': [
            "对高噪声设备加装隔声罩或消声器",
            "优化生产工艺，采用低噪声生产技术",
            "进行厂区合理布局，将高噪声车间远离厂界",
            "对车间进行吸声降噪处理",
            "建立设备定期维护制度，减少设备老化噪声"
        ],
        'life': [
            "加强社区噪声管理宣传，提高居民环保意识",
            "对商业活动场所进行隔声改造",
            "规范餐饮、娱乐场所的营业时间",
            "加强住宅区装修施工管理",
            "设置社区噪声监测点和公示制度"
        ],
        'unknown': [
            "建议增加监测频次和时长，积累更多数据",
            "可进行现场踏勘，确定实际噪声源",
            "考虑在不同时段和不同天气条件下监测",
            "可采用便携式设备进行加密监测"
        ]
    }
    
    recommendations.extend(source_recs.get(source_type, source_recs['unknown']))
    
    if zone_type is not None and zone_type in [0, 1]:
        recommendations.append("该区域属于声环境敏感区域，建议执行更严格的噪声控制标准。")
    
    return recommendations
