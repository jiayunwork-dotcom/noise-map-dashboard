import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import ast
import json
import math

from data_models import FREQUENCY_BANDS
from spectrum_analysis import parse_spectrum_string, spectrum_to_array
from time_analysis import detect_noise_events

SOUND_SPEED = 340.0


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def compute_station_distance_matrix(stations_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    station_ids = stations_df['station_id'].tolist()
    n = len(station_ids)
    dist_matrix = np.zeros((n, n))
    delay_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            lat1, lon1 = stations_df.iloc[i]['latitude'], stations_df.iloc[i]['longitude']
            lat2, lon2 = stations_df.iloc[j]['latitude'], stations_df.iloc[j]['longitude']
            d = haversine_distance(lat1, lon1, lat2, lon2)
            dist_matrix[i, j] = dist_matrix[j, i] = d
            delay_matrix[i, j] = delay_matrix[j, i] = d / SOUND_SPEED

    return dist_matrix, delay_matrix, station_ids


def get_event_spectrum_vector(event: Dict, measurements_df: pd.DataFrame) -> Optional[np.ndarray]:
    mask = ((measurements_df['measurement_time'] >= event['start_time']) &
            (measurements_df['measurement_time'] <= event['end_time']))
    spec_rows = measurements_df.loc[mask, 'spectrum'].dropna().tolist()

    vectors = []
    for s in spec_rows[:10]:
        spec_dict = parse_spectrum_string(s)
        if spec_dict:
            arr = spectrum_to_array(spec_dict)
            if arr is not None and not np.all(np.isnan(arr)):
                vectors.append(arr)

    if not vectors:
        return None

    stacked = np.vstack(vectors)
    avg = np.nanmean(stacked, axis=0)
    valid_mask = ~np.isnan(avg)
    if np.sum(valid_mask) < 5:
        return None
    return avg


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    if v1 is None or v2 is None:
        return 0.0
    mask = ~(np.isnan(v1) | np.isnan(v2))
    if np.sum(mask) < 3:
        return 0.0
    a = v1[mask]
    b = v2[mask]
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def detect_events_for_stations(station_ids: List[str],
                                threshold_db: float = 10.0,
                                window_hours: int = 3) -> Dict[str, List[Dict]]:
    from data_models import get_station_measurements

    all_events = {}
    for sid in station_ids:
        m_df = get_station_measurements(sid)
        if m_df.empty:
            all_events[sid] = []
            continue
        events = detect_noise_events(m_df, threshold_db=threshold_db, window_hours=window_hours)
        for e in events:
            e['station_id'] = sid
            e['_spectrum_vec'] = get_event_spectrum_vector(e, m_df)
        all_events[sid] = events
    return all_events


def match_cooperative_events(all_events: Dict[str, List[Dict]],
                              delay_matrix: np.ndarray,
                              station_ids: List[str],
                              spectrum_threshold: float = 0.7,
                              time_tolerance: float = 2.0) -> List[Dict]:
    sid_to_idx = {sid: i for i, sid in enumerate(station_ids)}
    all_event_list = []
    for sid, events in all_events.items():
        for e in events:
            all_event_list.append(e)

    all_event_list.sort(key=lambda x: x['start_time'])

    parent = list(range(len(all_event_list)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(all_event_list)):
        for j in range(i + 1, len(all_event_list)):
            e1 = all_event_list[i]
            e2 = all_event_list[j]
            s1 = e1['station_id']
            s2 = e2['station_id']
            if s1 == s2:
                continue

            t1 = e1['start_time']
            t2 = e2['start_time']
            dt = abs((t2 - t1).total_seconds())

            idx1 = sid_to_idx.get(s1)
            idx2 = sid_to_idx.get(s2)
            if idx1 is None or idx2 is None:
                continue
            expected_delay = delay_matrix[idx1, idx2]
            if dt > expected_delay + time_tolerance:
                if (t2 - t1).total_seconds() > expected_delay + time_tolerance + 10:
                    break
                continue

            sim = cosine_similarity(e1.get('_spectrum_vec'), e2.get('_spectrum_vec'))
            if sim >= spectrum_threshold:
                union(i, j)

    groups = {}
    for i in range(len(all_event_list)):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(all_event_list[i])

    cooperative_groups = []
    for gid, events in enumerate(sorted(groups.values(), key=lambda g: g[0]['start_time'])):
        participating_stations = list(set(e['station_id'] for e in events))
        if len(participating_stations) < 2:
            continue

        sorted_events = sorted(events, key=lambda e: e['start_time'])
        earliest_station = sorted_events[0]['station_id']
        earliest_time = sorted_events[0]['start_time']

        sims = []
        for a in range(len(events)):
            for b in range(a + 1, len(events)):
                s = cosine_similarity(events[a].get('_spectrum_vec'), events[b].get('_spectrum_vec'))
                if s > 0:
                    sims.append(s)
        avg_similarity = float(np.mean(sims)) if sims else 0.0

        time_diffs = {}
        for e in events:
            td = (e['start_time'] - earliest_time).total_seconds()
            if e['station_id'] not in time_diffs or td < time_diffs[e['station_id']]:
                time_diffs[e['station_id']] = td

        cooperative_groups.append({
            'group_id': f'CG{gid + 1:03d}',
            'events': events,
            'participating_stations': participating_stations,
            'earliest_station': earliest_station,
            'earliest_time': earliest_time,
            'latest_time': max(e['end_time'] for e in events),
            'avg_peak_leq': float(np.mean([e['peak_leq'] for e in events])),
            'time_diffs': time_diffs,
            'avg_spectrum_similarity': avg_similarity
        })

    return cooperative_groups


def _solve_tdoa_2station(station1: Tuple[float, float],
                          station2: Tuple[float, float],
                          delta_d: float,
                          num_points: int = 200,
                          extend_factor: float = 5.0) -> List[Tuple[float, float]]:
    lat1, lon1 = station1
    lat2, lon2 = station2
    d = haversine_distance(lat1, lon1, lat2, lon2)

    if d < 1.0:
        return []
    if abs(delta_d) >= d - 1e-3:
        delta_d = np.sign(delta_d) * (d - 1.0)

    mid_lat = (lat1 + lat2) / 2
    mid_lon = (lon1 + lon2) / 2
    scale_lat = 1.0 / 111000.0
    avg_lat_rad = math.radians(mid_lat)
    scale_lon = 1.0 / (111000.0 * math.cos(avg_lat_rad))

    x1 = (lon1 - mid_lon) / scale_lon
    y1 = (lat1 - mid_lat) / scale_lat
    x2 = (lon2 - mid_lon) / scale_lon
    y2 = (lat2 - mid_lat) / scale_lat

    a = delta_d / 2.0
    c = d / 2.0

    if abs(a) < 1e-6:
        a = 1e-6

    b_sq = c * c - a * a
    if b_sq <= 0:
        b_sq = max(b_sq, 1.0)
    b = math.sqrt(b_sq)

    dx = x2 - x1
    dy = y2 - y1
    angle = math.atan2(dy, dx)

    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    max_t = extend_factor * max(c, 200.0)
    t_values = np.linspace(-max_t, max_t, num_points)

    points = []
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    for t in t_values:
        if abs(t) >= abs(a):
            y_h = math.sqrt((t * t) / (a * a) - 1) * b if a != 0 else 0
            for sign in [1, -1]:
                x_local = t
                y_local = sign * y_h
                xr = cx + x_local * cos_a - y_local * sin_a
                yr = cy + x_local * sin_a + y_local * cos_a
                lon_r = mid_lon + xr * scale_lon
                lat_r = mid_lat + yr * scale_lat
                points.append((lat_r, lon_r))

    return points


def _grid_search_localization(stations: List[Tuple[float, float, str]],
                               measured_deltas: Dict[Tuple[str, str], float],
                               bounds: Tuple[float, float, float, float],
                               grid_step: float = 20.0) -> Dict:
    min_lon, min_lat, max_lon, max_lat = bounds
    mid_lat = (min_lat + max_lat) / 2
    mid_lon = (min_lon + max_lon) / 2

    scale_lat = 1.0 / 111000.0
    avg_lat_rad = math.radians(mid_lat)
    scale_lon = 1.0 / (111000.0 * math.cos(avg_lat_rad))

    width_m = (max_lon - min_lon) / scale_lon
    height_m = (max_lat - min_lat) / scale_lat

    nx = max(20, int(width_m / grid_step))
    ny = max(20, int(height_m / grid_step))

    xs = np.linspace(min_lon, max_lon, nx)
    ys = np.linspace(min_lat, max_lat, ny)

    best_score = float('inf')
    best_lat = mid_lat
    best_lon = mid_lon
    scores = np.full((ny, nx), np.nan)

    station_coords = {}
    for lat, lon, sid in stations:
        station_coords[sid] = (lat, lon)

    for j, lat in enumerate(ys):
        for i, lon in enumerate(xs):
            total_err = 0.0
            count = 0
            for (s1, s2), measured_delta in measured_deltas.items():
                if s1 not in station_coords or s2 not in station_coords:
                    continue
                lat1, lon1 = station_coords[s1]
                lat2, lon2 = station_coords[s2]
                d1 = haversine_distance(lat, lon, lat1, lon1)
                d2 = haversine_distance(lat, lon, lat2, lon2)
                predicted_delta = d2 - d1
                err = (predicted_delta - measured_delta) ** 2
                total_err += err
                count += 1
            if count > 0:
                rmse = math.sqrt(total_err / count)
                scores[j, i] = rmse
                if rmse < best_score:
                    best_score = rmse
                    best_lat = lat
                    best_lon = lon

    if not np.any(~np.isnan(scores)):
        return {'latitude': mid_lat, 'longitude': mid_lon, 'uncertainty_m': max(width_m, height_m), 'rmse': float('inf')}

    valid_scores = scores[~np.isnan(scores)]
    if len(valid_scores) > 0:
        threshold = np.percentile(valid_scores, min(90, 20 + len(valid_scores)))
        within_mask = scores <= threshold
        within_indices = np.argwhere(within_mask)
        if len(within_indices) > 0:
            lats_within = ys[within_indices[:, 0]]
            lons_within = xs[within_indices[:, 1]]
            center_lat = float(np.mean(lats_within))
            center_lon = float(np.mean(lons_within))
            dists = []
            for lt, ln in zip(lats_within, lons_within):
                dists.append(haversine_distance(center_lat, center_lon, lt, ln))
            uncertainty = float(np.percentile(dists, 95)) if dists else max(width_m, height_m) / 2
            return {
                'latitude': best_lat,
                'longitude': best_lon,
                'center_latitude': center_lat,
                'center_longitude': center_lon,
                'uncertainty_m': max(uncertainty, grid_step * 2),
                'rmse': best_score,
                'score_grid': scores,
                'grid_lats': ys,
                'grid_lons': xs
            }

    return {'latitude': best_lat, 'longitude': best_lon, 'uncertainty_m': max(width_m, height_m), 'rmse': best_score}


def estimate_source_location(group: Dict, stations_df: pd.DataFrame) -> Dict:
    participating = group['participating_stations']
    if len(participating) < 2:
        return {'located': False, 'reason': '参与站点不足'}

    stations_info = []
    for sid in participating:
        row = stations_df[stations_df['station_id'] == sid]
        if len(row) > 0:
            stations_info.append((float(row.iloc[0]['latitude']),
                                   float(row.iloc[0]['longitude']),
                                   sid))

    if len(stations_info) < 2:
        return {'located': False, 'reason': '站点坐标缺失'}

    earliest_sid = group['earliest_station']
    time_diffs = group['time_diffs']

    measured_deltas = {}
    hyperbolas = []
    for sid, td in time_diffs.items():
        if sid == earliest_sid:
            continue
        delta_d = SOUND_SPEED * td
        measured_deltas[(earliest_sid, sid)] = delta_d

        s1_info = None
        s2_info = None
        for info in stations_info:
            if info[2] == earliest_sid:
                s1_info = info
            if info[2] == sid:
                s2_info = info
        if s1_info and s2_info:
            hyp_points = _solve_tdoa_2station(
                (s1_info[0], s1_info[1]),
                (s2_info[0], s2_info[1]),
                delta_d
            )
            hyperbolas.append({
                'station_pair': (earliest_sid, sid),
                'delta_distance_m': delta_d,
                'points': hyp_points
            })

    lats = [s[0] for s in stations_info]
    lons = [s[1] for s in stations_info]
    center_lat = float(np.mean(lats))
    center_lon = float(np.mean(lons))

    max_dist = 0
    for s in stations_info:
        d = haversine_distance(center_lat, center_lon, s[0], s[1])
        max_dist = max(max_dist, d)

    extend_m = max(max_dist * 3, 1000)
    scale_lat = 1.0 / 111000.0
    avg_lat_rad = math.radians(center_lat)
    scale_lon = 1.0 / (111000.0 * math.cos(avg_lat_rad))

    bounds = (
        center_lon - extend_m * scale_lon,
        center_lat - extend_m * scale_lat,
        center_lon + extend_m * scale_lon,
        center_lat + extend_m * scale_lat
    )

    if len(stations_info) >= 3:
        grid_step = max(20, int(extend_m / 50))
        loc_result = _grid_search_localization(stations_info, measured_deltas, bounds, grid_step=grid_step)
        loc_result['located'] = True
        loc_result['hyperbolas'] = hyperbolas
    else:
        loc_result = {
            'located': True,
            'latitude': None,
            'longitude': None,
            'center_latitude': None,
            'center_longitude': None,
            'uncertainty_m': None,
            'rmse': None,
            'hyperbolas': hyperbolas,
            'reason': '仅2站，仅可绘制双曲线，无法精确定位'
        }

    first_sid = None
    ref_lat, ref_lon = 0, 0
    for s in stations_info:
        if s[2] == earliest_sid:
            ref_lat, ref_lon = s[0], s[1]
            first_sid = s[2]
            break

    if loc_result.get('latitude') and loc_result.get('longitude'):
        dlat = loc_result['latitude'] - ref_lat
        dlon = loc_result['longitude'] - ref_lon
        bearing = math.degrees(math.atan2(dlon * math.cos(math.radians(ref_lat)), dlat))
        if bearing < 0:
            bearing += 360
        loc_result['bearing_deg'] = round(bearing, 1)

        dist = haversine_distance(ref_lat, ref_lon, loc_result['latitude'], loc_result['longitude'])
        loc_result['distance_from_earliest_m'] = round(dist, 1)

    return loc_result


def export_traceability_geojson(cooperative_groups: List[Dict],
                                 stations_df: pd.DataFrame,
                                 location_results: Dict[str, Dict]) -> Dict:
    features = []

    for group in cooperative_groups:
        gid = group['group_id']
        loc = location_results.get(gid, {})

        if loc.get('latitude') and loc.get('longitude'):
            point_feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [loc['longitude'], loc['latitude']]
                },
                'properties': {
                    'type': 'estimated_source',
                    'group_id': gid,
                    'earliest_time': group['earliest_time'].isoformat() if isinstance(group['earliest_time'], datetime) else str(group['earliest_time']),
                    'participating_stations': group['participating_stations'],
                    'station_count': len(group['participating_stations']),
                    'avg_peak_leq': round(group['avg_peak_leq'], 1),
                    'avg_spectrum_similarity': round(group['avg_spectrum_similarity'], 3),
                    'bearing_deg': loc.get('bearing_deg'),
                    'distance_from_earliest_m': loc.get('distance_from_earliest_m'),
                    'localization_rmse': round(loc.get('rmse', 0), 1),
                    'uncertainty_m': round(loc.get('uncertainty_m', 0), 1)
                }
            }
            features.append(point_feature)

            uncertainty = loc.get('uncertainty_m', 100)
            if uncertainty and uncertainty > 0:
                num_points = 64
                circle_coords = []
                clat = loc['latitude']
                clon = loc['longitude']
                for i in range(num_points + 1):
                    angle = 2 * math.pi * i / num_points
                    dlat = (uncertainty / 111000.0) * math.cos(angle)
                    dlon = (uncertainty / (111000.0 * math.cos(math.radians(clat)))) * math.sin(angle)
                    circle_coords.append([clon + dlon, clat + dlat])
                circle_feature = {
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Polygon',
                        'coordinates': [circle_coords]
                    },
                    'properties': {
                        'type': 'uncertainty_circle',
                        'group_id': gid,
                        'uncertainty_m': round(uncertainty, 1),
                        'description': f'{gid} 声源定位不确定度范围'
                    }
                }
                features.append(circle_feature)

        for hyp in loc.get('hyperbolas', []):
            if hyp.get('points'):
                line_coords = [[lon, lat] for lat, lon in hyp['points']]
                hyp_feature = {
                    'type': 'Feature',
                    'geometry': {
                        'type': 'LineString',
                        'coordinates': line_coords
                    },
                    'properties': {
                        'type': 'tdoa_hyperbola',
                        'group_id': gid,
                        'station_pair': list(hyp['station_pair']),
                        'delta_distance_m': round(hyp['delta_distance_m'], 1)
                    }
                }
                features.append(hyp_feature)

    for _, row in stations_df.iterrows():
        sid = row['station_id']
        station_feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [float(row['longitude']), float(row['latitude'])]
            },
            'properties': {
                'type': 'monitoring_station',
                'station_id': sid,
                'station_name': row.get('station_name', sid),
                'region': row.get('region', '')
            }
        }
        features.append(station_feature)

    return {
        'type': 'FeatureCollection',
        'features': features,
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'group_count': len(cooperative_groups),
            'station_count': len(stations_df)
        }
    }
