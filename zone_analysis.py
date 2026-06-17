import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from shapely.geometry import Point, Polygon, shape, MultiPolygon
from shapely.ops import unary_union
from data_models import ZONE_STANDARDS

ZONE_COLORS = {
    0: {'fill': '#E0F7FA', 'stroke': '#0097A7', 'fillOpacity': 0.4},
    1: {'fill': '#E8F5E9', 'stroke': '#388E3C', 'fillOpacity': 0.4},
    2: {'fill': '#FFF8E1', 'stroke': '#F9A825', 'fillOpacity': 0.4},
    3: {'fill': '#FCE4EC', 'stroke': '#C2185B', 'fillOpacity': 0.4},
    4: {'fill': '#EFEBE9', 'stroke': '#5D4037', 'fillOpacity': 0.4}
}


def parse_geojson(geojson_str: str) -> List[Dict]:
    try:
        data = json.loads(geojson_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"GeoJSON解析失败: {e}")
    
    features = []
    
    if data.get('type') == 'FeatureCollection':
        for feature in data.get('features', []):
            parsed = _parse_single_feature(feature)
            if parsed:
                features.append(parsed)
    elif data.get('type') == 'Feature':
        parsed = _parse_single_feature(data)
        if parsed:
            features.append(parsed)
    else:
        raise ValueError("GeoJSON格式不正确，应为FeatureCollection或Feature类型")
    
    return features


def _parse_single_feature(feature: Dict) -> Optional[Dict]:
    try:
        geometry = feature.get('geometry')
        if not geometry:
            return None
        
        properties = feature.get('properties', {})
        
        zone_type = None
        for key in ['zone_type', 'type', 'class', 'category', '类别', '类型', '功能区类别']:
            if key in properties:
                val = properties[key]
                try:
                    zone_type = int(val)
                    break
                except (ValueError, TypeError):
                    continue
        
        if zone_type is None:
            return None
        
        if zone_type not in ZONE_STANDARDS:
            return None
        
        zone_name = None
        for key in ['name', 'zone_name', '名称', '区域名称']:
            if key in properties and properties[key]:
                zone_name = str(properties[key])
                break
        
        if not zone_name:
            zone_name = f"{ZONE_STANDARDS[zone_type]['name']}_{len(zone_name or '')}"
        
        geom_shape = shape(geometry)
        
        return {
            'zone_name': zone_name,
            'zone_type': zone_type,
            'geometry': geometry,
            'shape': geom_shape,
            'properties': properties
        }
    except Exception:
        return None


def process_uploaded_zones(geojson_bytes: bytes) -> Tuple[List[Dict], List[str]]:
    warnings = []
    try:
        geojson_str = geojson_bytes.decode('utf-8')
    except UnicodeDecodeError:
        try:
            geojson_str = geojson_bytes.decode('gbk')
        except UnicodeDecodeError:
            raise ValueError("GeoJSON文件编码无法识别，请使用UTF-8或GBK编码")
    
    features = parse_geojson(geojson_str)
    
    valid_types = set(ZONE_STANDARDS.keys())
    found_types = set(f['zone_type'] for f in features)
    invalid_types = found_types - valid_types
    
    if invalid_types:
        warnings.append(f"发现未知功能区类型: {invalid_types}，已跳过")
    
    if len(features) == 0:
        raise ValueError("未找到有效的功能区数据，请确保GeoJSON中包含有效的zone_type属性(0-4)")
    
    return features, warnings


def check_grid_in_zone(grid_lon: np.ndarray, grid_lat: np.ndarray,
                       zone_shape) -> np.ndarray:
    ny, nx = len(grid_lat), len(grid_lon)
    mask = np.zeros((ny, nx), dtype=bool)
    
    if isinstance(zone_shape, (Polygon, MultiPolygon)):
        bounds = zone_shape.bounds
        min_lon, min_lat, max_lon, max_lat = bounds
        
        lon_in_bounds = (grid_lon >= min_lon) & (grid_lon <= max_lon)
        lat_in_bounds = (grid_lat >= min_lat) & (grid_lat <= max_lat)
        
        if not np.any(lon_in_bounds) or not np.any(lat_in_bounds):
            return mask
        
        for i, lat in enumerate(grid_lat):
            if not lat_in_bounds[i]:
                continue
            for j, lon in enumerate(grid_lon):
                if not lon_in_bounds[j]:
                    continue
                if zone_shape.contains(Point(lon, lat)):
                    mask[i, j] = True
    
    return mask


def evaluate_zone_compliance(zones: List[Dict], z_grid: np.ndarray,
                             grid_lon: np.ndarray, grid_lat: np.ndarray,
                             time_period: str = 'day',
                             cell_resolution_m: float = 50.0) -> List[Dict]:
    results = []
    cell_area = cell_resolution_m ** 2
    
    for zone in zones:
        zone_type = zone['zone_type']
        zone_name = zone['zone_name']
        zone_shape = zone['shape']
        
        if time_period == 'day':
            standard = ZONE_STANDARDS[zone_type]['day']
        else:
            standard = ZONE_STANDARDS[zone_type]['night']
        
        mask = check_grid_in_zone(grid_lon, grid_lat, zone_shape)
        
        values_in_zone = z_grid[mask]
        valid_values = values_in_zone[~np.isnan(values_in_zone)]
        
        total_cells = len(valid_values)
        if total_cells == 0:
            results.append({
                'zone_name': zone_name,
                'zone_type': zone_type,
                'zone_type_name': ZONE_STANDARDS[zone_type]['name'],
                'standard': standard,
                'time_period': time_period,
                'total_cells': 0,
                'compliant_cells': 0,
                'non_compliant_cells': 0,
                'compliance_rate': 0.0,
                'non_compliance_rate': 0.0,
                'avg_value': 0.0,
                'max_value': 0.0,
                'min_value': 0.0,
                'std_value': 0.0,
                'area_km2': 0.0,
                'non_compliant_area_km2': 0.0,
                'avg_exceedance': 0.0,
                'max_exceedance': 0.0,
                'grid_mask': mask,
                'non_compliant_mask': np.zeros_like(mask, dtype=bool)
            })
            continue
        
        non_compliant_mask = (valid_values > standard)
        compliant_mask = ~non_compliant_mask
        
        compliant_cells = int(np.sum(compliant_mask))
        non_compliant_cells = int(np.sum(non_compliant_mask))
        
        exceedance_values = valid_values[non_compliant_mask] - standard
        
        full_non_compliant_mask = np.zeros_like(mask, dtype=bool)
        temp_idx = 0
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j] and not np.isnan(z_grid[i, j]):
                    if non_compliant_cells > 0 and temp_idx < len(non_compliant_mask) and non_compliant_mask[temp_idx]:
                        full_non_compliant_mask[i, j] = True
                    temp_idx += 1
        
        results.append({
            'zone_name': zone_name,
            'zone_type': zone_type,
            'zone_type_name': ZONE_STANDARDS[zone_type]['name'],
            'standard': standard,
            'time_period': time_period,
            'total_cells': total_cells,
            'compliant_cells': compliant_cells,
            'non_compliant_cells': non_compliant_cells,
            'compliance_rate': compliant_cells / total_cells * 100 if total_cells > 0 else 0,
            'non_compliance_rate': non_compliant_cells / total_cells * 100 if total_cells > 0 else 0,
            'avg_value': float(np.mean(valid_values)),
            'max_value': float(np.max(valid_values)),
            'min_value': float(np.min(valid_values)),
            'std_value': float(np.std(valid_values)),
            'area_km2': total_cells * cell_area / 1e6,
            'non_compliant_area_km2': non_compliant_cells * cell_area / 1e6,
            'avg_exceedance': float(np.mean(exceedance_values)) if len(exceedance_values) > 0 else 0.0,
            'max_exceedance': float(np.max(exceedance_values)) if len(exceedance_values) > 0 else 0.0,
            'grid_mask': mask,
            'non_compliant_mask': full_non_compliant_mask
        })
    
    return results


def zones_to_geojson_features(zones: List[Dict], evaluation_results: Optional[List[Dict]] = None,
                              show_non_compliant: bool = False) -> List[Dict]:
    features = []
    
    for idx, zone in enumerate(zones):
        zone_type = zone['zone_type']
        color_scheme = ZONE_COLORS.get(zone_type, ZONE_COLORS[2])
        
        result = None
        if evaluation_results and idx < len(evaluation_results):
            result = evaluation_results[idx]
        
        properties = {
            'zone_name': zone['zone_name'],
            'zone_type': zone_type,
            'zone_type_name': ZONE_STANDARDS[zone_type]['name'],
            'description': ZONE_STANDARDS[zone_type]['description'],
            'day_standard': ZONE_STANDARDS[zone_type]['day'],
            'night_standard': ZONE_STANDARDS[zone_type]['night']
        }
        
        if result:
            properties.update({
                'compliance_rate': round(result['compliance_rate'], 2),
                'non_compliance_rate': round(result['non_compliance_rate'], 2),
                'avg_value': round(result['avg_value'], 2),
                'max_value': round(result['max_value'], 2),
                'standard': result['standard'],
                'time_period': result['time_period']
            })
        
        style = {
            'color': color_scheme['stroke'],
            'weight': 2,
            'opacity': 0.8,
            'fillColor': color_scheme['fill'],
            'fillOpacity': color_scheme['fillOpacity']
        }
        
        if result and show_non_compliant and result['non_compliant_cells'] > 0:
            style.update({
                'color': '#D32F2F',
                'weight': 3,
                'fillColor': '#FFCDD2',
                'fillOpacity': 0.6,
                'dashArray': '5, 5'
            })
        
        properties.update({'style': style})
        
        feature = {
            'type': 'Feature',
            'properties': properties,
            'geometry': zone['geometry']
        }
        features.append(feature)
    
    return features


def generate_compliance_summary(evaluation_results: List[Dict]) -> pd.DataFrame:
    rows = []
    for result in evaluation_results:
        rows.append({
            '功能区名称': result['zone_name'],
            '功能区类别': result['zone_type_name'],
            '评价时段': '昼间' if result['time_period'] == 'day' else '夜间',
            '标准限值(dB)': result['standard'],
            '等效声级均值(dB)': round(result['avg_value'], 2),
            '等效声级最大值(dB)': round(result['max_value'], 2),
            '评价面积(km²)': round(result['area_km2'], 4),
            '达标比例(%)': round(result['compliance_rate'], 2),
            '超标面积(km²)': round(result['non_compliant_area_km2'], 4),
            '最大超标量(dB)': round(result['max_exceedance'], 2)
        })
    
    return pd.DataFrame(rows)


def get_overall_compliance_stats(evaluation_results: List[Dict]) -> Dict:
    total_area = sum(r['area_km2'] for r in evaluation_results)
    total_non_compliant_area = sum(r['non_compliant_area_km2'] for r in evaluation_results)
    overall_compliance = ((total_area - total_non_compliant_area) / total_area * 100) if total_area > 0 else 0
    
    zone_type_stats = {}
    for r in evaluation_results:
        zt = r['zone_type']
        if zt not in zone_type_stats:
            zone_type_stats[zt] = {
                'name': ZONE_STANDARDS[zt]['name'],
                'total_area': 0,
                'non_compliant_area': 0,
                'count': 0
            }
        zone_type_stats[zt]['total_area'] += r['area_km2']
        zone_type_stats[zt]['non_compliant_area'] += r['non_compliant_area_km2']
        zone_type_stats[zt]['count'] += 1
    
    for zt in zone_type_stats:
        za = zone_type_stats[zt]
        za['compliance_rate'] = ((za['total_area'] - za['non_compliant_area']) / za['total_area'] * 100) if za['total_area'] > 0 else 0
    
    return {
        'total_area': total_area,
        'total_non_compliant_area': total_non_compliant_area,
        'overall_compliance_rate': overall_compliance,
        'zone_count': len(evaluation_results),
        'non_compliant_zone_count': sum(1 for r in evaluation_results if r['compliance_rate'] < 100),
        'zone_type_stats': zone_type_stats
    }
