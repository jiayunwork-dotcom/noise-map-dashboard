import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
import json
import random

from data_models import FREQUENCY_BANDS, add_station, add_measurements, init_database

init_database()

CENTERS = {
    '东城区': (116.42, 39.93),
    '西城区': (116.37, 39.92),
    '朝阳区': (116.45, 39.93),
    '海淀区': (116.31, 39.96),
    '丰台区': (116.29, 39.85)
}

STATION_CONFIG = {
    '东城区': [
        ('ST001', '东直门站', 'traffic', 0.02, 0.01),
        ('ST002', '王府井站', 'life', 0.01, 0.02),
        ('ST003', '天坛东门站', 'life', -0.01, -0.02),
    ],
    '西城区': [
        ('ST004', '西单站', 'traffic', 0.01, 0.01),
        ('ST005', '金融街站', 'industrial', -0.02, 0.005),
        ('ST006', '西直门站', 'traffic', 0.015, 0.02),
    ],
    '朝阳区': [
        ('ST007', '国贸站', 'traffic', 0.02, 0.01),
        ('ST008', '三里屯站', 'life', 0.005, -0.015),
        ('ST009', 'CBD站', 'industrial', -0.01, 0.01),
        ('ST010', '望京站', 'traffic', 0.01, -0.02),
    ],
    '海淀区': [
        ('ST011', '中关村站', 'life', 0.02, 0.005),
        ('ST012', '学院路站', 'life', 0.005, -0.01),
        ('ST013', '上地站', 'industrial', -0.02, 0.015),
    ],
    '丰台区': [
        ('ST014', '北京西站', 'traffic', 0.02, 0.005),
        ('ST015', '丽泽商务区站', 'construction', 0.01, -0.015),
    ]
}

BASE_NOISE_LEVELS = {
    'traffic': (68, 8),
    'life': (55, 7),
    'industrial': (62, 5),
    'construction': (72, 10),
    'unknown': (58, 6)
}


def generate_spectrum(source_type: str, base_level: float) -> Dict[int, float]:
    spectrum = {}
    
    base_patterns = {
        'traffic': {
            63: 0.95, 80: 0.98, 100: 1.0, 125: 1.0, 160: 0.98,
            200: 0.96, 250: 0.93, 315: 0.90, 400: 0.87, 500: 0.84,
            630: 0.80, 800: 0.76, 1000: 0.72, 1250: 0.68
        },
        'construction': {
            125: 0.88, 160: 0.92, 200: 0.96, 250: 1.0, 315: 1.0,
            400: 0.98, 500: 0.95, 630: 0.92, 800: 0.88, 1000: 0.84
        },
        'industrial': {
            63: 0.85, 80: 0.90, 100: 0.94, 125: 0.97, 160: 0.99,
            200: 1.0, 250: 0.98, 315: 0.95, 400: 0.92, 500: 0.90,
            630: 0.88, 800: 0.85, 1000: 0.82, 1250: 0.80
        },
        'life': {
            250: 0.85, 315: 0.90, 400: 0.94, 500: 0.97, 630: 1.0,
            800: 1.0, 1000: 0.98, 1250: 0.96, 1600: 0.94, 2000: 0.92,
            2500: 0.90, 3150: 0.88, 4000: 0.85
        }
    }
    
    pattern = base_patterns.get(source_type, base_patterns['life'])
    
    for freq in FREQUENCY_BANDS:
        weight = 0.5
        for pf, pw in pattern.items():
            if abs(freq - pf) / pf < 0.3:
                weight = max(weight, pw)
        
        random_variation = np.random.uniform(-2, 2)
        spectrum[freq] = round(base_level * weight + random_variation - np.random.uniform(5, 15), 1)
    
    return spectrum


def generate_noise_value(source_type: str, hour: int, weekday: int) -> Tuple[float, float, float, float, float]:
    base_mean, base_std = BASE_NOISE_LEVELS.get(source_type, BASE_NOISE_LEVELS['unknown'])
    
    hour_factor = 0.0
    if 7 <= hour <= 9:
        hour_factor = 5.0 if source_type == 'traffic' else 3.0
    elif 12 <= hour <= 14:
        hour_factor = 2.0
    elif 18 <= hour <= 21:
        hour_factor = 4.0 if source_type in ['traffic', 'life'] else 2.5
    elif 0 <= hour <= 5:
        hour_factor = -8.0 if source_type == 'traffic' else -5.0
    
    weekday_factor = 0.0
    if weekday >= 5:
        if source_type == 'traffic':
            weekday_factor = -3.0
        elif source_type == 'life':
            weekday_factor = 2.0
    
    leq_mean = base_mean + hour_factor + weekday_factor
    leq_std = base_std
    
    leq = float(np.clip(np.random.normal(leq_mean, leq_std * 0.5), 20, 130))
    
    l10 = leq + np.random.uniform(3, 8 if source_type == 'traffic' else 5)
    l50 = leq + np.random.uniform(-1, 2)
    l90 = leq - np.random.uniform(3, 8 if source_type == 'traffic' else 5)
    lmax = leq + np.random.uniform(8, 20 if source_type == 'construction' else 15)
    
    return (
        round(leq, 1),
        round(lmax, 1),
        round(l10, 1),
        round(l50, 1),
        round(l90, 1)
    )


def generate_station_data(station_id: str, station_name: str, region: str,
                          source_type: str, lon_offset: float, lat_offset: float,
                          start_date: datetime, days: int = 60) -> pd.DataFrame:
    center_lon, center_lat = CENTERS[region]
    longitude = round(center_lon + lon_offset + np.random.uniform(-0.005, 0.005), 6)
    latitude = round(center_lat + lat_offset + np.random.uniform(-0.005, 0.005), 6)
    
    add_station(
        station_id=station_id,
        longitude=longitude,
        latitude=latitude,
        station_name=station_name,
        region=region,
        zone_type=2 if source_type == 'industrial' else 1
    )
    
    records = []
    
    for day in range(days):
        current_date = start_date + timedelta(days=day)
        weekday = current_date.weekday()
        
        for hour in range(24):
            if np.random.random() < 0.08:
                continue
            
            measurement_time = current_date.replace(hour=hour, minute=0, second=0)
            
            leq, lmax, l10, l50, l90 = generate_noise_value(source_type, hour, weekday)
            spectrum = generate_spectrum(source_type, leq)
            
            record = {
                'station_id': station_id,
                'station_name': station_name,
                'region': region,
                'longitude': longitude,
                'latitude': latitude,
                'measurement_time': measurement_time,
                'leq': leq,
                'lmax': lmax,
                'l10': l10,
                'l50': l50,
                'l90': l90,
            }
            
            for freq, val in spectrum.items():
                record[f'freq_{freq}'] = val
            
            records.append(record)
    
    df = pd.DataFrame(records)
    return df


def generate_sample_zones_geojson() -> Dict:
    features = []
    
    zone_defs = [
        ('居民区A', 1, [
            (116.35, 39.88), (116.38, 39.88),
            (116.38, 39.91), (116.35, 39.91)
        ]),
        ('商业区B', 2, [
            (116.38, 39.88), (116.42, 39.88),
            (116.42, 39.91), (116.38, 39.91)
        ]),
        ('工业区C', 3, [
            (116.42, 39.88), (116.46, 39.88),
            (116.46, 39.91), (116.42, 39.91)
        ]),
        ('文教区D', 1, [
            (116.28, 39.94), (116.33, 39.94),
            (116.33, 39.99), (116.28, 39.99)
        ]),
        ('交通干线E', 4, [
            (116.33, 39.915), (116.48, 39.915),
            (116.48, 39.935), (116.33, 39.935)
        ]),
    ]
    
    for zname, ztype, coords in zone_defs:
        coords_closed = coords + [coords[0]]
        feature = {
            'type': 'Feature',
            'properties': {
                'name': zname,
                'zone_name': zname,
                'zone_type': ztype,
                'type': ztype
            },
            'geometry': {
                'type': 'Polygon',
                'coordinates': [[list(pt) for pt in coords_closed]]
            }
        }
        features.append(feature)
    
    return {
        'type': 'FeatureCollection',
        'features': features
    }


def main():
    print("=" * 60)
    print("生成噪声监测示例数据...")
    print("=" * 60)
    
    start_date = datetime(2024, 1, 1)
    days = 75
    
    all_records = []
    
    for region, stations in STATION_CONFIG.items():
        for station_id, station_name, source_type, lon_off, lat_off in stations:
            print(f"  生成 {region}/{station_name} ({station_id}) - {source_type}...")
            df = generate_station_data(
                station_id, station_name, region,
                source_type, lon_off, lat_off,
                start_date, days
            )
            all_records.append(df)
            
            success, errors, error_list = add_measurements(df)
            print(f"    完成: {success}条记录, {errors}条错误")
    
    full_df = pd.concat(all_records, ignore_index=True)
    
    csv_path = 'sample_noise_data.csv'
    cols = ['station_id', 'station_name', 'region', 'longitude', 'latitude',
            'measurement_time', 'leq', 'lmax', 'l10', 'l50', 'l90'] + \
           [f'freq_{f}' for f in FREQUENCY_BANDS]
    cols = [c for c in cols if c in full_df.columns]
    
    full_df[cols].to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\nCSV数据已保存至: {csv_path} ({len(full_df)}条记录)")
    
    zones_geojson = generate_sample_zones_geojson()
    zones_path = 'sample_functional_zones.geojson'
    with open(zones_path, 'w', encoding='utf-8') as f:
        json.dump(zones_geojson, f, ensure_ascii=False, indent=2)
    print(f"功能区GeoJSON已保存至: {zones_path} ({len(zones_geojson['features'])}个区域)")
    
    total_stations = sum(len(s) for s in STATION_CONFIG.values())
    print("\n" + "=" * 60)
    print("数据生成完成!")
    print(f"  区域数: {len(STATION_CONFIG)}")
    print(f"  站点数: {total_stations}")
    print(f"  监测记录: {len(full_df)} 条")
    print(f"  监测天数: {days} 天")
    print("=" * 60)
    print("\n使用方法:")
    print("  1. 运行 streamlit run app.py 启动应用")
    print("  2. 在数据导入页面上传 sample_noise_data.csv")
    print("  3. 在功能区评价页面上传 sample_functional_zones.geojson")


if __name__ == '__main__':
    main()
