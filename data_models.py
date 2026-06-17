import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import Optional, Dict, List, Tuple

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'noise_data.db')

FREQUENCY_BANDS = [
    25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500,
    630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300,
    8000, 10000, 12500
]

ZONE_STANDARDS = {
    0: {'name': '0类声环境功能区', 'day': 50, 'night': 40, 'description': '康复疗养区等特别需要安静的区域'},
    1: {'name': '1类声环境功能区', 'day': 55, 'night': 45, 'description': '居民住宅、医疗卫生、文化教育等区域'},
    2: {'name': '2类声环境功能区', 'day': 60, 'night': 50, 'description': '商业金融、集市贸易，或居住、商业、工业混杂区域'},
    3: {'name': '3类声环境功能区', 'day': 65, 'night': 55, 'description': '工业生产、仓储物流为主要功能的区域'},
    4: {'name': '4类声环境功能区', 'day': 70, 'night': 55, 'description': '交通干线两侧一定距离之内的区域'}
}


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stations (
            station_id TEXT PRIMARY KEY,
            station_name TEXT,
            region TEXT,
            longitude REAL NOT NULL,
            latitude REAL NOT NULL,
            zone_type INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS measurements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station_id TEXT NOT NULL,
            measurement_time TIMESTAMP NOT NULL,
            leq REAL NOT NULL,
            lmax REAL,
            l10 REAL,
            l50 REAL,
            l90 REAL,
            spectrum TEXT,
            is_valid INTEGER DEFAULT 1,
            invalid_reason TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (station_id) REFERENCES stations(station_id),
            UNIQUE(station_id, measurement_time)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS functional_zones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            zone_name TEXT,
            zone_type INTEGER NOT NULL,
            geojson TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS noise_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            road_start_lon REAL,
            road_start_lat REAL,
            road_end_lon REAL,
            road_end_lat REAL,
            traffic_volume REAL,
            avg_speed REAL,
            heavy_vehicle_ratio REAL,
            road_surface TEXT,
            barrier_height REAL,
            barrier_position REAL,
            prediction_lon REAL,
            prediction_lat REAL,
            distance REAL,
            predicted_leq REAL,
            inserted_loss REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()


def add_station(station_id: str, longitude: float, latitude: float,
                station_name: Optional[str] = None, region: Optional[str] = None,
                zone_type: Optional[int] = None) -> bool:
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO stations 
            (station_id, station_name, region, longitude, latitude, zone_type)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (station_id, station_name or station_id, region or '未知区域',
              longitude, latitude, zone_type))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"添加站点失败: {e}")
        return False


def add_measurements(dataframe: pd.DataFrame) -> Tuple[int, int, List[Dict]]:
    conn = get_connection()
    cursor = conn.cursor()
    
    success_count = 0
    error_count = 0
    errors = []
    
    for idx, row in dataframe.iterrows():
        try:
            station_id = str(row['station_id'])
            cursor.execute('SELECT station_id FROM stations WHERE station_id = ?', (station_id,))
            if not cursor.fetchone():
                lon = float(row.get('longitude', 0))
                lat = float(row.get('latitude', 0))
                add_station(station_id, lon, lat)
            
            measurement_time = row['measurement_time']
            if isinstance(measurement_time, str):
                measurement_time = datetime.fromisoformat(measurement_time.replace('Z', '+00:00'))
            measurement_time = measurement_time.strftime('%Y-%m-%d %H:%M:%S')
            
            spectrum_dict = {}
            for freq in FREQUENCY_BANDS:
                col_name = f'freq_{freq}'
                if col_name in row and pd.notna(row[col_name]):
                    spectrum_dict[freq] = float(row[col_name])
            spectrum_str = str(spectrum_dict) if spectrum_dict else None
            
            cursor.execute('''
                INSERT OR REPLACE INTO measurements
                (station_id, measurement_time, leq, lmax, l10, l50, l90, spectrum, is_valid, invalid_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                station_id,
                measurement_time,
                float(row['leq']),
                float(row.get('lmax')) if pd.notna(row.get('lmax')) else None,
                float(row.get('l10')) if pd.notna(row.get('l10')) else None,
                float(row.get('l50')) if pd.notna(row.get('l50')) else None,
                float(row.get('l90')) if pd.notna(row.get('l90')) else None,
                spectrum_str,
                1,
                None
            ))
            success_count += 1
        except Exception as e:
            error_count += 1
            errors.append({'row': idx, 'error': str(e), 'data': row.to_dict()})
    
    conn.commit()
    conn.close()
    return success_count, error_count, errors


def get_all_stations() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query('SELECT * FROM stations ORDER BY region, station_id', conn)
    conn.close()
    return df


def get_stations_by_region() -> Dict[str, List[Dict]]:
    stations_df = get_all_stations()
    region_dict = {}
    for _, row in stations_df.iterrows():
        region = row['region'] or '未知区域'
        if region not in region_dict:
            region_dict[region] = []
        region_dict[region].append({
            'station_id': row['station_id'],
            'station_name': row['station_name'],
            'longitude': row['longitude'],
            'latitude': row['latitude'],
            'zone_type': row['zone_type']
        })
    return region_dict


def get_station_measurements(station_id: str, start_time: Optional[str] = None,
                             end_time: Optional[str] = None) -> pd.DataFrame:
    conn = get_connection()
    query = 'SELECT * FROM measurements WHERE station_id = ? AND is_valid = 1'
    params = [station_id]
    
    if start_time:
        query += ' AND measurement_time >= ?'
        params.append(start_time)
    if end_time:
        query += ' AND measurement_time <= ?'
        params.append(end_time)
    
    query += ' ORDER BY measurement_time'
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    if not df.empty:
        df['measurement_time'] = pd.to_datetime(df['measurement_time'])
    return df


def get_latest_measurements() -> pd.DataFrame:
    conn = get_connection()
    query = '''
        SELECT m.*, s.longitude, s.latitude, s.region, s.station_name, s.zone_type
        FROM measurements m
        INNER JOIN stations s ON m.station_id = s.station_id
        INNER JOIN (
            SELECT station_id, MAX(measurement_time) as max_time
            FROM measurements
            WHERE is_valid = 1
            GROUP BY station_id
        ) latest ON m.station_id = latest.station_id AND m.measurement_time = latest.max_time
        WHERE m.is_valid = 1
        ORDER BY m.station_id
    '''
    df = pd.read_sql_query(query, conn)
    conn.close()
    if not df.empty:
        df['measurement_time'] = pd.to_datetime(df['measurement_time'])
    return df


def get_measurement_time_range() -> Optional[Tuple[datetime, datetime]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT MIN(measurement_time), MAX(measurement_time) FROM measurements WHERE is_valid = 1')
    result = cursor.fetchone()
    conn.close()
    if result and result[0] and result[1]:
        return (datetime.fromisoformat(result[0]), datetime.fromisoformat(result[1]))
    return None


def add_functional_zone(zone_name: str, zone_type: int, geojson_str: str) -> bool:
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO functional_zones (zone_name, zone_type, geojson)
            VALUES (?, ?, ?)
        ''', (zone_name, zone_type, geojson_str))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"添加功能区失败: {e}")
        return False


def get_functional_zones() -> List[Dict]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM functional_zones ORDER BY zone_type, zone_name')
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def clear_functional_zones():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM functional_zones')
    conn.commit()
    conn.close()


def save_noise_prediction(prediction_data: Dict) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO noise_predictions
        (name, road_start_lon, road_start_lat, road_end_lon, road_end_lat,
         traffic_volume, avg_speed, heavy_vehicle_ratio, road_surface,
         barrier_height, barrier_position, prediction_lon, prediction_lat,
         distance, predicted_leq, inserted_loss)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        prediction_data.get('name', '未命名预测'),
        prediction_data.get('road_start_lon'),
        prediction_data.get('road_start_lat'),
        prediction_data.get('road_end_lon'),
        prediction_data.get('road_end_lat'),
        prediction_data.get('traffic_volume'),
        prediction_data.get('avg_speed'),
        prediction_data.get('heavy_vehicle_ratio'),
        prediction_data.get('road_surface'),
        prediction_data.get('barrier_height'),
        prediction_data.get('barrier_position'),
        prediction_data.get('prediction_lon'),
        prediction_data.get('prediction_lat'),
        prediction_data.get('distance'),
        prediction_data.get('predicted_leq'),
        prediction_data.get('inserted_loss')
    ))
    pred_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return pred_id


def get_noise_predictions() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query('SELECT * FROM noise_predictions ORDER BY created_at DESC', conn)
    conn.close()
    return df


def delete_prediction(pred_id: int):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM noise_predictions WHERE id = ?', (pred_id,))
    conn.commit()
    conn.close()


def get_data_statistics() -> Dict:
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM stations')
    station_count = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM measurements WHERE is_valid = 1')
    valid_measurements = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM measurements WHERE is_valid = 0')
    invalid_measurements = cursor.fetchone()[0]
    
    cursor.execute('SELECT MIN(measurement_time), MAX(measurement_time) FROM measurements WHERE is_valid = 1')
    time_range = cursor.fetchone()
    
    cursor.execute('SELECT COUNT(DISTINCT DATE(measurement_time)) FROM measurements WHERE is_valid = 1')
    days_count = cursor.fetchone()[0]
    
    conn.close()
    
    total = valid_measurements + invalid_measurements
    efficiency = (valid_measurements / total * 100) if total > 0 else 0
    
    return {
        'station_count': station_count,
        'valid_measurements': valid_measurements,
        'invalid_measurements': invalid_measurements,
        'total_measurements': total,
        'data_efficiency': efficiency,
        'time_start': time_range[0] if time_range else None,
        'time_end': time_range[1] if time_range else None,
        'days_count': days_count
    }


init_database()
