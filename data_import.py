import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, List, Dict, Optional
import io
import re
from data_models import FREQUENCY_BANDS, add_measurements, add_station

MIN_SOUND_LEVEL = 20
MAX_SOUND_LEVEL = 130

REQUIRED_COLUMNS = {
    'station_id': ['站点编号', 'station_id', 'StationID', 'site_id'],
    'longitude': ['经度', 'longitude', 'lon', 'Lng', '经度(°)'],
    'latitude': ['纬度', 'latitude', 'lat', 'Lat', '纬度(°)'],
    'measurement_time': ['监测时间', 'measurement_time', 'time', 'datetime', '时间'],
    'leq': ['等效连续声级', 'Leq', 'leq', 'LAeq', '等效声级']
}

OPTIONAL_COLUMNS = {
    'lmax': ['最大声级', 'Lmax', 'lmax'],
    'l10': ['L10', 'l10', '统计声级L10'],
    'l50': ['L50', 'l50', '统计声级L50'],
    'l90': ['L90', 'l90', '统计声级L90'],
    'region': ['区域', 'region', 'area', '所属区域'],
    'station_name': ['站点名称', 'station_name', 'name', '站点名']
}

FREQ_COLUMN_PATTERNS = [
    r'^freq_?(\d+)$',
    r'^(\d+)Hz$',
    r'^(\d+)赫兹$',
    r'^(\d+\.?\d*)$'
]


def detect_column_mapping(df_columns: List[str]) -> Dict[str, str]:
    column_mapping = {}
    
    for std_col, possible_names in REQUIRED_COLUMNS.items():
        for col in df_columns:
            col_lower = col.lower().strip()
            if col_lower in [name.lower() for name in possible_names]:
                column_mapping[col] = std_col
                break
    
    for std_col, possible_names in OPTIONAL_COLUMNS.items():
        for col in df_columns:
            col_lower = col.lower().strip()
            if col_lower in [name.lower() for name in possible_names]:
                column_mapping[col] = std_col
                break
    
    for col in df_columns:
        if col in column_mapping:
            continue
        col_clean = col.strip()
        for pattern in FREQ_COLUMN_PATTERNS:
            match = re.match(pattern, col_clean, re.IGNORECASE)
            if match:
                freq_value = float(match.group(1))
                closest_band = min(FREQUENCY_BANDS, key=lambda x: abs(x - freq_value))
                if abs(closest_band - freq_value) / closest_band < 0.2:
                    column_mapping[col] = f'freq_{closest_band}'
                    break
    
    return column_mapping


def validate_sound_level(value: float, field_name: str = '') -> Tuple[bool, Optional[str]]:
    if pd.isna(value):
        return True, None
    try:
        val = float(value)
        if val < MIN_SOUND_LEVEL or val > MAX_SOUND_LEVEL:
            return False, f'{field_name}数值{val:.1f}超出有效范围[{MIN_SOUND_LEVEL}, {MAX_SOUND_LEVEL}]'
        return True, None
    except (ValueError, TypeError):
        return False, f'{field_name}数值格式无效'


def parse_time_string(time_str: str) -> Optional[datetime]:
    if pd.isna(time_str):
        return None
    if isinstance(time_str, datetime):
        return time_str.replace(minute=0, second=0, microsecond=0)
    
    time_str = str(time_str).strip()
    
    formats = [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M',
        '%Y-%m-%d %H',
        '%Y/%m/%d %H:%M:%S',
        '%Y/%m/%d %H:%M',
        '%Y/%m/%d %H',
        '%Y%m%d%H',
        '%Y%m%d%H%M%S',
        '%m/%d/%Y %H:%M',
        '%d/%m/%Y %H:%M'
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(time_str, fmt)
            return dt.replace(minute=0, second=0, microsecond=0)
        except ValueError:
            continue
    
    try:
        dt = pd.to_datetime(time_str)
        if pd.notna(dt):
            return dt.to_pydatetime().replace(minute=0, second=0, microsecond=0)
    except Exception:
        pass
    
    return None


def load_csv_from_bytes(file_bytes: bytes) -> Tuple[Optional[pd.DataFrame], str]:
    try:
        content_str = file_bytes.decode('utf-8')
    except UnicodeDecodeError:
        try:
            content_str = file_bytes.decode('gbk')
        except UnicodeDecodeError:
            content_str = file_bytes.decode('utf-8', errors='ignore')
    
    try:
        df = pd.read_csv(io.StringIO(content_str))
        return df, ''
    except Exception as e:
        return None, f'CSV解析失败: {str(e)}'


def validate_and_transform_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict], List[str]]:
    warnings = []
    errors = []
    
    column_mapping = detect_column_mapping(df.columns.tolist())
    df_renamed = df.rename(columns=column_mapping)
    
    missing_required = []
    for std_col in REQUIRED_COLUMNS.keys():
        if std_col not in df_renamed.columns:
            missing_required.append(std_col)
    
    if missing_required:
        error_msg = f'缺少必需字段: {", ".join(missing_required)}。请确保CSV包含以下列: 站点编号, 经度, 纬度, 监测时间, 等效连续声级Leq'
        raise ValueError(error_msg)
    
    processed_rows = []
    
    for idx, row in df_renamed.iterrows():
        row_errors = []
        row_warnings = []
        processed_row = {}
        
        station_id = str(row['station_id']).strip()
        if not station_id or station_id.lower() in ['nan', 'none']:
            row_errors.append('站点编号不能为空')
        processed_row['station_id'] = station_id
        
        try:
            lon = float(row['longitude'])
            if not (-180 <= lon <= 180):
                row_errors.append(f'经度值{lon}超出有效范围[-180, 180]')
            processed_row['longitude'] = lon
        except (ValueError, TypeError):
            row_errors.append(f'经度格式无效: {row["longitude"]}')
            processed_row['longitude'] = np.nan
        
        try:
            lat = float(row['latitude'])
            if not (-90 <= lat <= 90):
                row_errors.append(f'纬度值{lat}超出有效范围[-90, 90]')
            processed_row['latitude'] = lat
        except (ValueError, TypeError):
            row_errors.append(f'纬度格式无效: {row["latitude"]}')
            processed_row['latitude'] = np.nan
        
        parsed_time = parse_time_string(row['measurement_time'])
        if parsed_time is None:
            row_errors.append(f'时间格式无法识别: {row["measurement_time"]}')
        processed_row['measurement_time'] = parsed_time
        
        try:
            leq = float(row['leq'])
            valid, msg = validate_sound_level(leq, 'Leq')
            if not valid:
                row_warnings.append(msg)
            processed_row['leq'] = leq
        except (ValueError, TypeError):
            row_errors.append(f'Leq格式无效: {row["leq"]}')
            processed_row['leq'] = np.nan
        
        for opt_col in ['lmax', 'l10', 'l50', 'l90']:
            if opt_col in df_renamed.columns and pd.notna(row[opt_col]):
                try:
                    val = float(row[opt_col])
                    valid, msg = validate_sound_level(val, opt_col.upper())
                    if not valid:
                        row_warnings.append(msg)
                    processed_row[opt_col] = val
                except (ValueError, TypeError):
                    row_warnings.append(f'{opt_col.upper()}格式无效')
                    processed_row[opt_col] = np.nan
            else:
                processed_row[opt_col] = np.nan
        
        for opt_col in ['region', 'station_name']:
            if opt_col in df_renamed.columns:
                val = row[opt_col]
                processed_row[opt_col] = None if pd.isna(val) else str(val).strip()
            else:
                processed_row[opt_col] = None
        
        for freq in FREQUENCY_BANDS:
            freq_col = f'freq_{freq}'
            if freq_col in df_renamed.columns and pd.notna(row[freq_col]):
                try:
                    val = float(row[freq_col])
                    valid, msg = validate_sound_level(val, f'{freq}Hz')
                    if not valid:
                        row_warnings.append(msg)
                    processed_row[freq_col] = val
                except (ValueError, TypeError):
                    row_warnings.append(f'{freq}Hz频谱数据格式无效')
                    processed_row[freq_col] = np.nan
        
        if row_errors:
            errors.append({
                'row': idx + 2,
                'station_id': station_id,
                'errors': row_errors,
                'data': {k: v for k, v in processed_row.items()}
            })
        else:
            processed_rows.append(processed_row)
            if row_warnings:
                for w in row_warnings:
                    warnings.append(f'第{idx + 2}行({station_id}): {w}')
    
    result_df = pd.DataFrame(processed_rows)
    
    if 'longitude' in result_df.columns and 'latitude' in result_df.columns and 'station_id' in result_df.columns:
        station_info = result_df[['station_id', 'longitude', 'latitude']].drop_duplicates('station_id')
        for _, srow in station_info.iterrows():
            region = None
            station_name = None
            if 'region' in result_df.columns:
                region_val = result_df[result_df['station_id'] == srow['station_id']]['region'].iloc[0]
                region = region_val if pd.notna(region_val) else None
            if 'station_name' in result_df.columns:
                name_val = result_df[result_df['station_id'] == srow['station_id']]['station_name'].iloc[0]
                station_name = name_val if pd.notna(name_val) else None
            add_station(
                station_id=srow['station_id'],
                longitude=float(srow['longitude']),
                latitude=float(srow['latitude']),
                station_name=station_name,
                region=region
            )
    
    return result_df, errors, warnings


def import_csv_data(file_bytes: bytes) -> Dict:
    result = {
        'success': False,
        'total_rows': 0,
        'imported_rows': 0,
        'error_rows': 0,
        'errors': [],
        'warnings': [],
        'message': ''
    }
    
    df, parse_error = load_csv_from_bytes(file_bytes)
    if parse_error:
        result['message'] = parse_error
        return result
    
    result['total_rows'] = len(df)
    
    try:
        processed_df, errors, warnings = validate_and_transform_data(df)
        result['errors'] = errors
        result['warnings'] = warnings
        result['error_rows'] = len(errors)
        
        if len(processed_df) > 0:
            success_count, db_errors, db_error_list = add_measurements(processed_df)
            result['imported_rows'] = success_count
            result['error_rows'] += len(db_errors)
            for err in db_error_list:
                result['errors'].append({
                    'row': err['row'] + 2,
                    'errors': [err['error']]
                })
        
        result['success'] = True
        result['message'] = f'导入完成! 共{result["total_rows"]}行, 成功导入{result["imported_rows"]}行, 错误{result["error_rows"]}行'
        
        if warnings:
            result['message'] += f', 警告{len(warnings)}条'
    except Exception as e:
        result['message'] = f'数据处理失败: {str(e)}'
        result['success'] = False
    
    return result


def get_import_template() -> bytes:
    columns = (
        ['站点编号', '站点名称', '区域', '经度', '纬度', '监测时间',
         '等效连续声级', '最大声级', 'L10', 'L50', 'L90'] +
        [f'{f}Hz' for f in FREQUENCY_BANDS]
    )
    
    example_row = [
        'ST001', '监测站A', '东城区', '116.4074', '39.9042',
        '2024-01-15 08:00:00', '58.5', '72.3', '62.1', '56.8', '52.4'
    ] + [f'{np.random.uniform(30, 70):.1f}' for _ in FREQUENCY_BANDS]
    
    output = io.StringIO()
    output.write(','.join(columns) + '\n')
    output.write(','.join(example_row) + '\n')
    
    return output.getvalue().encode('utf-8')
