import numpy as np
from typing import Dict, Optional, Tuple, List
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
from spatial_interpolation import haversine_distance

ROAD_SURFACE_CORRECTION = {
    'asphalt': {'name': '沥青路面', 'correction': 0},
    'concrete': {'name': '水泥路面', 'correction': 2}
}


def calculate_reference_leq(avg_speed: float, heavy_vehicle_ratio: float) -> float:
    """
    计算基准声级 (距道路中心线15m处)
    基于FHWA简化模型
    """
    speed_kmh = avg_speed
    
    base_level = 68.0
    
    speed_correction = 30 * np.log10(speed_kmh / 60.0)
    
    hv_ratio = min(max(heavy_vehicle_ratio, 0), 1.0)
    hv_correction = 10 * np.log10(1 + 5 * hv_ratio)
    
    return base_level + speed_correction + hv_correction


def calculate_distance_correction(traffic_volume: float, distance: float) -> float:
    """
    计算距离衰减修正项
    Q: 车流量(辆/小时)
    D: 距路中心线距离(m)
    """
    D = max(distance, 7.5)
    Q = max(traffic_volume, 1)
    
    correction = 10 * np.log10(Q / D)
    
    return correction


def calculate_barrier_insertion_loss(barrier_height: float, barrier_position: float,
                                     source_height: float = 0.5,
                                     receiver_height: float = 1.5,
                                     distance: float = 50.0,
                                     barrier_length: float = 50.0) -> float:
    """
    使用Maekawa公式计算声屏障插入损失
    barrier_height: 屏障高度(m)
    barrier_position: 屏障距声源距离(m)
    source_height: 声源高度(m) (假设为车辆排气高度0.5m)
    receiver_height: 接收点高度(m) (人耳高度1.5m)
    distance: 声源到接收点距离(m)
    barrier_length: 屏障长度(m)
    """
    if barrier_height <= 0 or distance <= 0:
        return 0.0
    
    a = max(barrier_position, 0.1)
    b = max(distance - barrier_position, 0.1)
    
    delta = np.sqrt(a ** 2 + barrier_height ** 2) + \
            np.sqrt(b ** 2 + barrier_height ** 2) - distance
    
    wavelength = 0.68
    N = (2 * delta) / wavelength
    
    if N <= -0.2:
        IL = 0
    elif N < 0:
        IL = 5 * (1 + 5 * N)
    else:
        IL = 5 + 20 * np.log10(
            np.sqrt(2 * np.pi * N) / np.tanh(np.sqrt(2 * np.pi * N))
        )
    
    length_ratio = barrier_length / (2 * distance)
    if length_ratio < 1.0:
        length_correction = 10 * np.log10(1 + length_ratio)
        IL = IL * (0.5 + 0.5 * length_ratio)
    
    IL = min(max(IL, 0), 25.0)
    
    return IL


def predict_road_traffic_noise(
    road_start_lon: float,
    road_start_lat: float,
    road_end_lon: float,
    road_end_lat: float,
    prediction_lon: float,
    prediction_lat: float,
    traffic_volume: float,
    avg_speed: float = 60.0,
    heavy_vehicle_ratio: float = 0.1,
    road_surface: str = 'asphalt',
    barrier_height: Optional[float] = None,
    barrier_position_ratio: float = 0.5,
    barrier_length: float = 50.0
) -> Dict:
    """
    道路交通噪声预测主函数
    """
    start_point = Point(road_start_lon, road_start_lat)
    end_point = Point(road_end_lon, road_end_lat)
    road_line = LineString([start_point, end_point])
    
    pred_point = Point(prediction_lon, prediction_lat)
    
    nearest_on_road, _ = nearest_points(road_line, pred_point)
    
    distance_lon_lat = haversine_distance(
        nearest_on_road.x, nearest_on_road.y,
        prediction_lon, prediction_lat
    )
    
    reference_leq = calculate_reference_leq(avg_speed, heavy_vehicle_ratio)
    
    dist_correction = calculate_distance_correction(traffic_volume, distance_lon_lat)
    
    surface_correction = ROAD_SURFACE_CORRECTION.get(road_surface, ROAD_SURFACE_CORRECTION['asphalt'])['correction']
    
    speed_correction_detail = 30 * np.log10(max(avg_speed, 1) / 60.0)
    hv_correction_detail = 10 * np.log10(1 + 5 * min(max(heavy_vehicle_ratio, 0), 1.0))
    
    inserted_loss = 0.0
    if barrier_height is not None and barrier_height > 0:
        road_length = haversine_distance(road_start_lon, road_start_lat, road_end_lon, road_end_lat)
        actual_barrier_pos = distance_lon_lat * barrier_position_ratio
        inserted_loss = calculate_barrier_insertion_loss(
            barrier_height=barrier_height,
            barrier_position=actual_barrier_pos,
            distance=distance_lon_lat,
            barrier_length=min(barrier_length, road_length)
        )
    
    predicted_leq = reference_leq + dist_correction + surface_correction - inserted_loss
    
    road_segment_len = haversine_distance(road_start_lon, road_start_lat,
                                          road_end_lon, road_end_lat)
    
    result = {
        'predicted_leq': float(predicted_leq),
        'reference_leq': float(reference_leq),
        'distance_to_road': float(distance_lon_lat),
        'distance_correction': float(dist_correction),
        'surface_correction': float(surface_correction),
        'speed_correction': float(speed_correction_detail),
        'heavy_vehicle_correction': float(hv_correction_detail),
        'barrier_insertion_loss': float(inserted_loss),
        'nearest_point_on_road': {
            'lon': float(nearest_on_road.x),
            'lat': float(nearest_on_road.y)
        },
        'road_length': float(road_segment_len),
        'road_surface_name': ROAD_SURFACE_CORRECTION.get(road_surface, {'name': '未知'})['name']
    }
    
    return result


def generate_prediction_contour(
    road_start_lon: float,
    road_start_lat: float,
    road_end_lon: float,
    road_end_lat: float,
    traffic_volume: float,
    avg_speed: float = 60.0,
    heavy_vehicle_ratio: float = 0.1,
    road_surface: str = 'asphalt',
    barrier_height: Optional[float] = None,
    barrier_position_ratio: float = 0.5,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    resolution: float = 50.0
) -> Dict:
    """
    生成预测噪声等值线网格数据
    """
    if bounds is None:
        min_lon = min(road_start_lon, road_end_lon) - 0.01
        max_lon = max(road_start_lon, road_end_lon) + 0.01
        min_lat = min(road_start_lat, road_end_lat) - 0.01
        max_lat = max(road_start_lat, road_end_lat) + 0.01
        bounds = (min_lon, min_lat, max_lon, max_lat)
    
    min_lon, min_lat, max_lon, max_lat = bounds
    
    center_lon = (min_lon + max_lon) / 2
    center_lat = (min_lat + max_lat) / 2
    
    dx = (max_lon - min_lon) * 111320 * np.cos(np.radians(center_lat))
    dy = (max_lat - min_lat) * 110540
    
    nx = max(2, int(np.ceil(dx / resolution)) + 1)
    ny = max(2, int(np.ceil(dy / resolution)) + 1)
    
    x_lon = np.linspace(min_lon, max_lon, nx)
    y_lat = np.linspace(min_lat, max_lat, ny)
    
    z_grid = np.zeros((ny, nx))
    
    for i in range(ny):
        for j in range(nx):
            result = predict_road_traffic_noise(
                road_start_lon=road_start_lon,
                road_start_lat=road_start_lat,
                road_end_lon=road_end_lon,
                road_end_lat=road_end_lat,
                prediction_lon=x_lon[j],
                prediction_lat=y_lat[i],
                traffic_volume=traffic_volume,
                avg_speed=avg_speed,
                heavy_vehicle_ratio=heavy_vehicle_ratio,
                road_surface=road_surface,
                barrier_height=barrier_height,
                barrier_position_ratio=barrier_position_ratio
            )
            z_grid[i, j] = result['predicted_leq']
    
    return {
        'grid_lon': x_lon,
        'grid_lat': y_lat,
        'z': z_grid,
        'bounds': bounds,
        'shape': (ny, nx),
        'resolution': resolution,
        'road_coords': {
            'start': {'lon': road_start_lon, 'lat': road_start_lat},
            'end': {'lon': road_end_lon, 'lat': road_end_lat}
        }
    }
