import numpy as np
from scipy.spatial import cKDTree, Voronoi
from scipy.spatial.distance import cdist
from scipy.optimize import least_squares
from typing import Tuple, Dict, Optional, List, Callable
import time


class SphericalVariogram:
    @staticmethod
    def model(h, nugget, sill, range_):
        h = np.asarray(h, dtype=np.float64)
        result = np.zeros_like(h)
        mask = h <= range_
        hr = h[mask] / range_
        result[mask] = nugget + (sill - nugget) * (1.5 * hr - 0.5 * hr ** 3)
        result[~mask] = sill
        return result

    @staticmethod
    def derivative(h, nugget, sill, range_):
        h = np.asarray(h, dtype=np.float64)
        result = np.zeros_like(h)
        mask = (h > 0) & (h <= range_)
        result[mask] = (sill - nugget) * (1.5 / range_ - 1.5 * h[mask] ** 2 / range_ ** 3)
        return result


class ExponentialVariogram:
    @staticmethod
    def model(h, nugget, sill, range_):
        h = np.asarray(h, dtype=np.float64)
        effective_range = range_ / 3.0
        return nugget + (sill - nugget) * (1 - np.exp(-h / effective_range))


class GaussianVariogram:
    @staticmethod
    def model(h, nugget, sill, range_):
        h = np.asarray(h, dtype=np.float64)
        effective_range = range_ / np.sqrt(3)
        return nugget + (sill - nugget) * (1 - np.exp(-(h / effective_range) ** 2))


VARIOGRAM_MODELS = {
    'spherical': SphericalVariogram,
    'exponential': ExponentialVariogram,
    'gaussian': GaussianVariogram
}

VARIOGRAM_NAMES = {
    'spherical': '球状模型',
    'exponential': '指数模型',
    'gaussian': '高斯模型'
}


def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    R = 6371000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def coordinates_to_meters(coords: np.ndarray, center_lon: Optional[float] = None,
                          center_lat: Optional[float] = None) -> np.ndarray:
    if len(coords) == 0:
        return coords
    
    if center_lon is None:
        center_lon = np.mean(coords[:, 0])
    if center_lat is None:
        center_lat = np.mean(coords[:, 1])
    
    meters = np.zeros_like(coords)
    meters[:, 0] = (coords[:, 0] - center_lon) * 111320 * np.cos(np.radians(center_lat))
    meters[:, 1] = (coords[:, 1] - center_lat) * 110540
    return meters


def create_grid(bounds: Tuple[float, float, float, float],
                resolution: float = 50.0) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    min_lon, min_lat, max_lon, max_lat = bounds
    
    center_lon = (min_lon + max_lon) / 2
    center_lat = (min_lat + max_lat) / 2
    
    dx = (max_lon - min_lon) * 111320 * np.cos(np.radians(center_lat))
    dy = (max_lat - min_lat) * 110540
    
    nx = max(2, int(np.ceil(dx / resolution)) + 1)
    ny = max(2, int(np.ceil(dy / resolution)) + 1)
    
    x_lon = np.linspace(min_lon, max_lon, nx)
    y_lat = np.linspace(min_lat, max_lat, ny)
    
    grid_lon, grid_lat = np.meshgrid(x_lon, y_lat)
    grid_points = np.column_stack([grid_lon.ravel(), grid_lat.ravel()])
    
    return grid_points, (x_lon, y_lat), (ny, nx)


def compute_experimental_variogram(coords_m: np.ndarray, values: np.ndarray,
                                   n_lags: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(coords_m)
    distances = []
    squared_diffs = []
    
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(coords_m[i] - coords_m[j])
            distances.append(d)
            squared_diffs.append((values[i] - values[j]) ** 2)
    
    distances = np.array(distances)
    squared_diffs = np.array(squared_diffs)
    
    if len(distances) == 0:
        return np.array([]), np.array([]), np.array([])
    
    max_dist = np.max(distances)
    lag_edges = np.linspace(0, max_dist, n_lags + 1)
    
    lags = []
    gammas = []
    counts = []
    
    for k in range(n_lags):
        mask = (distances >= lag_edges[k]) & (distances < lag_edges[k + 1])
        if np.sum(mask) > 0:
            lags.append((lag_edges[k] + lag_edges[k + 1]) / 2)
            gammas.append(np.mean(squared_diffs[mask]) / 2)
            counts.append(np.sum(mask))
    
    return np.array(lags), np.array(gammas), np.array(counts)


def fit_variogram(lags: np.ndarray, gammas: np.ndarray, model_name: str = 'spherical',
                  initial_params: Optional[List[float]] = None,
                  manual_params: Optional[Dict] = None) -> Dict:
    if len(lags) < 3 or len(gammas) < 3:
        return {
            'success': False,
            'model': model_name,
            'nugget': 0,
            'sill': np.var(gammas) if len(gammas) > 0 else 1,
            'range': 1000,
            'fitted': False
        }
    
    model_class = VARIOGRAM_MODELS.get(model_name, SphericalVariogram)
    
    if manual_params is not None:
        nugget = manual_params.get('nugget', 0)
        sill = manual_params.get('sill', np.max(gammas) if len(gammas) > 0 else 1)
        range_ = manual_params.get('range', np.max(lags) / 2 if len(lags) > 0 else 1000)
        return {
            'success': True,
            'model': model_name,
            'nugget': nugget,
            'sill': sill,
            'range': range_,
            'fitted': False,
            'manual': True
        }
    
    if initial_params is None:
        nugget_init = np.min(gammas) if len(gammas) > 0 else 0
        sill_init = np.max(gammas) if len(gammas) > 0 else 1
        range_init = np.max(lags) * 0.5 if len(lags) > 0 else 1000
        initial_params = [nugget_init, sill_init, range_init]
    
    def residuals(params):
        nugget, sill, range_ = params
        if range_ <= 0 or sill <= nugget:
            return np.full_like(gammas, 1e10)
        predicted = model_class.model(lags, nugget, sill, range_)
        return predicted - gammas
    
    try:
        bounds_lower = [0, 0, 1]
        bounds_upper = [np.inf, np.inf, np.inf]
        result = least_squares(residuals, initial_params,
                               bounds=(bounds_lower, bounds_upper),
                               method='trf', max_nfev=1000)
        
        nugget, sill, range_ = result.x
        return {
            'success': True,
            'model': model_name,
            'nugget': max(0, nugget),
            'sill': max(nugget, sill),
            'range': max(1, range_),
            'fitted': True,
            'manual': False,
            'cost': result.cost
        }
    except Exception as e:
        print(f"Variogram fitting failed: {e}")
        nugget = max(0, initial_params[0])
        sill = max(nugget, initial_params[1])
        range_ = max(1, initial_params[2])
        return {
            'success': True,
            'model': model_name,
            'nugget': nugget,
            'sill': sill,
            'range': range_,
            'fitted': False,
            'manual': False,
            'error': str(e)
        }


def interpolate_idw(sample_coords: np.ndarray, sample_values: np.ndarray,
                    grid_points: np.ndarray, power: float = 2.0,
                    max_points: int = 20) -> np.ndarray:
    if len(sample_coords) == 0:
        return np.full(len(grid_points), np.nan)
    
    sample_meters = coordinates_to_meters(sample_coords)
    center_lon = np.mean(sample_coords[:, 0])
    center_lat = np.mean(sample_coords[:, 1])
    grid_meters = coordinates_to_meters(grid_points, center_lon, center_lat)
    
    tree = cKDTree(sample_meters)
    k = min(max_points, len(sample_coords))
    
    distances, indices = tree.query(grid_meters, k=k)
    
    if k == 1:
        distances = distances.reshape(-1, 1)
        indices = indices.reshape(-1, 1)
    
    result = np.zeros(len(grid_points))
    
    for i in range(len(grid_points)):
        dists = distances[i]
        idxs = indices[i]
        vals = sample_values[idxs]
        
        mask_zero = dists < 1e-10
        if np.any(mask_zero):
            result[i] = vals[mask_zero][0]
        else:
            weights = 1.0 / (dists ** power)
            weights_sum = np.sum(weights)
            if weights_sum > 0:
                result[i] = np.sum(weights * vals) / weights_sum
            else:
                result[i] = np.mean(vals)
    
    return result


def interpolate_kriging(sample_coords: np.ndarray, sample_values: np.ndarray,
                        grid_points: np.ndarray, variogram_params: Dict) -> Tuple[np.ndarray, np.ndarray]:
    n = len(sample_coords)
    if n == 0:
        return (np.full(len(grid_points), np.nan), np.full(len(grid_points), np.nan))
    
    sample_meters = coordinates_to_meters(sample_coords)
    center_lon = np.mean(sample_coords[:, 0])
    center_lat = np.mean(sample_coords[:, 1])
    grid_meters = coordinates_to_meters(grid_points, center_lon, center_lat)
    
    model_class = VARIOGRAM_MODELS.get(variogram_params.get('model', 'spherical'), SphericalVariogram)
    nugget = variogram_params.get('nugget', 0)
    sill = variogram_params.get('sill', 1)
    range_ = variogram_params.get('range', 1000)
    
    K = np.zeros((n + 1, n + 1))
    for i in range(n):
        for j in range(n):
            if i == j:
                K[i, j] = nugget
            else:
                d = np.linalg.norm(sample_meters[i] - sample_meters[j])
                K[i, j] = model_class.model(d, nugget, sill, range_)
        K[i, n] = 1.0
        K[n, i] = 1.0
    K[n, n] = 0.0
    
    try:
        K_inv = np.linalg.inv(K)
    except np.linalg.LinAlgError:
        K_perturbed = K + np.eye(n + 1) * 1e-8
        K_inv = np.linalg.inv(K_perturbed)
    
    predictions = np.zeros(len(grid_points))
    variances = np.zeros(len(grid_points))
    
    y = np.zeros(n + 1)
    y[:n] = sample_values
    
    for g_idx in range(len(grid_points)):
        gp = grid_meters[g_idx]
        dists = np.linalg.norm(sample_meters - gp, axis=1)
        k = np.zeros(n + 1)
        k[:n] = model_class.model(dists, nugget, sill, range_)
        k[n] = 1.0
        
        weights = K_inv @ k
        predictions[g_idx] = np.dot(weights[:n], sample_values)
        
        sigma2 = model_class.model(0, nugget, sill, range_) - np.dot(weights, k)
        variances[g_idx] = max(0, sigma2)
    
    return predictions, variances


def interpolate_natural_neighbor(sample_coords: np.ndarray, sample_values: np.ndarray,
                                 grid_points: np.ndarray) -> np.ndarray:
    n = len(sample_coords)
    if n == 0:
        return np.full(len(grid_points), np.nan)
    
    sample_meters = coordinates_to_meters(sample_coords)
    center_lon = np.mean(sample_coords[:, 0])
    center_lat = np.mean(sample_coords[:, 1])
    grid_meters = coordinates_to_meters(grid_points, center_lon, center_lat)
    
    if n < 4:
        tree = cKDTree(sample_meters)
        distances, indices = tree.query(grid_meters, k=min(n, 3))
        if n == 1:
            return np.full(len(grid_points), sample_values[0])
        
        predictions = np.zeros(len(grid_points))
        for i in range(len(grid_points)):
            if n == 2:
                dists = distances[i]
                idxs = indices[i]
                if dists[0] < 1e-10:
                    predictions[i] = sample_values[idxs[0]]
                elif dists[1] < 1e-10:
                    predictions[i] = sample_values[idxs[1]]
                else:
                    w = 1.0 / dists ** 2
                    predictions[i] = np.sum(w * sample_values[idxs]) / np.sum(w)
            else:
                dists = distances[i]
                idxs = indices[i]
                mask_zero = dists < 1e-10
                if np.any(mask_zero):
                    predictions[i] = sample_values[idxs[mask_zero][0]]
                else:
                    w = 1.0 / dists ** 2
                    predictions[i] = np.sum(w * sample_values[idxs]) / np.sum(w)
        return predictions
    
    tree = cKDTree(sample_meters)
    predictions = np.zeros(len(grid_points))
    
    for g_idx in range(len(grid_points)):
        gp = grid_meters[g_idx]
        
        k_neighbors = min(15, n - 1)
        dists, indices = tree.query(gp, k=k_neighbors + 1)
        
        if dists[0] < 1e-10:
            predictions[g_idx] = sample_values[indices[0]]
            continue
        
        neighbor_indices = indices[:k_neighbors]
        neighbor_coords = sample_meters[neighbor_indices]
        neighbor_values = sample_values[neighbor_indices]
        
        extended_coords = np.vstack([neighbor_coords, gp])
        
        try:
            vor = Voronoi(extended_coords)
            
            point_region = vor.point_region[-1]
            region_vertices = vor.regions[point_region]
            region_vertices = [v for v in region_vertices if v != -1]
            
            if len(region_vertices) < 3:
                d = dists[:min(4, n)]
                idx = indices[:min(4, n)]
                w = 1.0 / d ** 2
                predictions[g_idx] = np.sum(w * sample_values[idx]) / np.sum(w)
                continue
            
            original_volumes = np.zeros(k_neighbors)
            for nb_i in range(k_neighbors):
                pr = vor.point_region[nb_i]
                rv = vor.regions[pr]
                rv = [v for v in rv if v != -1]
                if len(rv) >= 3:
                    verts = vor.vertices[rv]
                    original_volumes[nb_i] = _polygon_area(verts)
            
            extended_coords_no_gp = neighbor_coords
            vor_no_gp = Voronoi(extended_coords_no_gp)
            
            new_volumes = np.zeros(k_neighbors)
            for nb_i in range(k_neighbors):
                pr = vor_no_gp.point_region[nb_i]
                rv = vor_no_gp.regions[pr]
                rv = [v for v in rv if v != -1]
                if len(rv) >= 3:
                    verts = vor_no_gp.vertices[rv]
                    new_volumes[nb_i] = _polygon_area(verts)
            
            lambda_weights = new_volumes - original_volumes
            lambda_weights = np.clip(lambda_weights, 0, None)
            
            weight_sum = np.sum(lambda_weights)
            if weight_sum > 1e-10:
                lambda_weights /= weight_sum
                predictions[g_idx] = np.sum(lambda_weights * neighbor_values)
            else:
                d = dists[:min(4, n)]
                idx = indices[:min(4, n)]
                w = 1.0 / d ** 2
                predictions[g_idx] = np.sum(w * sample_values[idx]) / np.sum(w)
                
        except Exception:
            d = dists[:min(4, n)]
            idx = indices[:min(4, n)]
            w = 1.0 / d ** 2
            predictions[g_idx] = np.sum(w * sample_values[idx]) / np.sum(w)
    
    return predictions


def _polygon_area(vertices: np.ndarray) -> float:
    if len(vertices) < 3:
        return 0.0
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def run_interpolation(sample_coords: np.ndarray, sample_values: np.ndarray,
                      method: str = 'idw', bounds: Optional[Tuple] = None,
                      resolution: float = 50.0, **kwargs) -> Dict:
    start_time = time.time()
    
    if bounds is None:
        if len(sample_coords) > 0:
            margin_lon = (np.max(sample_coords[:, 0]) - np.min(sample_coords[:, 0])) * 0.15
            margin_lat = (np.max(sample_coords[:, 1]) - np.min(sample_coords[:, 1])) * 0.15
            margin_lon = max(margin_lon, 0.01)
            margin_lat = max(margin_lat, 0.01)
            bounds = (
                np.min(sample_coords[:, 0]) - margin_lon,
                np.min(sample_coords[:, 1]) - margin_lat,
                np.max(sample_coords[:, 0]) + margin_lon,
                np.max(sample_coords[:, 1]) + margin_lat
            )
        else:
            bounds = (116.0, 39.5, 117.0, 40.5)
    
    grid_points, axes, grid_shape = create_grid(bounds, resolution)
    
    variance = None
    
    if method == 'idw':
        power = kwargs.get('power', 2.0)
        max_points = kwargs.get('max_points', 20)
        z = interpolate_idw(sample_coords, sample_values, grid_points, power, max_points)
    
    elif method == 'kriging':
        variogram_params = kwargs.get('variogram_params', None)
        if variogram_params is None:
            sample_m = coordinates_to_meters(sample_coords)
            lags, gammas, _ = compute_experimental_variogram(sample_m, sample_values)
            model_name = kwargs.get('variogram_model', 'spherical')
            variogram_params = fit_variogram(lags, gammas, model_name)
        z, variance = interpolate_kriging(sample_coords, sample_values, grid_points, variogram_params)
        if variance is not None:
            variance = variance.reshape(grid_shape)
    
    elif method == 'natural_neighbor':
        z = interpolate_natural_neighbor(sample_coords, sample_values, grid_points)
    
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    
    z_grid = z.reshape(grid_shape)
    
    end_time = time.time()
    
    return {
        'grid_points': grid_points,
        'grid_lon': axes[0],
        'grid_lat': axes[1],
        'z': z_grid,
        'variance': variance,
        'bounds': bounds,
        'shape': grid_shape,
        'resolution': resolution,
        'method': method,
        'time': end_time - start_time,
        'variogram_params': variogram_params if method == 'kriging' else None
    }
