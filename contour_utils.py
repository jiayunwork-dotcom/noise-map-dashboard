import numpy as np
from typing import List, Tuple, Dict, Optional
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, Normalize
import time

NOISE_COLOR_BREAKPOINTS = [
    (40, '#0000FF'),
    (45, '#0066FF'),
    (50, '#00CCCC'),
    (55, '#00CC00'),
    (60, '#66CC00'),
    (65, '#FFFF00'),
    (70, '#FF9900'),
    (75, '#FF0000'),
    (80, '#990000'),
    (85, '#660033')
]

NOISE_LEVEL_RANGES = [
    {'min': 0, 'max': 45, 'label': '安静 (<45dB)', 'color': '#0066FF', 'r': 0, 'g': 102, 'b': 255},
    {'min': 45, 'max': 55, 'label': '一般 (45-55dB)', 'color': '#00CC00', 'r': 0, 'g': 204, 'b': 0},
    {'min': 55, 'max': 65, 'label': '较吵 (55-65dB)', 'color': '#FFFF00', 'r': 255, 'g': 255, 'b': 0},
    {'min': 65, 'max': 75, 'label': '嘈杂 (65-75dB)', 'color': '#FF9900', 'r': 255, 'g': 153, 'b': 0},
    {'min': 75, 'max': 200, 'label': '严重 (>75dB)', 'color': '#FF0000', 'r': 255, 'g': 0, 'b': 0}
]


def create_noise_colormap():
    positions = []
    colors = []
    
    sorted_bp = sorted(NOISE_COLOR_BREAKPOINTS, key=lambda x: x[0])
    min_db = sorted_bp[0][0]
    max_db = sorted_bp[-1][0]
    db_range = max_db - min_db
    
    for db_val, hex_color in sorted_bp:
        position = (db_val - min_db) / db_range if db_range > 0 else 0
        position = max(0.0, min(1.0, position))
        positions.append(position)
        
        h = hex_color.lstrip('#')
        rgb = tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))
        colors.append(rgb)
    
    cdict = {'red': [], 'green': [], 'blue': []}
    for pos, rgb in zip(positions, colors):
        cdict['red'].append((pos, rgb[0], rgb[0]))
        cdict['green'].append((pos, rgb[1], rgb[1]))
        cdict['blue'].append((pos, rgb[2], rgb[2]))
    
    return LinearSegmentedColormap('noise_cmap', cdict)


NOISE_CMAP = create_noise_colormap()


def value_to_rgb(value: float) -> Tuple[int, int, int, int]:
    if np.isnan(value):
        return (0, 0, 0, 0)
    
    vmin = 40
    vmax = 85
    
    if value <= vmin:
        h = NOISE_COLOR_BREAKPOINTS[0][1].lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4)) + (200,)
    if value >= vmax:
        h = NOISE_COLOR_BREAKPOINTS[-1][1].lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4)) + (200,)
    
    sorted_bp = sorted(NOISE_COLOR_BREAKPOINTS, key=lambda x: x[0])
    
    for i in range(len(sorted_bp) - 1):
        db1, c1 = sorted_bp[i]
        db2, c2 = sorted_bp[i + 1]
        
        if db1 <= value <= db2:
            t = (value - db1) / (db2 - db1) if db2 != db1 else 0
            h1 = c1.lstrip('#')
            h2 = c2.lstrip('#')
            r1, g1, b1 = tuple(int(h1[j:j+2], 16) for j in (0, 2, 4))
            r2, g2, b2 = tuple(int(h2[j:j+2], 16) for j in (0, 2, 4))
            
            r = int(r1 + t * (r2 - r1))
            g = int(g1 + t * (g2 - g1))
            b = int(b1 + t * (b2 - b1))
            return (r, g, b, 200)
    
    return (128, 128, 128, 200)


def value_to_hex(value: float) -> str:
    r, g, b, _ = value_to_rgb(value)
    return f'#{r:02x}{g:02x}{b:02x}'


def get_noise_level_label(value: float) -> str:
    for level in NOISE_LEVEL_RANGES:
        if level['min'] <= value < level['max']:
            return level['label']
    return NOISE_LEVEL_RANGES[-1]['label']


def marching_squares(grid: np.ndarray, grid_lon: np.ndarray, grid_lat: np.ndarray,
                     levels: Optional[List[float]] = None,
                     interval: float = 5.0) -> List[Dict]:
    start_time = time.time()
    
    if grid.size == 0:
        return []
    
    valid_data = grid[~np.isnan(grid)]
    if len(valid_data) == 0:
        return []
    
    vmin = float(np.floor(np.nanmin(grid) / interval) * interval)
    vmax = float(np.ceil(np.nanmax(grid) / interval) * interval)
    
    if levels is None:
        levels = list(np.arange(vmin, vmax + interval, interval))
    else:
        levels = sorted([float(l) for l in levels])
    
    all_contours = []
    
    for level in levels:
        segments = _extract_contour_segments(grid, level)
        
        lon_lines = []
        for seg in segments:
            lon_seg = []
            for (row, col) in seg:
                row = int(np.clip(row, 0, len(grid_lat) - 1))
                col = int(np.clip(col, 0, len(grid_lon) - 1))
                
                row_frac = row - int(row)
                col_frac = col - int(col)
                
                r0 = min(int(row), len(grid_lat) - 1)
                r1 = min(int(row) + 1, len(grid_lat) - 1)
                c0 = min(int(col), len(grid_lon) - 1)
                c1 = min(int(col) + 1, len(grid_lon) - 1)
                
                lat = grid_lat[r0] + row_frac * (grid_lat[r1] - grid_lat[r0])
                lon = grid_lon[c0] + col_frac * (grid_lon[c1] - grid_lon[c0])
                
                lon_seg.append((float(lon), float(lat)))
            
            if len(lon_seg) >= 2:
                lon_lines.append(lon_seg)
        
        if lon_lines:
            all_contours.append({
                'level': level,
                'lines': lon_lines
            })
    
    elapsed = time.time() - start_time
    
    return all_contours


def _extract_contour_segments(grid: np.ndarray, level: float) -> List[List[Tuple[float, float]]]:
    rows, cols = grid.shape
    if rows < 2 or cols < 2:
        return []
    
    segments = []
    
    for i in range(rows - 1):
        for j in range(cols - 1):
            v00 = grid[i, j]
            v10 = grid[i + 1, j]
            v01 = grid[i, j + 1]
            v11 = grid[i + 1, j + 1]
            
            if any(np.isnan([v00, v10, v01, v11])):
                continue
            
            index = 0
            if v00 >= level: index |= 1
            if v10 >= level: index |= 2
            if v11 >= level: index |= 4
            if v01 >= level: index |= 8
            
            if index == 0 or index == 15:
                continue
            
            def interp_y(row_a, row_b, val_a, val_b):
                if val_b == val_a:
                    return row_a
                return row_a + (level - val_a) / (val_b - val_a) * (row_b - row_a)
            
            def interp_x(col_a, col_b, val_a, val_b):
                if val_b == val_a:
                    return col_a
                return col_a + (level - val_a) / (val_b - val_a) * (col_b - col_a)
            
            edges = {
                'top': (i, interp_x(j, j + 1, v00, v01)),
                'bottom': (i + 1, interp_x(j, j + 1, v10, v11)),
                'left': (interp_y(i, i + 1, v00, v10), j),
                'right': (interp_y(i, i + 1, v01, v11), j + 1)
            }
            
            def seg_points(idx_val):
                segs_map = {
                    1: [edges['top'], edges['left']],
                    2: [edges['bottom'], edges['left']],
                    3: [edges['top'], edges['bottom']],
                    4: [edges['bottom'], edges['right']],
                    5: [edges['top'], edges['left'], edges['bottom'], edges['right']],
                    6: [edges['left'], edges['right']],
                    7: [edges['top'], edges['right']],
                    8: [edges['top'], edges['right']],
                    9: [edges['left'], edges['right']],
                    10: [edges['top'], edges['left'], edges['bottom'], edges['right']],
                    11: [edges['bottom'], edges['right']],
                    12: [edges['top'], edges['bottom']],
                    13: [edges['bottom'], edges['left']],
                    14: [edges['top'], edges['left']]
                }
                return segs_map.get(idx_val, [])
            
            pts = seg_points(index)
            
            if len(pts) == 2:
                segments.append([pts[0], pts[1]])
            elif len(pts) == 4:
                segments.append([pts[0], pts[1]])
                segments.append([pts[2], pts[3]])
    
    return _connect_segments(segments)


def _connect_segments(segments: List[List[Tuple[float, float]]],
                      tolerance: float = 1e-6) -> List[List[Tuple[float, float]]]:
    if not segments:
        return []
    
    connected = []
    used = [False] * len(segments)
    
    for start_idx in range(len(segments)):
        if used[start_idx]:
            continue
        
        current = list(segments[start_idx])
        used[start_idx] = True
        changed = True
        
        while changed:
            changed = False
            
            for i in range(len(segments)):
                if used[i]:
                    continue
                
                seg = segments[i]
                
                dist_start_start = _point_distance(current[0], seg[0])
                dist_start_end = _point_distance(current[0], seg[-1])
                dist_end_start = _point_distance(current[-1], seg[0])
                dist_end_end = _point_distance(current[-1], seg[-1])
                
                min_dist = min(dist_start_start, dist_start_end, dist_end_start, dist_end_end)
                
                if min_dist < tolerance:
                    used[i] = True
                    changed = True
                    
                    if dist_end_start == min_dist:
                        current.extend(seg[1:])
                    elif dist_end_end == min_dist:
                        current.extend(reversed(seg[:-1]))
                    elif dist_start_start == min_dist:
                        current = list(reversed(seg[:-1])) + current
                    elif dist_start_end == min_dist:
                        current = seg + current[1:]
        
        connected.append(current)
    
    return connected


def _point_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def grid_to_heatmap_overlay(z_grid: np.ndarray, grid_lon: np.ndarray, grid_lat: np.ndarray,
                            opacity: float = 0.6) -> Dict:
    rgb_grid = np.zeros((z_grid.shape[0], z_grid.shape[1], 4), dtype=np.uint8)
    
    for i in range(z_grid.shape[0]):
        for j in range(z_grid.shape[1]):
            val = z_grid[i, j]
            if np.isnan(val):
                rgb_grid[i, j] = [0, 0, 0, 0]
            else:
                r, g, b, _ = value_to_rgb(val)
                rgb_grid[i, j] = [r, g, b, int(255 * opacity)]
    
    bounds = [[float(grid_lat[-1]), float(grid_lon[0])],
              [float(grid_lat[0]), float(grid_lon[-1])]]
    
    return {
        'image_data': rgb_grid,
        'bounds': bounds,
        'x_labels': grid_lon,
        'y_labels': grid_lat[::-1]
    }


def generate_legend_items() -> List[Dict]:
    items = []
    for level in NOISE_LEVEL_RANGES:
        items.append({
            'color': level['color'],
            'label': level['label'],
            'min': level['min'],
            'max': level['max']
        })
    return items


def classify_grid_cells(z_grid: np.ndarray, grid_lon: np.ndarray, grid_lat: np.ndarray) -> np.ndarray:
    rows, cols = z_grid.shape
    classified = np.zeros((rows, cols), dtype=int)
    
    for idx, level in enumerate(NOISE_LEVEL_RANGES):
        mask = (z_grid >= level['min']) & (z_grid < level['max'])
        classified[mask] = idx
    
    classified[z_grid >= NOISE_LEVEL_RANGES[-1]['max']] = len(NOISE_LEVEL_RANGES) - 1
    classified[np.isnan(z_grid)] = -1
    
    return classified


def compute_area_statistics(z_grid: np.ndarray, grid_lon: np.ndarray, grid_lat: np.ndarray,
                            cell_resolution_m: float = 50.0) -> Dict:
    cell_area = cell_resolution_m ** 2
    
    stats = {}
    total_valid_cells = np.sum(~np.isnan(z_grid))
    total_area = total_valid_cells * cell_area / 1e6
    
    for level in NOISE_LEVEL_RANGES:
        if level['max'] == 200:
            mask = z_grid >= level['min']
        else:
            mask = (z_grid >= level['min']) & (z_grid < level['max'])
        
        count = np.sum(mask & ~np.isnan(z_grid))
        area_km2 = count * cell_area / 1e6
        percentage = (count / total_valid_cells * 100) if total_valid_cells > 0 else 0
        
        stats[level['label']] = {
            'cell_count': int(count),
            'area_km2': float(area_km2),
            'percentage': float(percentage)
        }
    
    exceed_mask = z_grid > 70
    exceed_count = np.sum(exceed_mask & ~np.isnan(z_grid))
    exceed_area = exceed_count * cell_area / 1e6
    
    stats['超标(>70dB)'] = {
        'cell_count': int(exceed_count),
        'area_km2': float(exceed_area),
        'percentage': float((exceed_count / total_valid_cells * 100) if total_valid_cells > 0 else 0)
    }
    
    stats['_total_area_km2'] = float(total_area)
    stats['_total_cells'] = int(total_valid_cells)
    
    return stats
