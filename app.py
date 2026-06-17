import streamlit as st
import pandas as pd
import numpy as np
import json
import folium
from streamlit_folium import st_folium
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from data_models import (
    get_all_stations, get_stations_by_region, get_station_measurements,
    get_latest_measurements, get_data_statistics, get_functional_zones,
    add_functional_zone, clear_functional_zones, ZONE_STANDARDS, FREQUENCY_BANDS,
    save_noise_prediction, get_noise_predictions, delete_prediction
)
from data_import import import_csv_data, get_import_template, MIN_SOUND_LEVEL, MAX_SOUND_LEVEL
from spatial_interpolation import run_interpolation, VARIOGRAM_MODELS, VARIOGRAM_NAMES
from contour_utils import (
    value_to_hex, get_noise_level_label, marching_squares,
    generate_legend_items, compute_area_statistics, NOISE_LEVEL_RANGES
)
from zone_analysis import (
    process_uploaded_zones, evaluate_zone_compliance, zones_to_geojson_features,
    generate_compliance_summary, get_overall_compliance_stats
)
from time_analysis import get_station_time_analysis, calculate_hourly_pattern
from spectrum_analysis import get_station_source_analysis, generate_source_recommendations
from noise_prediction import predict_road_traffic_noise, generate_prediction_contour
from report_generator import generate_report_pdf

st.set_page_config(
    page_title="城市噪声地图分析系统",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded"
)

APP_TITLE = "城市噪声地图分析系统"
APP_SUBTITLE = "基于多源监测数据的噪声时空分析与预测平台"


def init_session_state():
    defaults = {
        'interpolation_result': None,
        'contour_data': None,
        'functional_zones': [],
        'zone_evaluation': None,
        'compliance_time_period': 'day',
        'selected_station': None,
        'map_center': [39.9042, 116.4074],
        'map_zoom': 12,
        'heatmap_opacity': 0.6,
        'show_contours': True,
        'contour_interval': 5.0,
        'show_zones': True,
        'show_stations': True,
        'show_predictions': True,
        'interpolation_method': 'idw',
        'idw_power': 2.0,
        'idw_max_points': 20,
        'variogram_model': 'spherical',
        'grid_resolution': 50.0,
        'interpolation_done': False,
        'variogram_params': None,
        'selected_time_filter': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


def main():
    st.markdown(f"# {APP_TITLE}")
    st.markdown(f"*{APP_SUBTITLE}*")
    
    st.sidebar.markdown("## 📋 功能导航")
    page = st.sidebar.radio(
        "选择功能模块",
        [
            "📊 数据概览",
            "📥 数据导入",
            "🗺️ 噪声地图",
            "🏙️ 功能区评价",
            "📈 时间分析",
            "🔊 声源识别",
            "🛣️ 噪声预测",
            "📄 统计报告"
        ]
    )
    
    st.sidebar.markdown("---")
    render_sidebar_station_tree()
    
    if page.startswith("📊"):
        page_overview()
    elif page.startswith("📥"):
        page_data_import()
    elif page.startswith("🗺️"):
        page_noise_map()
    elif page.startswith("🏙️"):
        page_zone_evaluation()
    elif page.startswith("📈"):
        page_time_analysis()
    elif page.startswith("🔊"):
        page_source_identification()
    elif page.startswith("🛣️"):
        page_noise_prediction()
    elif page.startswith("📄"):
        page_report()


def render_sidebar_station_tree():
    st.sidebar.markdown("### 📍 监测站点")
    
    stations_by_region = get_stations_by_region()
    
    if not stations_by_region:
        st.sidebar.info("暂无监测站点数据，请先导入CSV数据。")
        return
    
    regions = list(stations_by_region.keys())
    
    for region in sorted(regions):
        stations = stations_by_region[region]
        with st.sidebar.expander(f"📁 {region} ({len(stations)}站)", expanded=True):
            for station in stations:
                sid = station['station_id']
                sname = station.get('station_name', sid)
                
                if st.button(
                    f"📍 {sname}",
                    key=f"sidebar_station_{sid}",
                    use_container_width=True
                ):
                    st.session_state.selected_station = sid
                    st.session_state.map_center = [station['latitude'], station['longitude']]
                    st.session_state.map_zoom = 14
                    st.rerun()


def page_overview():
    st.header("📊 数据概览")
    
    stats = get_data_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("监测站点数", f"{stats['station_count']} 个")
    with col2:
        st.metric("有效监测记录", f"{stats['valid_measurements']:,} 条")
    with col3:
        st.metric("数据有效率", f"{stats['data_efficiency']:.1f} %")
    with col4:
        st.metric("监测覆盖天数", f"{stats['days_count']} 天")
    
    if stats['time_start'] and stats['time_end']:
        st.info(f"**监测时段**: {stats['time_start']} 至 {stats['time_end']}")
    
    st.markdown("---")
    
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("监测站点分布")
        
        stations_df = get_all_stations()
        latest_df = get_latest_measurements()
        
        if not stations_df.empty:
            center_lat = stations_df['latitude'].mean()
            center_lon = stations_df['longitude'].mean()
            
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=11,
                tiles='OpenStreetMap'
            )
            
            if not latest_df.empty:
                for _, row in latest_df.iterrows():
                    leq = row.get('leq', 0)
                    color = value_to_hex(leq)
                    popup_html = f"""
                    <b>{row.get('station_name', row['station_id'])}</b><br>
                    站点编号: {row['station_id']}<br>
                    区域: {row.get('region', '未知')}<br>
                    Leq: {leq:.1f} dB(A)<br>
                    时间: {row.get('measurement_time', 'N/A')}
                    """
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=8,
                        popup=folium.Popup(popup_html, max_width=300),
                        color='white',
                        weight=2,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.9
                    ).add_to(m)
            else:
                for _, row in stations_df.iterrows():
                    popup_html = f"""
                    <b>{row.get('station_name', row['station_id'])}</b><br>
                    站点编号: {row['station_id']}<br>
                    区域: {row.get('region', '未知')}
                    """
                    folium.Marker(
                        location=[row['latitude'], row['longitude']],
                        popup=folium.Popup(popup_html, max_width=300),
                        icon=folium.Icon(color='blue', icon='info-sign')
                    ).add_to(m)
            
            st_folium(m, width='100%', height=450, returned_objects=[])
        else:
            st.info("暂无站点数据")
    
    with col_right:
        st.subheader("站点Leq分布")
        
        latest_df = get_latest_measurements()
        if not latest_df.empty and 'leq' in latest_df.columns:
            fig, ax = plt.subplots(figsize=(6, 5))
            
            bins = [30, 40, 45, 50, 55, 60, 65, 70, 80, 90]
            n, patches = ax.hist(latest_df['leq'], bins=bins, edgecolor='white',
                                linewidth=0.8, alpha=0.9)
            
            for i, patch in enumerate(patches):
                bin_mid = (bins[i] + bins[i+1]) / 2
                patch.set_facecolor(value_to_hex(bin_mid))
            
            ax.axvline(x=55, color='#FF5722', linestyle='--', linewidth=1.5,
                      alpha=0.8, label='推荐限值 (55dB)')
            
            ax.set_xlabel('Leq [dB(A)]', fontsize=11)
            ax.set_ylabel('站点数量', fontsize=11)
            ax.set_title('各站点最新Leq分布直方图', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(axis='y', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("暂无监测数据")
    
    if st.session_state.selected_station:
        st.markdown("---")
        st.subheader(f"📍 当前选中站点: {st.session_state.selected_station}")
        
        station_df = get_station_measurements(st.session_state.selected_station)
        
        if not station_df.empty:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Leq 均值", f"{station_df['leq'].mean():.1f} dB")
            with col2:
                st.metric("Leq 最大值", f"{station_df['leq'].max():.1f} dB")
            with col3:
                st.metric("Leq 最小值", f"{station_df['leq'].min():.1f} dB")
            with col4:
                st.metric("记录数", f"{len(station_df)} 条")
            
            if len(station_df) > 5:
                st.subheader("Leq时序变化")
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot(station_df['measurement_time'], station_df['leq'],
                       'b-', alpha=0.7, linewidth=1)
                sc = ax.scatter(station_df['measurement_time'], station_df['leq'],
                          s=15, alpha=0.6, c=station_df['leq'], cmap='coolwarm', vmin=40, vmax=80)
                plt.colorbar(sc, ax=ax, label='Leq [dB]')
                ax.axhline(y=station_df['leq'].mean(), color='r', linestyle='--',
                          linewidth=1, alpha=0.7, label=f'均值: {station_df["leq"].mean():.1f}dB')
                ax.set_xlabel('时间', fontsize=11)
                ax.set_ylabel('Leq [dB(A)]', fontsize=11)
                ax.legend()
                ax.grid(alpha=0.3)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.info("该站点暂无监测数据")


def page_data_import():
    st.header("📥 数据导入")
    
    st.markdown("### 📝 数据格式说明")
    st.info("""
    **必需字段**: 站点编号、经度、纬度、监测时间、等效连续声级Leq(dB(A))  
    **可选字段**: 站点名称、区域、最大声级Lmax、统计声级L10/L50/L90、  
    1/3倍频程频谱数据(25Hz~12500Hz共28个频段，列名如"25Hz"或"freq_25")
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 📥 下载导入模板")
        template_bytes = get_import_template()
        st.download_button(
            label="⬇️ 下载CSV模板",
            data=template_bytes,
            file_name="噪声监测数据_模板.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        st.markdown("#### ✅ 数值校验规则")
        st.markdown(f"""
        - 声级数值范围: **{MIN_SOUND_LEVEL}~{MAX_SOUND_LEVEL} dB(A)**
        - 经度范围: **-180° ~ 180°**
        - 纬度范围: **-90° ~ 90°**
        - 监测时间: 支持多种格式(YYYY-MM-DD HH、YYYY/MM/DD HH:MM等)
        """)
    
    st.markdown("---")
    st.markdown("### 📤 上传CSV数据")
    
    uploaded_file = st.file_uploader("选择CSV文件", type=['csv'])
    
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_size = len(file_bytes) / 1024
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.info(f"文件名: {uploaded_file.name}  \n文件大小: {file_size:.1f} KB")
        
        with col2:
            if st.button("🚀 开始导入数据", type="primary", use_container_width=True):
                with st.spinner("正在解析和导入数据..."):
                    result = import_csv_data(file_bytes)
                
                if result['success']:
                    st.success(result['message'])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("总行数", result['total_rows'])
                    with col2:
                        st.metric("✅ 成功导入", result['imported_rows'])
                    with col3:
                        st.metric("❌ 错误行数", result['error_rows'])
                    
                    if result['warnings']:
                        with st.expander(f"⚠️ {len(result['warnings'])} 条数据警告(超出阈值)", expanded=False):
                            for w in result['warnings'][:50]:
                                st.warning(w)
                            if len(result['warnings']) > 50:
                                st.info(f"... 还有 {len(result['warnings']) - 50} 条警告未显示")
                    
                    if result['errors']:
                        with st.expander(f"❌ {len(result['errors'])} 条数据错误(未导入)", expanded=False):
                            for err in result['errors'][:20]:
                                st.error(f"第{err.get('row', '?')}行 - 站点: {err.get('station_id', 'N/A')}")
                                for e in err.get('errors', []):
                                    st.write(f"  • {e}")
                            if len(result['errors']) > 20:
                                st.info(f"... 还有 {len(result['errors']) - 20} 条错误未显示")
                else:
                    st.error(result['message'])
    
    st.markdown("---")
    st.markdown("### 📋 已导入数据预览")
    
    stations_df = get_all_stations()
    latest_df = get_latest_measurements()
    
    if not latest_df.empty:
        display_cols = [c for c in ['station_id', 'station_name', 'region', 'longitude',
                                     'latitude', 'measurement_time', 'leq', 'lmax',
                                     'l10', 'l50', 'l90', 'zone_type']
                       if c in latest_df.columns]
        st.dataframe(
            latest_df[display_cols].style.applymap(
                lambda x: 'background-color: #ffebee; color: #c62828; font-weight: bold'
                if isinstance(x, (int, float)) and (x < MIN_SOUND_LEVEL or x > MAX_SOUND_LEVEL) else ''
            ),
            use_container_width=True,
            height=350
        )
    else:
        st.info("暂无已导入数据")


def render_interpolation_controls():
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ 插值设置")
    
    method = st.sidebar.selectbox(
        "插值方法",
        options=['idw', 'kriging', 'natural_neighbor'],
        format_func=lambda x: {
            'idw': '反距离加权 (IDW)',
            'kriging': '克里金 (Kriging)',
            'natural_neighbor': '自然邻域'
        }[x],
        index=['idw', 'kriging', 'natural_neighbor'].index(st.session_state.interpolation_method),
        key='ctrl_method'
    )
    st.session_state.interpolation_method = method
    
    st.session_state.grid_resolution = st.sidebar.slider(
        "网格分辨率(m)",
        min_value=20, max_value=200, value=int(st.session_state.grid_resolution),
        step=10, key='ctrl_res'
    )
    
    if method == 'idw':
        st.session_state.idw_power = st.sidebar.slider(
            "IDW幂次参数",
            min_value=1.0, max_value=5.0, value=st.session_state.idw_power,
            step=0.1, key='ctrl_power'
        )
        st.session_state.idw_max_points = st.sidebar.slider(
            "最大邻点数",
            min_value=4, max_value=50, value=st.session_state.idw_max_points,
            step=2, key='ctrl_maxpts'
        )
    elif method == 'kriging':
        st.session_state.variogram_model = st.sidebar.selectbox(
            "半变异函数模型",
            options=['spherical', 'exponential', 'gaussian'],
            format_func=lambda x: VARIOGRAM_NAMES.get(x, x),
            index=['spherical', 'exponential', 'gaussian'].index(st.session_state.variogram_model),
            key='ctrl_vario'
        )
    
    st.sidebar.markdown("#### 🎨 显示设置")
    st.session_state.heatmap_opacity = st.sidebar.slider(
        "热力图透明度",
        min_value=0.1, max_value=1.0, value=st.session_state.heatmap_opacity,
        step=0.05, key='ctrl_opacity'
    )
    st.session_state.show_contours = st.sidebar.checkbox(
        "显示等值线", value=st.session_state.show_contours, key='ctrl_contours'
    )
    if st.session_state.show_contours:
        st.session_state.contour_interval = st.sidebar.slider(
            "等值线间隔(dB)",
            min_value=1.0, max_value=10.0, value=st.session_state.contour_interval,
            step=0.5, key='ctrl_contour_int'
        )
    st.session_state.show_stations = st.sidebar.checkbox(
        "显示监测站点", value=st.session_state.show_stations, key='ctrl_showsta'
    )


def perform_interpolation():
    latest_df = get_latest_measurements()
    
    if latest_df.empty:
        st.error("没有可用的监测数据用于插值")
        return None
    
    coords = latest_df[['longitude', 'latitude']].values
    values = latest_df['leq'].values
    
    method = st.session_state.interpolation_method
    kwargs = {}
    
    if method == 'idw':
        kwargs['power'] = st.session_state.idw_power
        kwargs['max_points'] = st.session_state.idw_max_points
    elif method == 'kriging':
        kwargs['variogram_model'] = st.session_state.variogram_model
    
    with st.spinner(f"正在执行{method.upper()}插值..."):
        result = run_interpolation(
            sample_coords=coords,
            sample_values=values,
            method=method,
            resolution=st.session_state.grid_resolution,
            **kwargs
        )
    
    return result


def page_noise_map():
    st.header("🗺️ 噪声地图")
    
    render_interpolation_controls()
    
    latest_df = get_latest_measurements()
    
    if latest_df.empty:
        st.warning("暂无监测数据，请先导入CSV数据后查看噪声地图。")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("参与插值站点数", f"{len(latest_df)}")
    with col2:
        st.metric("Leq均值", f"{latest_df['leq'].mean():.1f} dB")
    with col3:
        st.metric("Leq最小值", f"{latest_df['leq'].min():.1f} dB")
    with col4:
        st.metric("Leq最大值", f"{latest_df['leq'].max():.1f} dB")
    
    col_ctrl1, col_ctrl2, _ = st.columns([1, 1, 3])
    with col_ctrl1:
        run_btn = st.button("🔄 重新生成插值", type="primary", use_container_width=True)
    with col_ctrl2:
        if st.button("🗺️ 只看地图", use_container_width=True):
            pass
    
    if run_btn or st.session_state.interpolation_result is None:
        result = perform_interpolation()
        if result:
            st.session_state.interpolation_result = result
            st.session_state.interpolation_done = True
            st.session_state.variogram_params = result.get('variogram_params')
            
            with st.spinner("生成等值线..."):
                contours = marching_squares(
                    result['z'], result['grid_lon'], result['grid_lat'],
                    interval=st.session_state.contour_interval
                )
                st.session_state.contour_data = contours
    
    result = st.session_state.interpolation_result
    if not result:
        return
    
    st.info(f"**插值方法**: {result['method'].upper()} | "
           f"**网格尺寸**: {result['shape'][1]}×{result['shape'][0]} ({result['shape'][0]*result['shape'][1]:,}个点) | "
           f"**耗时**: {result['time']:.2f}秒")
    
    if result['method'] == 'kriging' and st.session_state.variogram_params:
        vp = st.session_state.variogram_params
        st.info(f"**半变异函数**: {vp.get('model', 'spherical')} | "
               f"块金值: {vp.get('nugget', 0):.2f} | "
               f"基台值: {vp.get('sill', 0):.2f} | "
               f"变程: {vp.get('range', 0):.0f}m")
    
    st.markdown("---")
    
    m = create_noise_map(result, latest_df)
    
    st_folium(m, width='100%', height=600, returned_objects=[])
    
    st.markdown("---")
    
    render_legend_and_stats(result)


def create_noise_map(interp_result: Dict, stations_df: pd.DataFrame) -> folium.Map:
    z = interp_result['z']
    grid_lon = interp_result['grid_lon']
    grid_lat = interp_result['grid_lat']
    bounds = interp_result['bounds']
    
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    from PIL import Image as PILImage
    import io
    
    ny, nx = z.shape
    rgb_img = np.zeros((ny, nx, 4), dtype=np.uint8)
    
    opacity = st.session_state.heatmap_opacity
    
    from contour_utils import value_to_rgb
    
    for i in range(ny):
        for j in range(nx):
            val = z[i, j]
            if np.isnan(val):
                continue
            r, g, b, _ = value_to_rgb(val)
            rgb_img[i, j] = [r, g, b, int(255 * opacity)]
    
    rgb_img = np.flipud(rgb_img)
    
    pil_img = PILImage.fromarray(rgb_img, 'RGBA')
    img_bytes = io.BytesIO()
    pil_img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    img_bounds = [[grid_lat[-1], grid_lon[0]], [grid_lat[0], grid_lon[-1]]]
    
    folium.raster_layers.ImageOverlay(
        image=img_bytes,
        bounds=img_bounds,
        opacity=1.0,
        interactive=False,
        cross_origin=False,
        zindex=1
    ).add_to(m)
    
    if st.session_state.show_contours and st.session_state.contour_data:
        for contour in st.session_state.contour_data:
            level = contour['level']
            for line in contour['lines']:
                if len(line) >= 2:
                    latlons = [(lat, lon) for lon, lat in line]
                    color = value_to_hex(level)
                    folium.PolyLine(
                        latlons,
                        color=color,
                        weight=1.5,
                        opacity=0.9,
                        tooltip=f'{level:.0f} dB'
                    ).add_to(m)
    
    if st.session_state.show_stations:
        for _, row in stations_df.iterrows():
            leq = row.get('leq', 55)
            color = value_to_hex(leq)
            
            popup_content = f"""
            <div style="font-size:13px;">
                <b>{row.get('station_name', row['station_id'])}</b><br/>
                编号: {row['station_id']}<br/>
                区域: {row.get('region', '未知')}<br/>
                Leq: <b style="color:{color};">{leq:.1f} dB(A)</b><br/>
                等级: {get_noise_level_label(leq)}<br/>
                时间: {row.get('measurement_time', 'N/A')}
            </div>
            """
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=9,
                popup=folium.Popup(popup_content, max_width=300),
                color='white',
                weight=2.5,
                fill=True,
                fill_color=color,
                fill_opacity=1.0
            ).add_to(m)
    
    zones = st.session_state.get('functional_zones', [])
    if st.session_state.get('show_zones', True) and zones:
        zone_evaluation = st.session_state.get('zone_evaluation')
        zone_features = zones_to_geojson_features(zones, zone_evaluation, show_non_compliant=True)
        
        if zone_features:
            folium.GeoJson(
                {"type": "FeatureCollection", "features": zone_features},
                style_function=lambda feature: feature['properties'].get('style', {}),
                tooltip=folium.GeoJsonTooltip(
                    fields=['zone_name', 'zone_type_name', 'compliance_rate', 'standard'],
                    aliases=['名称:', '类别:', '达标率(%):', '标准:'],
                    localize=True
                )
            ).add_to(m)
    
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    
    return m


def render_legend_and_stats(interp_result: Dict):
    col_legend, col_stats = st.columns([1, 2])
    
    with col_legend:
        st.markdown("#### 🎨 噪声等级图例")
        legend_items = generate_legend_items()
        for item in legend_items:
            col_c, col_l = st.columns([1, 5])
            with col_c:
                st.markdown(
                    f"<div style='background:{item['color']};height:22px;width:100%;"
                    f"border-radius:4px;'></div>",
                    unsafe_allow_html=True
                )
            with col_l:
                st.caption(item['label'])
    
    with col_stats:
        st.markdown("#### 📊 面积分布统计")
        z = interp_result['z']
        grid_lon = interp_result['grid_lon']
        grid_lat = interp_result['grid_lat']
        res = interp_result.get('resolution', st.session_state.grid_resolution)
        
        area_stats = compute_area_statistics(z, grid_lon, grid_lat, res)
        total_area = area_stats.get('_total_area_km2', 0)
        
        stat_rows = []
        for level in NOISE_LEVEL_RANGES:
            s = area_stats.get(level['label'], {})
            stat_rows.append({
                '噪声等级': level['label'],
                '面积(km²)': f"{s.get('area_km2', 0):.3f}",
                '占比(%)': f"{s.get('percentage', 0):.1f}"
            })
        
        stat_df = pd.DataFrame(stat_rows)
        st.dataframe(stat_df, use_container_width=True, hide_index=True)
        
        exceed_key = '超标(>70dB)'
        if exceed_key in area_stats and area_stats[exceed_key]['area_km2'] > 0:
            s = area_stats[exceed_key]
            st.error(f"⚠️ **超标区域 (>70dB)**: {s['area_km2']:.3f} km², 占总面积的 {s['percentage']:.1f}%")


def page_zone_evaluation():
    st.header("🏙️ 功能区划分与达标评价")
    
    st.markdown("### 📚 国标GB3096-2008声环境功能区限值")
    
    limits_data = []
    for zt, info in ZONE_STANDARDS.items():
        limits_data.append({
            '功能区类别': info['name'],
            '昼间限值(dB)': info['day'],
            '夜间限值(dB)': info['night'],
            '适用区域': info['description']
        })
    st.table(pd.DataFrame(limits_data))
    
    st.markdown("---")
    st.markdown("### 📤 上传功能区划分数据")
    st.info("请上传GeoJSON格式文件，每个多边形要素需包含 zone_type 属性(0-4)")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        zone_file = st.file_uploader("选择GeoJSON文件", type=['geojson', 'json'])
    with col2:
        st.session_state.compliance_time_period = st.radio(
            "评价时段",
            options=['day', 'night'],
            format_func=lambda x: {'day': '☀️ 昼间(6:00-22:00)', 'night': '🌙 夜间(22:00-6:00)'}[x],
            horizontal=True
        )
    
    col3, col4, _ = st.columns([1, 1, 3])
    with col3:
        import_btn = st.button("📥 导入功能区数据", type="primary", use_container_width=True)
    with col4:
        clear_btn = st.button("🗑️ 清除功能区", use_container_width=True)
    
    if zone_file is not None and import_btn:
        try:
            file_bytes = zone_file.getvalue()
            zones, warnings = process_uploaded_zones(file_bytes)
            
            clear_functional_zones()
            for z in zones:
                add_functional_zone(z['zone_name'], z['zone_type'], json.dumps(z['geometry']))
            
            st.session_state.functional_zones = zones
            st.success(f"✅ 成功导入 {len(zones)} 个功能区")
            if warnings:
                for w in warnings:
                    st.warning(w)
        except Exception as e:
            st.error(f"导入失败: {e}")
    
    if clear_btn:
        clear_functional_zones()
        st.session_state.functional_zones = []
        st.session_state.zone_evaluation = None
        st.success("已清除所有功能区数据")
    
    if not st.session_state.functional_zones:
        db_zones = get_functional_zones()
        if db_zones:
            loaded_zones = []
            for z in db_zones:
                try:
                    geo_obj = json.loads(z['geojson'])
                    from shapely.geometry import shape
                    loaded_zones.append({
                        'zone_name': z['zone_name'],
                        'zone_type': z['zone_type'],
                        'geometry': geo_obj,
                        'shape': shape(geo_obj)
                    })
                except Exception:
                    continue
            st.session_state.functional_zones = loaded_zones
    
    zones = st.session_state.functional_zones
    
    if not zones:
        st.info("请先上传功能区划分GeoJSON数据")
        return
    
    st.markdown("---")
    
    col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
    with col_sum1:
        st.metric("功能区总数", f"{len(zones)} 个")
    zone_types = set(z['zone_type'] for z in zones)
    with col_sum2:
        st.metric("涉及类别", f"{len(zone_types)} 类")
    
    result = st.session_state.interpolation_result
    if result is None:
        st.warning("请先在「噪声地图」页面完成空间插值")
        return
    
    if st.button("🔍 开始达标评价计算", type="primary"):
        with st.spinner("正在计算功能区达标情况..."):
            eval_results = evaluate_zone_compliance(
                zones=zones,
                z_grid=result['z'],
                grid_lon=result['grid_lon'],
                grid_lat=result['grid_lat'],
                time_period=st.session_state.compliance_time_period,
                cell_resolution_m=result.get('resolution', 50)
            )
        st.session_state.zone_evaluation = eval_results
        st.success("达标评价计算完成！")
    
    eval_results = st.session_state.zone_evaluation
    
    if eval_results:
        overall = get_overall_compliance_stats(eval_results)
        
        with col_sum3:
            st.metric("总体达标率", f"{overall['overall_compliance_rate']:.1f} %")
        with col_sum4:
            st.metric("超标功能区数", f"{overall['non_compliant_zone_count']} 个")
        
        st.markdown("---")
        st.subheader("📋 达标评价汇总表")
        summary_df = generate_compliance_summary(eval_results)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.subheader("📊 各功能区达标率对比")
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            zones_plot = [r['zone_name'] for r in eval_results]
            rates_plot = [r['compliance_rate'] for r in eval_results]
            bar_colors = ['#4CAF50' if r >= 95 else '#FFC107' if r >= 80 else '#F44336'
                         for r in rates_plot]
            
            bars = ax1.barh(range(len(zones_plot)), rates_plot, color=bar_colors, edgecolor='white')
            ax1.set_yticks(range(len(zones_plot)))
            ax1.set_yticklabels(zones_plot, fontsize=9)
            ax1.set_xlabel('达标率 (%)', fontsize=11)
            ax1.set_xlim(0, 110)
            ax1.axvline(x=90, color='#FF5722', linestyle='--', linewidth=1.2, label='90%目标')
            ax1.legend(fontsize=9)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            
            for i, (bar, rate) in enumerate(zip(bars, rates_plot)):
                ax1.text(rate + 1, i, f'{rate:.1f}%', va='center', fontsize=9, fontweight='bold')
            
            fig1.tight_layout()
            st.pyplot(fig1)
            plt.close(fig1)
        
        with col_chart2:
            st.subheader("📈 超标幅度统计")
            exceed_data = []
            for r in eval_results:
                if r['non_compliant_cells'] > 0:
                    exceed_data.append({
                        'name': r['zone_name'],
                        '超标面积': r['non_compliant_area_km2'],
                        '最大超标': r['max_exceedance']
                    })
            
            if exceed_data:
                fig2, ax2 = plt.subplots(figsize=(8, 5))
                edf = pd.DataFrame(exceed_data)
                x = range(len(edf))
                w = 0.35
                ax2.bar([i - w/2 for i in x], edf['超标面积'], w,
                       label='超标面积(km²)', color='#EF5350', alpha=0.85)
                ax2_twin = ax2.twinx()
                ax2_twin.bar([i + w/2 for i in x], edf['最大超标'], w,
                            label='最大超标(dB)', color='#FFA726', alpha=0.85)
                
                ax2.set_xticks(x)
                ax2.set_xticklabels(edf['name'], rotation=30, ha='right', fontsize=9)
                ax2.set_ylabel('超标面积 (km²)', fontsize=10, color='#EF5350')
                ax2_twin.set_ylabel('最大超标量 (dB)', fontsize=10, color='#FFA726')
                ax2.set_title('超标功能区统计', fontsize=12, fontweight='bold')
                
                lines1, labels1 = ax2.get_legend_handles_labels()
                lines2, labels2 = ax2_twin.get_legend_handles_labels()
                ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper right')
                
                fig2.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)
            else:
                st.success("🎉 所有功能区均达标！")
    
    st.markdown("---")
    st.subheader("🗺️ 功能区空间分布")
    
    m = folium.Map(
        location=[(result['bounds'][1] + result['bounds'][3])/2,
                 (result['bounds'][0] + result['bounds'][2])/2],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    zone_features = zones_to_geojson_features(zones, eval_results, show_non_compliant=True)
    folium.GeoJson(
        {"type": "FeatureCollection", "features": zone_features},
        style_function=lambda feature: feature['properties'].get('style', {}),
        tooltip=folium.GeoJsonTooltip(
            fields=['zone_name', 'zone_type_name', 'day_standard', 'night_standard',
                    'compliance_rate', 'avg_value'],
            aliases=['名称:', '类别:', '昼间限:', '夜间限:', '达标率:', '均值Leq:'],
            localize=True
        )
    ).add_to(m)
    
    st_folium(m, width='100%', height=500, returned_objects=[])


def page_time_analysis():
    st.header("📈 时间维度分析")
    
    stations_df = get_all_stations()
    
    if stations_df.empty:
        st.warning("暂无站点数据，请先导入数据")
        return
    
    col1, col2 = st.columns([1, 3])
    with col1:
        station_options = stations_df.apply(
            lambda x: f"{x.get('station_name', x['station_id'])} ({x['station_id']})",
            axis=1
        ).tolist()
        
        default_idx = 0
        if st.session_state.selected_station:
            for i, (_, row) in enumerate(stations_df.iterrows()):
                if row['station_id'] == st.session_state.selected_station:
                    default_idx = i
                    break
        
        selected = st.selectbox("选择分析站点", station_options, index=default_idx)
        selected_station_id = stations_df.iloc[station_options.index(selected)]['station_id']
        st.session_state.selected_station = selected_station_id
    
    measurements_df = get_station_measurements(selected_station_id)
    
    if measurements_df.empty:
        st.info("该站点暂无监测数据")
        return
    
    with st.spinner("正在进行时间维度分析..."):
        analysis = get_station_time_analysis(measurements_df, selected_station_id)
    
    st.markdown("---")
    st.subheader("📊 基础统计与昼夜等效声级")
    
    bs = analysis['basic_stats']
    ldn_info = analysis['ldn']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("监测记录数", bs['total_records'])
    with col2:
        st.metric("监测时段", bs['date_range'])
    with col3:
        st.metric("Leq均值", f"{bs['leq_mean']:.1f} dB")
    with col4:
        if ldn_info.get('ldn'):
            st.metric("Ldn昼夜等效", f"{ldn_info['ldn']:.1f} dB")
    
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        if ldn_info.get('ld'):
            st.metric("☀️ 昼间Ld", f"{ldn_info['ld']:.1f} dB", f"{ldn_info['day_count']}条")
    with col6:
        if ldn_info.get('ln'):
            st.metric("🌙 夜间Ln", f"{ldn_info['ln']:.1f} dB", f"{ldn_info['night_count']}条")
    with col7:
        st.metric("Leq最大", f"{bs['leq_max']:.1f} dB")
    with col8:
        st.metric("Leq标准差", f"{bs['leq_std']:.1f} dB")
    
    st.markdown("---")
    
    weekly = analysis['weekly']
    hourly, peak_hours = analysis['hourly'], analysis['peak_hours']
    monthly, monthly_trend = analysis['monthly'], analysis['monthly_trend']
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("📅 一周变化模式")
        if weekly is not None and not weekly.empty:
            fig_w, ax_w = plt.subplots(figsize=(8, 4.5))
            weekdays = weekly['weekday_name'].values
            means = weekly['mean'].values
            colors_w = ['#2196F3'] * 5 + ['#4CAF50'] * 2
            colors_w = colors_w[:len(weekdays)]
            
            bars = ax_w.bar(range(len(weekdays)), means, color=colors_w,
                          edgecolor='white', linewidth=0.5, alpha=0.9)
            ax_w.set_xticks(range(len(weekdays)))
            ax_w.set_xticklabels(weekdays, fontsize=10)
            ax_w.set_ylabel('平均 Leq [dB(A)]', fontsize=11)
            ax_w.set_title('各工作日平均Leq对比', fontsize=12, fontweight='bold')
            
            overall_m = np.mean(means)
            ax_w.axhline(y=overall_m, color='#FF5722', linestyle='--', linewidth=1.2,
                        alpha=0.8, label=f'周均值 {overall_m:.1f}dB')
            ax_w.legend(fontsize=9)
            
            for bar, m_val in zip(bars, means):
                ax_w.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                         f'{m_val:.1f}', ha='center', fontsize=9, fontweight='bold')
            
            ax_w.spines['top'].set_visible(False)
            ax_w.spines['right'].set_visible(False)
            ax_w.grid(axis='y', alpha=0.3)
            fig_w.tight_layout()
            st.pyplot(fig_w)
            plt.close(fig_w)
        else:
            st.info("数据不足以分析周变化")
    
    with col_b:
        st.subheader("⏰ 24小时变化曲线")
        if hourly is not None and not hourly.empty:
            fig_h, ax_h = plt.subplots(figsize=(8, 4.5))
            hours = hourly['hour'].values
            means = hourly['mean'].values
            
            ax_h.plot(hours, means, 'o-', color='#1976D2', linewidth=2.2, markersize=6,
                     markerfacecolor='white', markeredgewidth=2, label='平均Leq')
            
            if peak_hours:
                peak_h_arr = hours[np.isin(hours, peak_hours)]
                peak_m_arr = means[np.isin(hours, peak_hours)]
                ax_h.scatter(peak_h_arr, peak_m_arr, s=150, color='#F44336', marker='*', zorder=5,
                            label=f'高峰时段 ({len(peak_hours)}h)')
            
            daily_mean_col = hourly.get('daily_mean', pd.Series([np.mean(means)] * len(hourly)))
            threshold_col = hourly.get('threshold', pd.Series([np.mean(means) + 5] * len(hourly)))
            daily_mean = float(daily_mean_col.iloc[0])
            threshold = float(threshold_col.iloc[0])
            
            ax_h.axhline(y=daily_mean, color='#4CAF50', linestyle='--', linewidth=1,
                        alpha=0.8, label=f'日均值 {daily_mean:.1f}dB')
            ax_h.axhline(y=threshold, color='#FF9800', linestyle=':', linewidth=1,
                        alpha=0.8, label=f'阈值 {threshold:.1f}dB')
            
            ax_h.fill_between(hours, means, alpha=0.12, color='#1976D2')
            ax_h.set_xticks(range(0, 24, 2))
            ax_h.set_xlabel('时刻 (小时)', fontsize=11)
            ax_h.set_ylabel('平均 Leq [dB(A)]', fontsize=11)
            ax_h.set_title('24小时噪声变化规律', fontsize=12, fontweight='bold')
            ax_h.legend(fontsize=9, loc='upper left')
            ax_h.spines['top'].set_visible(False)
            ax_h.spines['right'].set_visible(False)
            ax_h.grid(axis='y', alpha=0.3)
            fig_h.tight_layout()
            st.pyplot(fig_h)
            plt.close(fig_h)
        else:
            st.info("数据不足以分析24小时变化")
    
    st.markdown("---")
    st.subheader("📉 月度趋势分析")
    
    if monthly is not None and not monthly.empty:
        fig_m, ax_m = plt.subplots(figsize=(10, 4.5))
        labels = monthly['year_month_str'].values
        means_m = monthly['mean'].values
        x = np.arange(len(labels))
        
        ax_m.plot(x, means_m, 'o-', color='#7B1FA2', linewidth=2, markersize=7,
                 markerfacecolor='white', markeredgewidth=2, label='月均Leq')
        
        if monthly_trend.get('predicted') and len(monthly_trend['predicted']) == len(x):
            tc = monthly_trend.get('trend_color', '#666')
            tt = monthly_trend.get('trend', '')
            ax_m.plot(x, monthly_trend['predicted'], '--', color=tc, linewidth=1.8,
                     label=f'趋势线 ({tt})')
        
        ax_m.set_xticks(x)
        ax_m.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax_m.set_ylabel('月均 Leq [dB(A)]', fontsize=11)
        ax_m.set_title('月度噪声变化趋势', fontsize=12, fontweight='bold')
        ax_m.legend(fontsize=9)
        ax_m.grid(axis='y', alpha=0.3)
        ax_m.spines['top'].set_visible(False)
        ax_m.spines['right'].set_visible(False)
        
        if 'monthly_change_dB' in monthly_trend:
            change = monthly_trend['monthly_change_dB']
            if abs(change) >= 0.05:
                direction = '恶化(上升)' if change > 0 else '改善(下降)'
                tc = monthly_trend.get('trend_color', '#666')
                ax_m.text(0.02, 0.98, f'月变化率: {change:+.2f} dB/月\n{direction}',
                         transform=ax_m.transAxes, fontsize=10, va='top',
                         bbox=dict(boxstyle='round,pad=0.5', facecolor=tc, alpha=0.25))
        
        fig_m.tight_layout()
        st.pyplot(fig_m)
        plt.close(fig_m)
        
        t_info = monthly_trend.get('trend', 'N/A')
        r2 = monthly_trend.get('r_squared', 0)
        st.caption(f"趋势判定: **{t_info}** | 线性回归R²: {r2:.3f}")
    else:
        st.info("数据不足以分析月度趋势(至少需要3个月数据)")


def page_source_identification():
    st.header("🔊 声源识别与频谱分析")
    
    stations_df = get_all_stations()
    
    if stations_df.empty:
        st.warning("暂无站点数据")
        return
    
    station_list = []
    for _, row in stations_df.iterrows():
        measurements = get_station_measurements(row['station_id'])
        if not measurements.empty:
            station_list.append((row['station_id'], row.get('station_name', row['station_id'])))
    
    if not station_list:
        st.warning("暂无有效监测数据")
        return
    
    col1, col2 = st.columns([1, 2])
    with col1:
        multi_select = st.multiselect(
            "选择分析站点 (可多选)",
            options=[sid for sid, _ in station_list],
            format_func=lambda x: next((name for sid, name in station_list if sid == x), x),
            default=[station_list[0][0]]
        )
    
    st.markdown("---")
    
    all_sources = []
    icon_map = {'traffic': '🚗', 'construction': '🏗️', 'industrial': '🏭', 'life': '👥', 'unknown': '❓'}
    color_map = {'traffic': '#FF9800', 'construction': '#9C27B0', 'industrial': '#607D8B',
                'life': '#4CAF50', 'unknown': '#9E9E9E'}
    
    for idx, station_id in enumerate(multi_select):
        station_name = next((name for sid, name in station_list if sid == station_id), station_id)
        
        measurements_df = get_station_measurements(station_id)
        if measurements_df.empty:
            continue
        
        try:
            hourly_df, _ = calculate_hourly_pattern(measurements_df)
        except Exception:
            hourly_df = None
        
        with st.spinner(f"正在分析 {station_name} 的声源特征..."):
            try:
                src_result = get_station_source_analysis(measurements_df, hourly_df)
            except Exception as e:
                st.warning(f"{station_name} 分析出错: {e}")
                continue
        
        if not src_result:
            continue
        
        src_id = src_result.get('source_identification', {})
        spectrum = src_result.get('spectrum', {})
        temporal = src_result.get('temporal_features', {})
        
        all_sources.append({
            'station_id': station_id,
            'station_name': station_name,
            'source': src_id,
            'avg_leq': temporal.get('avg_leq', 0),
            'spectrum_values': spectrum.get('values', [])
        })
        
        with st.container():
            st.subheader(f"📍 {station_name} ({station_id})")
            
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            
            primary = src_id.get('primary_source', 'unknown')
            primary_name = src_id.get('primary_source_name', '未知')
            conf = src_id.get('confidence', 0)
            
            with col_s1:
                st.markdown(
                    f"<div style='background:{color_map.get(primary, '#999')};"
                    f"color:white;padding:12px;border-radius:8px;text-align:center;'>"
                    f"<div style='font-size:28px;'>{icon_map.get(primary, '❓')}</div>"
                    f"<div style='font-size:14px;font-weight:bold;'>{primary_name}</div>"
                    f"<div style='font-size:11px;opacity:0.9;'>置信度 {conf:.0f}%</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            
            features = src_id.get('features', {})
            with col_s2:
                diff_val = features.get('l10_l90_diff')
                st.metric("L10-L90差值", f"{diff_val} dB" if diff_val else "N/A",
                         help="反映噪声波动性")
            with col_s3:
                ml_diff = features.get('lmax_leq_diff')
                st.metric("Lmax-Leq差值", f"{ml_diff} dB" if ml_diff else "N/A",
                         help="反映脉冲特性")
            with col_s4:
                st.metric("频谱峰值数", f"{features.get('peak_count', 0)} 个",
                         help="特征频率尖峰数量")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            col_scores, col_spectrum = st.columns([1, 2])
            
            with col_scores:
                st.markdown("**各类型匹配得分**")
                ranked = src_id.get('ranked_sources', [])
                max_score = max([r['score'] for r in ranked]) if ranked else 1
                for r in ranked:
                    pct = min(100, r['score'] / max_score * 100) if max_score > 0 else 0
                    st.markdown(
                        f"<div style='margin-bottom:8px;'>"
                        f"<div style='display:flex;justify-content:space-between;font-size:12px;margin-bottom:3px;'>"
                        f"<span>{r['icon']} {r['name']}</span>"
                        f"<b>{r['score']:.0f}分</b></div>"
                        f"<div style='background:#eee;border-radius:4px;height:8px;overflow:hidden;'>"
                        f"<div style='background:{r['color']};height:100%;width:{pct:.0f}%;'></div>"
                        f"</div></div>",
                        unsafe_allow_html=True
                    )
            
            with col_spectrum:
                st.markdown("**1/3倍频程频谱特征**")
                spec_vals = spectrum.get('values', [])
                freqs = FREQUENCY_BANDS
                
                if spec_vals and any(v is not None for v in spec_vals):
                    fig_spec, ax_spec = plt.subplots(figsize=(9, 3.8))
                    
                    valid_x = []
                    valid_y = []
                    for f, v in zip(freqs, spec_vals):
                        if v is not None and not (isinstance(v, float) and np.isnan(v)):
                            valid_x.append(f)
                            valid_y.append(float(v))
                    
                    if valid_x:
                        ax_spec.semilogx(valid_x, valid_y, 'o-', color='#1976D2',
                                        linewidth=1.8, markersize=4, alpha=0.9, base=10)
                        ax_spec.fill_between(valid_x, valid_y, alpha=0.15, color='#1976D2')
                        
                        ax_spec.axvspan(63, 500, alpha=0.12, color='#FF9800', label='交通特征带(63-500Hz)')
                        ax_spec.axvspan(250, 1000, alpha=0.1, color='#9C27B0', label='施工特征带(250-1000Hz)')
                        ax_spec.axvspan(1000, 4000, alpha=0.08, color='#4CAF50', label='生活特征带(1-4kHz)')
                        
                        ax_spec.set_xlabel('1/3倍频程中心频率 (Hz)', fontsize=10)
                        ax_spec.set_ylabel('声压级 [dB]', fontsize=10)
                        ax_spec.set_title(f'{station_name} 平均频谱', fontsize=11, fontweight='bold')
                        ax_spec.grid(True, which='both', alpha=0.3, linestyle='-')
                        ax_spec.legend(fontsize=8, loc='lower left')
                        ax_spec.spines['top'].set_visible(False)
                        ax_spec.spines['right'].set_visible(False)
                        fig_spec.tight_layout()
                        st.pyplot(fig_spec)
                        plt.close(fig_spec)
                    
                    centroid = spectrum.get('centroid')
                    rolloff = spectrum.get('rolloff_85')
                    if centroid:
                        st.caption(f"频谱质心: {centroid:.0f} Hz | 谱滚降(85%): {rolloff} Hz")
                else:
                    st.info("该站点暂无频谱数据")
            
            with st.expander("💡 改善建议"):
                recommendations = generate_source_recommendations(primary)
                for i, rec in enumerate(recommendations, 1):
                    st.markdown(f"{i}. {rec}")
        
        if idx < len(multi_select) - 1:
            st.markdown("---")
    
    if all_sources:
        st.markdown("---")
        st.subheader("📊 多站点声源类型分布")
        
        type_counts = {}
        for s in all_sources:
            pt = s['source'].get('primary_source', 'unknown')
            type_counts[pt] = type_counts.get(pt, 0) + 1
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if type_counts:
                fig_pie, ax_pie = plt.subplots(figsize=(6, 5))
                actual_labels = []
                actual_values = []
                actual_colors = []
                for src_type, count in type_counts.items():
                    label_parts = [
                        f"{icon_map.get(src_type, '❓')} {s['source'].get('primary_source_name', src_type)}"
                        for s in all_sources
                        if s['source'].get('primary_source', 'unknown') == src_type
                    ]
                    unique_label = list(dict.fromkeys(label_parts))
                    actual_labels.append(unique_label[0] if unique_label else src_type)
                    actual_values.append(count)
                    actual_colors.append(color_map.get(src_type, '#999'))
                
                wedges, texts, autotexts = ax_pie.pie(
                    actual_values, labels=actual_labels, colors=actual_colors,
                    autopct='%1.0f%%', startangle=90, pctdistance=0.78,
                    textprops={'fontsize': 10}
                )
                for t in autotexts:
                    t.set_fontweight('bold')
                ax_pie.set_title('主要噪声源分布', fontsize=12, fontweight='bold', pad=15)
                centre_circle = plt.Circle((0, 0), 0.62, fc='white')
                ax_pie.add_artist(centre_circle)
                ax_pie.text(0, 0, f'Total\n{len(all_sources)}站', ha='center', va='center',
                           fontsize=11, fontweight='bold')
                fig_pie.tight_layout()
                st.pyplot(fig_pie)
                plt.close(fig_pie)


def page_noise_prediction():
    st.header("🛣️ 道路交通噪声预测")
    
    st.markdown("### 📐 FHWA简化预测模型")
    st.info("""
    **预测公式**: Leq = 基准声级 + 10·lg(Q/D) + 修正项(车速/路面/重车/屏障)  
    **声屏障**: 采用Maekawa公式计算插入损失，考虑绕射衰减
    """)
    
    col_form1, col_form2 = st.columns(2)
    
    with col_form1:
        st.markdown("#### 🛣️ 道路与交通参数")
        pred_name = st.text_input("预测方案名称", value="道路噪声预测_1")
        
        st.markdown("**道路起终点坐标**")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            start_lon = st.number_input("起点经度", value=116.4000, step=0.001, format="%.5f")
            start_lat = st.number_input("起点纬度", value=39.9000, step=0.001, format="%.5f")
        with col_r2:
            end_lon = st.number_input("终点经度", value=116.4200, step=0.001, format="%.5f")
            end_lat = st.number_input("终点纬度", value=39.9100, step=0.001, format="%.5f")
        
        traffic_q = st.number_input("车流量 (辆/小时)", value=800, min_value=1, step=50)
        col_sp1, col_sp2 = st.columns(2)
        with col_sp1:
            avg_speed = st.slider("平均车速 (km/h)", 20, 120, 60, step=5)
        with col_sp2:
            hv_ratio = st.slider("重型车比例", 0.0, 0.5, 0.10, step=0.01)
        road_surface = st.selectbox("路面类型", options=['asphalt', 'concrete'],
                                   format_func=lambda x: {'asphalt': '🛣️ 沥青路面', 'concrete': '🧱 水泥路面'}[x])
    
    with col_form2:
        st.markdown("#### 📍 预测点与屏障")
        pred_lon = st.number_input("预测点经度", value=116.4100, step=0.001, format="%.5f")
        pred_lat = st.number_input("预测点纬度", value=39.9050, step=0.001, format="%.5f")
        
        st.markdown("**声屏障(可选)**")
        has_barrier = st.checkbox("设置声屏障", value=False)
        barrier_h = 3.0
        barrier_pos_ratio = 0.5
        barrier_len = 50.0
        if has_barrier:
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                barrier_h = st.slider("屏障高度 (m)", 1.0, 10.0, 3.0, step=0.5)
                barrier_pos_ratio = st.slider("屏障位置比", 0.1, 0.9, 0.5, step=0.05,
                                             help="距道路侧占总距离的比例")
            with col_b2:
                barrier_len = st.slider("屏障长度 (m)", 10, 500, 50, step=10)
    
    col_calc, col_save, _ = st.columns([1, 1, 3])
    with col_calc:
        calc_btn = st.button("🔍 计算预测值", type="primary", use_container_width=True)
    with col_save:
        show_contour_pred = st.checkbox("生成等值线区域", value=True)
    
    if calc_btn:
        with st.spinner("正在执行噪声预测计算..."):
            pred_result = predict_road_traffic_noise(
                road_start_lon=start_lon, road_start_lat=start_lat,
                road_end_lon=end_lon, road_end_lat=end_lat,
                prediction_lon=pred_lon, prediction_lat=pred_lat,
                traffic_volume=traffic_q, avg_speed=avg_speed,
                heavy_vehicle_ratio=hv_ratio, road_surface=road_surface,
                barrier_height=barrier_h if has_barrier else None,
                barrier_position_ratio=barrier_pos_ratio,
                barrier_length=barrier_len
            )
        
        if pred_result:
            st.markdown("---")
            st.subheader("📊 预测结果")
            
            col_pr1, col_pr2, col_pr3, col_pr4 = st.columns(4)
            with col_pr1:
                st.metric("预测Leq", f"{pred_result['predicted_leq']:.1f} dB(A)")
            with col_pr2:
                st.metric("距路距离", f"{pred_result['distance_to_road']:.1f} m")
            with col_pr3:
                st.metric("基准声级", f"{pred_result['reference_leq']:.1f} dB")
            with col_pr4:
                if pred_result['barrier_insertion_loss'] > 0:
                    st.metric("屏障插入损失", f"{pred_result['barrier_insertion_loss']:.1f} dB")
                else:
                    st.metric("屏障状态", "无声屏障")
            
            st.markdown("")
            
            col_d1, col_d2, col_d3, col_d4 = st.columns(4)
            with col_d1:
                st.caption(f"距离衰减: **{pred_result['distance_correction']:+.1f} dB**")
            with col_d2:
                st.caption(f"车速修正: **{pred_result['speed_correction']:+.1f} dB**")
            with col_d3:
                st.caption(f"路面修正: **{pred_result['surface_correction']:+.1f} dB**")
            with col_d4:
                st.caption(f"重车修正: **{pred_result['heavy_vehicle_correction']:+.1f} dB**")
            
            save_data = {
                'name': pred_name,
                'road_start_lon': start_lon,
                'road_start_lat': start_lat,
                'road_end_lon': end_lon,
                'road_end_lat': end_lat,
                'traffic_volume': traffic_q,
                'avg_speed': avg_speed,
                'heavy_vehicle_ratio': hv_ratio,
                'road_surface': road_surface,
                'barrier_height': barrier_h if has_barrier else None,
                'barrier_position': barrier_pos_ratio,
                'prediction_lon': pred_lon,
                'prediction_lat': pred_lat,
                'distance': pred_result['distance_to_road'],
                'predicted_leq': pred_result['predicted_leq'],
                'inserted_loss': pred_result['barrier_insertion_loss']
            }
            save_noise_prediction(save_data)
            
            if show_contour_pred:
                with st.spinner("生成预测等值线..."):
                    pred_contour = generate_prediction_contour(
                        start_lon, start_lat, end_lon, end_lat,
                        traffic_q, avg_speed, hv_ratio, road_surface,
                        barrier_h if has_barrier else None,
                        barrier_pos_ratio
                    )
                
                if pred_contour:
                    st.markdown("---")
                    st.subheader("🗺️ 预测结果空间分布")
                    
                    m_pred = folium.Map(
                        location=[(start_lat + end_lat + pred_lat) / 3,
                                 (start_lon + end_lon + pred_lon) / 3],
                        zoom_start=14,
                        tiles='OpenStreetMap'
                    )
                    
                    folium.PolyLine(
                        [(start_lat, start_lon), (end_lat, end_lon)],
                        color='#D32F2F', weight=5, opacity=0.9,
                        tooltip=f'道路: {traffic_q}辆/h, {avg_speed}km/h'
                    ).add_to(m_pred)
                    
                    from contour_utils import value_to_hex
                    z = pred_contour['z']
                    glon = pred_contour['grid_lon']
                    glat = pred_contour['grid_lat']
                    
                    levels = list(range(40, 85, 5))
                    for level in levels:
                        level_contours = marching_squares(z, glon, glat, levels=[level])
                        for lc in level_contours:
                            for line in lc['lines']:
                                if len(line) >= 2:
                                    latlons = [(lat, lon) for lon, lat in line]
                                    folium.PolyLine(
                                        latlons,
                                        color=value_to_hex(level),
                                        weight=1.5,
                                        opacity=0.85,
                                        dash_array='5, 5',
                                        tooltip=f'预测 {level} dB'
                                    ).add_to(m_pred)
                    
                    folium.Marker(
                        location=[pred_lat, pred_lon],
                        popup=f'预测点<br>Leq={pred_result["predicted_leq"]:.1f}dB<br>D={pred_result["distance_to_road"]:.1f}m',
                        icon=folium.Icon(color='blue', icon='flag')
                    ).add_to(m_pred)
                    
                    m_pred.fit_bounds([[start_lat, start_lon], [end_lat, end_lon],
                                       [pred_lat, pred_lon]])
                    st_folium(m_pred, width='100%', height=500, returned_objects=[])
    
    st.markdown("---")
    st.subheader("📋 历史预测记录")
    preds = get_noise_predictions()
    if not preds.empty:
        display_preds = preds[[
            'name', 'road_surface', 'traffic_volume', 'avg_speed',
            'heavy_vehicle_ratio', 'distance', 'predicted_leq',
            'inserted_loss', 'created_at'
        ]].copy()
        display_preds.columns = [
            '方案名称', '路面类型', '车流量', '车速(km/h)',
            '重车比例', '距离(m)', '预测Leq(dB)', '屏障IL(dB)', '创建时间'
        ]
        st.dataframe(display_preds, use_container_width=True, height=220)
    else:
        st.info("暂无历史预测记录")


def page_report():
    st.header("📄 统计报告生成")
    
    st.markdown("### 📑 报告内容概览")
    
    report_items = [
        ("监测概况", "站点数量、监测时段、数据有效率"),
        ("功能区达标评价", "各功能区达标率表格、柱状图和超标统计"),
        ("噪声空间分布", "热力图和等值线图"),
        ("时间变化分析", "周变化、24小时变化、月度趋势图"),
        ("声源识别分析", "各站点主要噪声源类型和置信度"),
        ("改善建议", "针对超标区域和主要声源的控制措施")
    ]
    
    for title, desc in report_items:
        st.markdown(f"- **{title}**: {desc}")
    
    st.markdown("---")
    st.markdown("### ⚙️ 报告生成设置")
    
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        stations_df = get_all_stations()
        if not stations_df.empty:
            station_ids = stations_df['station_id'].tolist()
            report_stations = st.multiselect(
                "选择包含在报告中的站点 (默认全选)",
                options=station_ids,
                default=station_ids
            )
        else:
            st.warning("暂无站点数据")
            report_stations = []
    with col_sel2:
        include_prediction = st.checkbox("包含预测数据(如有)", value=True)
    
    gen_btn = st.button("📄 一键生成PDF评估报告", type="primary", use_container_width=True)
    
    if gen_btn:
        if not report_stations:
            st.error("请至少选择一个站点")
            return
        
        with st.spinner("正在生成PDF报告，请稍候..."):
            latest_df = get_latest_measurements()
            
            compliance_df = pd.DataFrame()
            eval_results = st.session_state.get('zone_evaluation')
            if eval_results:
                compliance_df = generate_compliance_summary(eval_results)
            
            z_grid = None
            grid_lon = None
            grid_lat = None
            interp = st.session_state.interpolation_result
            if interp:
                z_grid = interp['z']
                grid_lon = interp['grid_lon']
                grid_lat = interp['grid_lat']
            
            weekly_df = None
            hourly_df = None
            monthly_df = None
            monthly_trend = {}
            source_analysis = []
            
            if report_stations:
                first_station = report_stations[0]
                m_df = get_station_measurements(first_station)
                if not m_df.empty:
                    analysis = get_station_time_analysis(m_df, first_station)
                    weekly_df = analysis.get('weekly')
                    hourly_df = analysis.get('hourly')
                    monthly_df = analysis.get('monthly')
                    monthly_trend = analysis.get('monthly_trend', {})
                
                for sid in report_stations[:5]:
                    m_df_s = get_station_measurements(sid)
                    if not m_df_s.empty:
                        try:
                            h_df, _ = calculate_hourly_pattern(m_df_s)
                        except Exception:
                            h_df = None
                        src = get_station_source_analysis(m_df_s, h_df)
                        if src:
                            source_analysis.append({
                                'station_id': sid,
                                'station_name': stations_df[stations_df['station_id']==sid]['station_name'].iloc[0]
                                if 'station_name' in stations_df.columns else sid,
                                'source': src.get('source_identification', {}),
                                'avg_leq': src.get('temporal_features', {}).get('avg_leq', 0)
                            })
            
            area_stats = {}
            if interp:
                area_stats = compute_area_statistics(
                    z_grid, grid_lon, grid_lat,
                    interp.get('resolution', 50)
                )
            
            recommendations = None
            
            try:
                pdf_bytes = generate_report_pdf(
                    compliance_df=compliance_df,
                    area_stats=area_stats,
                    z_grid=z_grid,
                    grid_lon=grid_lon,
                    grid_lat=grid_lat,
                    weekly_df=weekly_df,
                    hourly_df=hourly_df,
                    monthly_df=monthly_df,
                    trend_info=monthly_trend,
                    source_analysis=source_analysis,
                    recommendations=recommendations
                )
                
                st.success("✅ 报告生成成功！")
                
                col_dl1, col_dl2, _ = st.columns([1, 1, 3])
                with col_dl1:
                    report_date = datetime.now().strftime("%Y%m%d_%H%M")
                    st.download_button(
                        label="⬇️ 下载PDF报告",
                        data=pdf_bytes,
                        file_name=f"声环境质量评估报告_{report_date}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                with col_dl2:
                    st.metric("报告大小", f"{len(pdf_bytes)/1024:.1f} KB")
                
                st.markdown("---")
                st.subheader("👀 报告内容预览")
                
                if not compliance_df.empty:
                    st.markdown("#### 功能区达标评价")
                    st.dataframe(compliance_df, use_container_width=True, hide_index=True)
                
                if source_analysis:
                    st.markdown("#### 主要噪声源识别")
                    for sa in source_analysis:
                        src_name = sa['source'].get('primary_source_name', '未知')
                        conf = sa['source'].get('confidence', 0)
                        st.caption(f"{sa['station_name']}: **{src_name}** (置信度 {conf:.0f}%)")
                        
            except Exception as e:
                st.error(f"报告生成失败: {e}")
                import traceback
                st.code(traceback.format_exc())


if __name__ == '__main__':
    main()