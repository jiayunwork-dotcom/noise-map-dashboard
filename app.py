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
    save_noise_prediction, get_noise_predictions, delete_prediction,
    get_measurement_time_range
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
from time_analysis import get_station_time_analysis, calculate_hourly_pattern, detect_noise_events, get_event_statistics
from spectrum_analysis import get_station_source_analysis, generate_source_recommendations
from noise_prediction import predict_road_traffic_noise, generate_prediction_contour
from report_generator import generate_report_pdf, generate_comparison_report_pdf
from correlation_analysis import (
    compute_station_distance_matrix, detect_events_for_stations,
    match_cooperative_events, estimate_source_location,
    export_traceability_geojson, haversine_distance, SOUND_SPEED
)
from alert_manager import (
    load_alert_rules, save_alert_rules, add_alert_rule, delete_alert_rule,
    load_alert_history, save_alert_history, run_alert_engine_all_stations,
    get_alert_statistics, get_alerts_by_station, has_active_alerts,
    evaluate_alert_rule, ALERT_LEVELS, METRIC_NAMES, COMPARE_NAMES,
    _count_noise_events, _parse_datetime, filter_alerts_by_date,
    RULE_TEMPLATES, detect_rule_conflicts, apply_rule_template,
    enable_all_rules, disable_all_rules, export_rules_to_json
)

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
        'selected_time_filter': None,
        'event_threshold': 10.0,
        'coop_selected_stations': [],
        'coop_spectrum_threshold': 0.7,
        'coop_time_tolerance': 2.0,
        'coop_result': None,
        'coop_locations': None,
        'coop_compare_groups': [],
        'coop_selected_group': None,
        'coop_pdf_data': None,
        'coop_pdf_error': None,
        'alert_editing_rule': None,
        'alert_selected_level': '全部',
        'alert_selected_station': '全部',
        'alert_date_range': None,
        'alert_expanded_row': None,
        'alert_subtab': 'rules',
        'alert_coop_cache': {},
        'alert_rule_form_key': 0,
        'alert_date_range_initialized': False,
        'alert_default_date_range': None,
        'alert_conflict_results': None,
        'alert_template_applied_msg': None
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
            "🎯 协同溯源",
            "🔔 告警管理",
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
    elif page.startswith("🎯"):
        page_cooperative_tracing()
    elif page.startswith("🔔"):
        page_alert_management()
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
            n, _, patches = ax.hist(latest_df['leq'], bins=bins, edgecolor='white',
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
            latest_df[display_cols].style.map(
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

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🚨 事件检测设置")
    st.session_state.event_threshold = st.sidebar.slider(
        "事件阈值 (dB)",
        min_value=5, max_value=20, value=int(st.session_state.event_threshold),
        step=1, key='ctrl_event_threshold',
        help="Leq相对背景升高超过此阈值即标记为噪声事件"
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
    
    try:
        from PIL import Image as PILImage
        import io
        import base64
        
        ny, nx = z.shape
        rgb_img = np.zeros((ny, nx, 4), dtype=np.uint8)
        
        opacity = st.session_state.heatmap_opacity
        
        from contour_utils import value_to_rgb
        
        valid_count = 0
        for i in range(ny):
            for j in range(nx):
                val = z[i, j]
                if np.isnan(val):
                    continue
                r, g, b, _ = value_to_rgb(val)
                rgb_img[i, j] = [r, g, b, int(255 * opacity)]
                valid_count += 1
        
        if valid_count == 0:
            st.warning("⚠️ 没有有效的插值数据可用于渲染热力图")
        else:
            rgb_img = np.flipud(rgb_img)
            
            max_size = 2000
            if ny > max_size or nx > max_size:
                scale = min(max_size / ny, max_size / nx)
                new_ny = max(2, int(ny * scale))
                new_nx = max(2, int(nx * scale))
                pil_img = PILImage.fromarray(rgb_img, 'RGBA')
                try:
                    resample_method = PILImage.Resampling.BILINEAR
                except AttributeError:
                    resample_method = PILImage.BILINEAR
                pil_img = pil_img.resize((new_nx, new_ny), resample_method)
                st.caption(f"ℹ️ 热力图已自动缩放至 {new_nx}×{new_ny} 以提高性能")
            else:
                pil_img = PILImage.fromarray(rgb_img, 'RGBA')
            
            img_bytes = io.BytesIO()
            pil_img.save(img_bytes, format='PNG', optimize=True)
            img_bytes.seek(0)
            
            img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
            img_data_uri = f'data:image/png;base64,{img_base64}'
            
            img_bounds = [[grid_lat[0], grid_lon[0]], [grid_lat[-1], grid_lon[-1]]]
            
            folium.raster_layers.ImageOverlay(
                image=img_data_uri,
                bounds=img_bounds,
                opacity=1.0,
                interactive=True,
                zindex=10
            ).add_to(m)
            
    except Exception as e:
        st.error(f"❌ 热力图渲染失败: {str(e)}")
        import traceback
        st.code(traceback.format_exc(), language="python")
    
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


def _render_noise_events_tab(measurements_df: pd.DataFrame, selected_station_id: str):
    st.markdown("### 🚨 噪声事件检测")
    st.caption(f"当前检测阈值: **{st.session_state.event_threshold} dB** (前后3小时滑动均值对比)")

    with st.spinner("正在检测噪声事件..."):
        events = detect_noise_events(
            measurements_df,
            threshold_db=float(st.session_state.event_threshold),
            window_hours=3
        )
        event_stats = get_event_statistics(events)

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.metric("总事件数", f"{event_stats['total_events']} 次")
    with col_s2:
        avg_dur = event_stats['avg_duration_minutes']
        st.metric("平均持续时长", f"{avg_dur:.0f} 分钟" if avg_dur >= 60 else f"{avg_dur:.1f} 分钟")
    with col_s3:
        st.metric("最大峰值声级", f"{event_stats['max_peak_leq']:.1f} dB")
    with col_s4:
        mfh = event_stats['most_frequent_hour']
        if mfh is not None:
            st.metric("最常发生时段", f"{mfh:02d}:00 时段")
        else:
            st.metric("最常发生时段", "N/A")

    st.markdown("---")

    if not events:
        st.info("未检测到符合阈值的噪声事件，可尝试降低检测阈值。")
        return

    st.markdown("#### 📊 事件散点图")
    fig_e, ax_e = plt.subplots(figsize=(12, 5))

    event_times = [e['start_time'] for e in events]
    peak_leqs = [e['peak_leq'] for e in events]
    durations = np.array([e['duration_minutes'] for e in events])
    amplitudes = np.array([e['amplitude_db'] for e in events])

    sizes = np.clip(durations * 2, 30, 500)

    sc = ax_e.scatter(
        event_times, peak_leqs,
        s=sizes,
        c=amplitudes,
        cmap='YlOrRd',
        vmin=st.session_state.event_threshold,
        vmax=max(st.session_state.event_threshold + 5, amplitudes.max() if len(amplitudes) > 0 else st.session_state.event_threshold + 5),
        alpha=0.8,
        edgecolors='white',
        linewidths=0.8,
        zorder=3
    )

    cbar = plt.colorbar(sc, ax=ax_e, pad=0.01)
    cbar.set_label('超标幅度 (dB)', fontsize=10)

    ax_e.set_xlabel('时间', fontsize=11)
    ax_e.set_ylabel('峰值声级 Leq [dB(A)]', fontsize=11)
    ax_e.set_title(f'噪声事件分布 (共{len(events)}次)', fontsize=12, fontweight='bold')
    ax_e.grid(alpha=0.3, zorder=0)
    ax_e.spines['top'].set_visible(False)
    ax_e.spines['right'].set_visible(False)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFB74D',
               markersize=8, label='短时长事件', markeredgecolor='white'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E64A19',
               markersize=14, label='长时长事件', markeredgecolor='white')
    ]
    ax_e.legend(handles=legend_elements, loc='upper left', fontsize=9)

    fig_e.autofmt_xdate()
    fig_e.tight_layout()
    st.pyplot(fig_e)
    plt.close(fig_e)

    st.markdown("---")
    st.markdown(f"#### 📋 事件详情列表 (共 {len(events)} 条)")

    rows = []
    for idx, e in enumerate(events, 1):
        dur_h = int(e['duration_minutes'] // 60)
        dur_m = int(e['duration_minutes'] % 60)
        if dur_h > 0:
            dur_str = f"{dur_h}h{dur_m}min"
        else:
            dur_str = f"{dur_m}min"

        rows.append({
            '序号': idx,
            '开始时间': e['start_time'].strftime('%Y-%m-%d %H:%M'),
            '结束时间': e['end_time'].strftime('%Y-%m-%d %H:%M'),
            '峰值(dB)': e['peak_leq'],
            '背景(dB)': e['background_db'],
            '超标(dB)': e['amplitude_db'],
            '持续时长': dur_str,
            '推测来源': f"{e.get('icon', '❓')} {e.get('source_name', '未知')}"
        })

    events_df = pd.DataFrame(rows)

    def _highlight_amplitude(val):
        try:
            v = float(val)
            if v >= 20:
                return 'background-color: #ffebee; color: #c62828; font-weight: bold'
            elif v >= 15:
                return 'background-color: #fff3e0; color: #e65100; font-weight: bold'
            elif v >= 10:
                return 'background-color: #fffde7; color: #f57f17'
        except Exception:
            pass
        return ''

    styled = events_df.style.map(_highlight_amplitude, subset=['超标(dB)'])
    st.dataframe(styled, use_container_width=True, height=400, hide_index=True)

    with st.expander("🔍 查看各事件判定依据"):
        for idx, e in enumerate(events, 1):
            st.markdown(
                f"**事件{idx}**: {e['start_time'].strftime('%m-%d %H:%M')} ~ "
                f"{e['end_time'].strftime('%m-%d %H:%M')} | "
                f"<span style='color:{e.get('color', '#999')};font-weight:bold;'>"
                f"{e.get('icon', '❓')} {e.get('source_name', '未知')}</span>",
                unsafe_allow_html=True
            )
            st.caption(f"  判定依据: {e.get('reason', '特征不明显')}")
            st.markdown("")


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

    tab_pattern, tab_events = st.tabs(["📊 时序变化模式", "🚨 噪声事件检测"])

    with tab_pattern:
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

    with tab_events:
        _render_noise_events_tab(measurements_df, selected_station_id)


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


def page_alert_management():
    st.header("🔔 噪声事件告警与阈值管理")
    
    alert_tab_rules, alert_tab_history = st.tabs(["📋 告警规则配置", "📊 告警历史与统计"])
    
    with alert_tab_rules:
        _render_alert_rules_panel()
    
    with alert_tab_history:
        _render_alert_history_panel()


def _render_alert_rules_panel():
    st.subheader("告警规则配置")

    stations_df = get_all_stations()
    rules = load_alert_rules()

    rules_sorted = sorted(rules, key=lambda r: r.get('priority', 5), reverse=True)

    st.markdown("#### 📚 规则模板（批量创建）")
    col_tpl1, col_tpl2, col_tpl3, col_tpl4 = st.columns([1, 1, 1, 1.5])
    with col_tpl1:
        tpl_monitor = st.radio(
            "模板监控范围",
            options=['all', 'single'],
            format_func=lambda x: '全部站点' if x == 'all' else '单个站点',
            horizontal=True,
            key="tpl_monitor"
        )
    tpl_station_id = None
    if tpl_monitor == 'single':
        with col_tpl2:
            tpl_station_options = [(row['station_id'], f"{row.get('station_name', row['station_id'])} ({row['station_id']})")
                                   for _, row in stations_df.iterrows()]
            if tpl_station_options:
                tpl_station_id = st.selectbox(
                    "选择站点",
                    options=[s[0] for s in tpl_station_options],
                    format_func=lambda x: next((lbl for sid, lbl in tpl_station_options if sid == x), x),
                    key="tpl_station"
                )
    with col_tpl3:
        selected_template = st.selectbox(
            "选择模板",
            options=list(RULE_TEMPLATES.keys()),
            format_func=lambda k: RULE_TEMPLATES[k]['name'],
            key="tpl_selector"
        )
    with col_tpl4:
        st.caption(RULE_TEMPLATES[selected_template]['description'])
        if st.button("📋 应用模板", type="primary", use_container_width=True, key="apply_template_btn"):
            created = apply_rule_template(selected_template, tpl_monitor, tpl_station_id)
            if created:
                st.session_state.alert_template_applied_msg = f"✅ 成功创建 {len(created)} 条规则: {', '.join(r['rule_name'] for r in created)}"
                st.rerun()

    if st.session_state.alert_template_applied_msg:
        st.success(st.session_state.alert_template_applied_msg)
        st.session_state.alert_template_applied_msg = None

    st.markdown("---")

    st.markdown(f"#### 📋 规则列表 ({len(rules_sorted)}条)")
    col_batch1, col_batch2, col_batch3, _ = st.columns([1, 1, 1, 2])
    with col_batch1:
        if st.button("🟢 全部启用", use_container_width=True, key="enable_all_btn"):
            if enable_all_rules():
                st.success("已启用全部规则")
                st.rerun()
            else:
                st.error("操作失败")
    with col_batch2:
        if st.button("🔴 全部禁用", use_container_width=True, key="disable_all_btn"):
            if disable_all_rules():
                st.success("已禁用全部规则")
                st.rerun()
            else:
                st.error("操作失败")
    with col_batch3:
        export_json = export_rules_to_json()
        st.download_button(
            label="⬇️ 导出规则",
            data=export_json,
            file_name=f"alert_rules_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            use_container_width=True,
            key="export_rules_btn"
        )

    col_form, col_list = st.columns([1, 1.5])

    with col_form:
        st.markdown("#### ✏️ 新建/编辑规则")

        editing_rule = st.session_state.alert_editing_rule
        if st.session_state.alert_conflict_results is None:
            conflict_display_state = False
        else:
            conflict_display_state = True

        with st.form("alert_rule_form"):
            rule_name = st.text_input(
                "规则名称",
                value=editing_rule.get('rule_name', '') if editing_rule else '',
                placeholder="请输入规则名称"
            )

            monitor_target = st.radio(
                "监控对象",
                options=['all', 'single'],
                format_func=lambda x: '全部站点' if x == 'all' else '单个站点',
                index=0 if (not editing_rule or editing_rule.get('monitor_target') == 'all') else 1,
                horizontal=True
            )

            station_id = None
            if monitor_target == 'single':
                station_options = [(row['station_id'], f"{row.get('station_name', row['station_id'])} ({row['station_id']})")
                                   for _, row in stations_df.iterrows()]
                default_station = editing_rule.get('station_id') if editing_rule else (station_options[0][0] if station_options else None)
                station_id = st.selectbox(
                    "选择站点",
                    options=[s[0] for s in station_options],
                    format_func=lambda x: next((lbl for sid, lbl in station_options if sid == x), x),
                    index=0
                )

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                priority = st.slider(
                    "优先级 (1-10)",
                    min_value=1,
                    max_value=10,
                    value=int(editing_rule.get('priority', 5)) if editing_rule else 5,
                    step=1,
                    help="数字越大优先级越高，同站点同窗口多条同时命中时仅保留最高优先级的告警"
                )
            with col_p2:
                silent_period = st.number_input(
                    "静默期 (分钟)",
                    min_value=0,
                    max_value=1440,
                    value=int(editing_rule.get('silent_period', 0)) if editing_rule else 0,
                    step=5,
                    help="同一条规则针对同一站点触发后，在静默期内不重复生成告警，0表示不静默"
                )

            metric_type = st.selectbox(
                "触发指标",
                options=['leq_mean', 'leq_peak', 'event_frequency'],
                format_func=lambda x: METRIC_NAMES.get(x, x),
                index=['leq_mean', 'leq_peak', 'event_frequency'].index(
                    editing_rule.get('metric_type', 'leq_mean')
                ) if editing_rule else 0,
                key="alert_form_metric_type"
            )

            compare_options = ['greater_than', 'greater_equal']
            if metric_type != 'event_frequency':
                compare_options.append('continuous_minutes')

            prev_compare = editing_rule.get('compare_type', 'greater_than') if editing_rule else 'greater_than'
            if prev_compare not in compare_options:
                prev_compare = compare_options[0]

            compare_type = st.selectbox(
                "比较方式",
                options=compare_options,
                format_func=lambda x: COMPARE_NAMES.get(x, x),
                index=compare_options.index(prev_compare),
                key=f"alert_form_compare_{metric_type}"
            )

            threshold = st.number_input(
                "阈值数值",
                value=float(editing_rule.get('threshold', 60.0)) if editing_rule else 60.0,
                min_value=0.0,
                step=0.5,
                help="Leq单位为dB，事件频次单位为次"
            )

            continuous_minutes = 5
            if compare_type == 'continuous_minutes' and metric_type != 'event_frequency':
                continuous_minutes = st.slider(
                    "连续N分钟",
                    min_value=2,
                    max_value=60,
                    value=int(editing_rule.get('continuous_minutes', 5)) if editing_rule else 5,
                    step=1,
                    key=f"alert_form_continuous_{metric_type}"
                )

            alert_level = st.selectbox(
                "告警等级",
                options=['info', 'warning', 'critical'],
                format_func=lambda x: ALERT_LEVELS.get(x, {}).get('name', x),
                index=['info', 'warning', 'critical'].index(
                    editing_rule.get('alert_level', 'warning')
                ) if editing_rule else 1
            )

            time_period = st.radio(
                "生效时段",
                options=['all_day', 'custom'],
                format_func=lambda x: '全天' if x == 'all_day' else '自定义时段',
                index=0 if (not editing_rule or editing_rule.get('time_period') == 'all_day') else 1,
                horizontal=True
            )

            start_time = "08:00"
            end_time = "20:00"
            if time_period == 'custom':
                col_st, col_et = st.columns(2)
                with col_st:
                    start_time = st.text_input(
                        "开始时间",
                        value=editing_rule.get('start_time', '08:00') if editing_rule else '08:00',
                        placeholder="HH:MM"
                    )
                with col_et:
                    end_time = st.text_input(
                        "结束时间",
                        value=editing_rule.get('end_time', '20:00') if editing_rule else '20:00',
                        placeholder="HH:MM"
                    )

            enabled = st.toggle(
                "启用规则",
                value=editing_rule.get('enabled', True) if editing_rule else True
            )

            col_submit, col_cancel, col_check = st.columns([1.2, 1, 1])
            with col_submit:
                submit_label = "保存修改" if editing_rule else "➕ 新建规则"
                submitted = st.form_submit_button(submit_label, type="primary", use_container_width=True)
            with col_cancel:
                if editing_rule:
                    cancel = st.form_submit_button("取消编辑", use_container_width=True)
                    if cancel:
                        st.session_state.alert_editing_rule = None
                        st.session_state.alert_conflict_results = None
                        st.rerun()
            with col_check:
                conflict_check = st.form_submit_button("⚠️ 冲突检测", use_container_width=True)

            if conflict_check:
                temp_rule_data = {
                    'rule_name': rule_name,
                    'monitor_target': monitor_target,
                    'station_id': station_id,
                    'metric_type': metric_type,
                    'compare_type': compare_type,
                    'threshold': threshold,
                    'continuous_minutes': continuous_minutes,
                    'priority': priority
                }
                exclude_id = editing_rule.get('rule_id') if editing_rule else None
                conflicts = detect_rule_conflicts(temp_rule_data, exclude_id)
                st.session_state.alert_conflict_results = conflicts
                conflict_display_state = True

            if submitted:
                rule_data = {
                    'rule_name': rule_name,
                    'monitor_target': monitor_target,
                    'station_id': station_id,
                    'metric_type': metric_type,
                    'compare_type': compare_type,
                    'threshold': threshold,
                    'continuous_minutes': continuous_minutes,
                    'alert_level': alert_level,
                    'time_period': time_period,
                    'start_time': start_time,
                    'end_time': end_time,
                    'enabled': enabled,
                    'priority': priority,
                    'silent_period': silent_period
                }
                if editing_rule:
                    rule_data['rule_id'] = editing_rule['rule_id']

                if not rule_name.strip():
                    st.error("请输入规则名称")
                else:
                    result = add_alert_rule(rule_data)
                    if result:
                        st.success(f"规则已保存: {result['rule_name']}")
                        st.session_state.alert_editing_rule = None
                        st.session_state.alert_conflict_results = None
                        st.rerun()
                    else:
                        st.error("保存规则失败")

        if st.session_state.alert_conflict_results is not None:
            conflicts = st.session_state.alert_conflict_results
            if conflicts:
                st.markdown("##### ⚠️ 检测到规则冲突")
                for c in conflicts:
                    st.warning(
                        f"**冲突规则**: {c['rule_name']} (ID: {c['rule_id']})\n\n"
                        f"**原因**: {c['reason']}\n\n"
                        f"**优先级对比**: 新规则 {c['new_priority']} vs 现有规则 {c['existing_priority']}"
                    )
            else:
                st.success("✅ 未检测到冲突规则")

    with col_list:
        st.markdown("#### 规则详情（按优先级从高到低排序）")

        if not rules_sorted:
            st.info("暂无告警规则，请在左侧创建或应用模板")
        else:
            for i, rule in enumerate(rules_sorted):
                level_info = ALERT_LEVELS.get(rule.get('alert_level', 'info'), {})
                status_icon = "✅" if rule.get('enabled', True) else "⏸️"
                target_text = "全部站点" if rule.get('monitor_target') == 'all' else f"站点: {rule.get('station_id', '-')}"
                priority_val = rule.get('priority', 5)
                priority_stars = "⭐" * min(priority_val, 10)

                with st.expander(
                    f"{status_icon} **{rule['rule_name']}** [{priority_val}] {priority_stars} - {level_info.get('name', '')}",
                    expanded=False
                ):
                    col_info, col_actions = st.columns([3, 1])

                    with col_info:
                        st.markdown(f"**优先级**: {priority_val}/10")
                        st.markdown(f"**监控对象**: {target_text}")
                        st.markdown(f"**触发指标**: {METRIC_NAMES.get(rule.get('metric_type'), '-')}")
                        st.markdown(f"**比较方式**: {COMPARE_NAMES.get(rule.get('compare_type'), '-')}")
                        threshold_text = f"{rule.get('threshold', 0)}"
                        if rule.get('compare_type') == 'continuous_minutes':
                            threshold_text += f" (连续{rule.get('continuous_minutes', 5)}分钟)"
                        st.markdown(f"**阈值**: {threshold_text}")
                        period_text = "全天" if rule.get('time_period') == 'all_day' else f"{rule.get('start_time', '')} - {rule.get('end_time', '')}"
                        st.markdown(f"**生效时段**: {period_text}")
                        st.markdown(f"**告警等级**: :{level_info.get('color', '#666')}[{level_info.get('name', '')}]")
                        sp_text = f"{rule.get('silent_period', 0)} 分钟" if rule.get('silent_period', 0) > 0 else "不静默"
                        st.markdown(f"**静默期**: {sp_text}")

                    with col_actions:
                        if st.button("✏️ 编辑", key=f"edit_rule_{rule['rule_id']}", use_container_width=True):
                            st.session_state.alert_editing_rule = rule
                            st.session_state.alert_conflict_results = None
                            st.rerun()

                        if st.button("🗑️ 删除", key=f"del_rule_{rule['rule_id']}", use_container_width=True):
                            if delete_alert_rule(rule['rule_id']):
                                st.success("规则已删除")
                                st.rerun()
                            else:
                                st.error("删除失败")


def _render_alert_history_panel():
    st.subheader("告警历史记录")
    
    history = load_alert_history()
    stations_df = get_all_stations()
    stats = get_alert_statistics()
    
    st.markdown("### 📈 统计概览")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("今日告警数", stats['today_count'])
    with col2:
        st.metric("本周告警数", stats['week_count'])
    with col3:
        st.metric("历史总告警", stats['total_count'])
    with col4:
        st.metric("⬆️ 升级告警数", stats.get('upgraded_count', 0))
    
    col_pie, col_trend, col_top = st.columns([1, 1.5, 1])
    
    with col_pie:
        st.markdown("**各等级占比**")
        level_counts = stats['level_counts']
        fig_pie, ax_pie = plt.subplots(figsize=(4, 3))
        labels = [ALERT_LEVELS[k]['name'] for k in ['critical', 'warning', 'info'] if level_counts.get(k, 0) > 0]
        sizes = [level_counts.get(k, 0) for k in ['critical', 'warning', 'info'] if level_counts.get(k, 0) > 0]
        colors = [ALERT_LEVELS[k]['color'] for k in ['critical', 'warning', 'info'] if level_counts.get(k, 0) > 0]
        if sizes and sum(sizes) > 0:
            ax_pie.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%', startangle=90)
            ax_pie.set_aspect('equal')
        else:
            ax_pie.text(0.5, 0.5, '暂无数据', ha='center', va='center', transform=ax_pie.transAxes)
            ax_pie.axis('off')
        st.pyplot(fig_pie)
        plt.close(fig_pie)
    
    with col_trend:
        st.markdown("**最近7天告警趋势**")
        daily_data = stats['daily_trend']
        fig_trend, ax_trend = plt.subplots(figsize=(6, 3))
        dates = [d['date'][5:] for d in daily_data]
        counts = [d['count'] for d in daily_data]
        ax_trend.plot(dates, counts, marker='o', linewidth=2, color='#2196F3')
        ax_trend.fill_between(dates, counts, alpha=0.2, color='#2196F3')
        ax_trend.set_ylabel('告警数')
        ax_trend.grid(True, alpha=0.3)
        for spine in ax_trend.spines.values():
            spine.set_visible(False)
        st.pyplot(fig_trend)
        plt.close(fig_trend)
    
    with col_top:
        st.markdown("**告警最频繁Top3站点**")
        top_stations = stats['top_stations']
        if top_stations:
            for idx, (sid, count) in enumerate(top_stations):
                station_name = sid
                if not stations_df.empty:
                    row = stations_df[stations_df['station_id'] == sid]
                    if not row.empty:
                        station_name = row.iloc[0].get('station_name', sid)
                medal = ["🥇", "🥈", "🥉"][idx] if idx < 3 else "  "
                st.markdown(f"{medal} **{station_name}**: {count}次")
        else:
            st.info("暂无数据")
    
    st.markdown("---")
    st.markdown("### 🔍 筛选条件")
    
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        level_options = ['全部', '严重', '警告', '提示']
        level_map = {'全部': 'all', '严重': 'critical', '警告': 'warning', '提示': 'info'}
        selected_level_label = st.selectbox(
            "告警等级",
            options=level_options,
            index=level_options.index(st.session_state.alert_selected_level)
        )
        st.session_state.alert_selected_level = selected_level_label
        selected_level = level_map[selected_level_label]
    
    with col_f2:
        station_options = ['全部'] + stations_df['station_id'].tolist() if not stations_df.empty else ['全部']
        selected_station = st.selectbox(
            "选择站点",
            options=station_options,
            index=station_options.index(st.session_state.alert_selected_station) if st.session_state.alert_selected_station in station_options else 0
        )
        st.session_state.alert_selected_station = selected_station
    
    with col_f3:
        if 'alert_date_range_initialized' not in st.session_state or not st.session_state.alert_date_range_initialized:
            default_start = datetime.now() - timedelta(days=7)
            default_end = datetime.now()
            if history:
                alert_times = []
                for a in history:
                    dt = _parse_datetime(a.get('alert_time', ''))
                    if dt:
                        alert_times.append(dt)
                if alert_times:
                    min_t = min(alert_times).date()
                    max_t = max(alert_times).date()
                    default_start = min_t
                    default_end = max_t
            st.session_state.alert_default_date_range = (default_start, default_end)
            st.session_state.alert_date_range_initialized = True
        
        date_range = st.date_input(
            "时间范围",
            value=st.session_state.alert_default_date_range,
            key="alert_date_range_input"
        )
    
    filtered_history = history.copy()
    
    if selected_level != 'all':
        filtered_history = [a for a in filtered_history if a.get('alert_level') == selected_level]
    
    if selected_station != '全部':
        filtered_history = [a for a in filtered_history if a.get('station_id') == selected_station]
    
    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        filtered_history = filter_alerts_by_date(filtered_history, start_date, end_date)
    
    filtered_history.sort(key=lambda x: x['alert_time'], reverse=True)
    
    st.markdown(f"#### 📋 告警记录 ({len(filtered_history)}条)")
    
    if not filtered_history:
        st.info("暂无告警记录")
    else:
        for alert in filtered_history:
            level_info = ALERT_LEVELS.get(alert.get('alert_level', 'info'), {})
            bg_color = level_info.get('bg_color', '#fff')
            text_color = level_info.get('color', '#333')
            is_upgraded = alert.get('upgraded', False)
            original_level_info = ALERT_LEVELS.get(alert.get('original_level', alert.get('alert_level', 'info')), {})

            is_expanded = st.session_state.alert_expanded_row == alert['alert_id']

            with st.container():
                upgrade_icon = "⬆️" if is_upgraded else "&nbsp;&nbsp;"
                upgrade_badge = ""
                if is_upgraded:
                    upgrade_badge = (
                        f"<span style='margin-left:8px; background:#FF5722; color:white; "
                        f"padding:2px 8px; border-radius:10px; font-size:11px; font-weight:bold;'>"
                        f"升级: {original_level_info.get('name', '')}→{level_info.get('name', '')}</span>"
                    )

                st.markdown(
                    f"""
                    <div style="background-color: {bg_color}; padding: 12px; border-radius: 8px; margin-bottom: 4px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div style="display: flex; align-items: center;">
                                <span style="font-size: 20px; margin-right: 8px; min-width: 24px; text-align: center;">{upgrade_icon}</span>
                                <div>
                                    <span style="font-weight: bold; color: {text_color};">
                                        [{level_info.get('name', '')}] {alert['rule_name']}
                                    </span>
                                    {upgrade_badge}
                                    <span style="margin-left: 12px; color: #666;">
                                        站点: {alert['station_id']}
                                    </span>
                                </div>
                            </div>
                            <div style="color: #666; font-size: 0.9em;">
                                {alert['alert_time']}
                            </div>
                        </div>
                        <div style="margin-top: 6px; color: #555; margin-left: 32px;">
                            实测值: <strong>{alert['measured_value']}</strong> / 阈值: {alert['threshold']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                if st.button("📊 查看详情", key=f"detail_{alert['alert_id']}"):
                    if is_expanded:
                        st.session_state.alert_expanded_row = None
                    else:
                        st.session_state.alert_expanded_row = alert['alert_id']
                    st.rerun()
                
                if is_expanded:
                    st.markdown("##### 📈 触发时刻前后Leq曲线")
                    
                    alert_time = _parse_datetime(alert.get('alert_time', ''))
                    if not alert_time:
                        alert_time = datetime.now()
                    
                    window_start = alert_time - timedelta(hours=2)
                    window_end = alert_time + timedelta(hours=2)
                    
                    detail_df = get_station_measurements(
                        alert['station_id'],
                        window_start.strftime('%Y-%m-%d %H:%M:%S'),
                        window_end.strftime('%Y-%m-%d %H:%M:%S')
                    )
                    
                    if detail_df.empty:
                        window_start = alert_time - timedelta(hours=24)
                        window_end = alert_time + timedelta(hours=24)
                        detail_df = get_station_measurements(
                            alert['station_id'],
                            window_start.strftime('%Y-%m-%d %H:%M:%S'),
                            window_end.strftime('%Y-%m-%d %H:%M:%S')
                        )
                    
                    if not detail_df.empty:
                        fig_detail, ax_detail = plt.subplots(figsize=(10, 3))
                        times = detail_df['measurement_time']
                        leq_vals = detail_df['leq']
                        ax_detail.plot(times, leq_vals, 'b-o', linewidth=1.5, markersize=4, label='Leq')
                        ax_detail.axhline(y=float(alert['threshold']), color='red', linestyle='--', 
                                         label=f'阈值 ({alert["threshold"]} dB)', alpha=0.7)
                        ax_detail.axvline(x=pd.Timestamp(alert_time), color='orange', linestyle=':', 
                                         label='告警触发时刻', alpha=0.7, linewidth=2)
                        ax_detail.set_ylabel('Leq (dB)')
                        ax_detail.set_title(f"站点 {alert['station_id']} - 告警触发前后 ({window_start.strftime('%m-%d %H:%M')} ~ {window_end.strftime('%m-%d %H:%M')})")
                        ax_detail.legend(loc='best')
                        ax_detail.grid(True, alpha=0.3)
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig_detail)
                        plt.close(fig_detail)
                        
                        peak_val = float(detail_df['leq'].max())
                        peak_time = detail_df.loc[detail_df['leq'].idxmax(), 'measurement_time']
                        st.caption(f"📊 窗口内峰值: {peak_val:.1f} dB @ {peak_time}")
                    else:
                        st.warning("该时段暂无监测数据，可能数据时间范围不匹配")
                        data_range = get_measurement_time_range()
                        if data_range:
                            st.info(f"💡 现有数据范围: {data_range[0]} ~ {data_range[1]}")
                    
                    coop_info = _get_coop_event_info_for_alert(alert)
                    if not coop_info:
                        coop_info = _auto_run_coop_for_alert(alert)
                    if coop_info:
                        st.markdown("##### 🔗 协同溯源关联")
                        st.info(
                            f"该站点属于协同事件组 **{coop_info['group_id']}**\n\n"
                            f"- 参与站点数: {coop_info['station_count']}个\n"
                            f"- 定位状态: {coop_info['location_status']}"
                        )
                    
                    st.markdown("---")


def _auto_run_coop_for_alert(alert: Dict) -> Optional[Dict]:
    alert_time = _parse_datetime(alert.get('alert_time', ''))
    if not alert_time:
        return None
    
    station_id = alert.get('station_id')
    cache_key = f"{station_id}_{alert_time.strftime('%Y%m%d')}"
    
    if 'alert_coop_cache' not in st.session_state:
        st.session_state.alert_coop_cache = {}
    
    if cache_key in st.session_state.alert_coop_cache:
        return _get_coop_event_info_for_alert(alert)
    
    stations_df = get_all_stations()
    if stations_df.empty:
        return None
    
    window_start = alert_time - timedelta(hours=6)
    window_end = alert_time + timedelta(hours=6)
    
    nearby_stations = []
    try:
        target_row = stations_df[stations_df['station_id'] == station_id]
        if not target_row.empty:
            target_lat = float(target_row.iloc[0]['latitude'])
            target_lon = float(target_row.iloc[0]['longitude'])
            for _, row in stations_df.iterrows():
                dist = haversine_distance(target_lat, target_lon, 
                                          float(row['latitude']), float(row['longitude']))
                if dist < 5000 or len(nearby_stations) < 5:
                    nearby_stations.append((row['station_id'], dist))
            nearby_stations.sort(key=lambda x: x[1])
            selected_ids = [s[0] for s in nearby_stations[:min(8, len(nearby_stations))]]
        else:
            selected_ids = stations_df['station_id'].tolist()[:min(8, len(stations_df))]
    except Exception:
        selected_ids = stations_df['station_id'].tolist()[:min(8, len(stations_df))]
    
    if len(selected_ids) < 3:
        st.session_state.alert_coop_cache[cache_key] = False
        return None
    
    try:
        with st.spinner('正在关联协同溯源分析，请稍候...'):
            dist_matrix, delay_matrix, ordered_ids = compute_station_distance_matrix(
                stations_df[stations_df['station_id'].isin(selected_ids)].drop_duplicates('station_id')
            )
            if len(ordered_ids) < 3:
                st.session_state.alert_coop_cache[cache_key] = False
                return None
            
            all_events = detect_events_for_stations(ordered_ids, threshold_db=5.0)
            cooperative_groups = match_cooperative_events(
                all_events, dist_matrix, ordered_ids,
                spectrum_threshold=0.5, time_tolerance=5.0
            )
            location_results = {}
            for group in cooperative_groups:
                selected_stations_subset = stations_df[
                    stations_df['station_id'].isin(group['participating_stations'])
                ].drop_duplicates('station_id')
                loc = estimate_source_location(group, selected_stations_subset)
                location_results[group['group_id']] = loc
            
            existing_coop = st.session_state.get('coop_result', []) or []
            existing_loc = st.session_state.get('coop_locations', {}) or {}
            
            existing_ids = {g['group_id'] for g in existing_coop}
            for g in cooperative_groups:
                if g['group_id'] not in existing_ids:
                    existing_coop.append(g)
            
            existing_loc.update(location_results)
            
            st.session_state.coop_result = existing_coop
            st.session_state.coop_locations = existing_loc
            st.session_state.alert_coop_cache[cache_key] = True
        
        return _get_coop_event_info_for_alert(alert)
    except Exception as e:
        st.session_state.alert_coop_cache[cache_key] = False
        return None


def _get_coop_event_info_for_alert(alert: Dict) -> Optional[Dict]:
    coop_result = st.session_state.get('coop_result')
    if not coop_result:
        return None
    
    station_id = alert.get('station_id')
    alert_time = _parse_datetime(alert.get('alert_time', ''))
    if not alert_time:
        return None
    
    best_match = None
    min_time_diff = float('inf')
    
    for group in coop_result:
        participating_stations = group.get('participating_stations', [])
        if station_id in participating_stations:
            group_start = group.get('earliest_time')
            group_end = group.get('latest_time')
            if isinstance(group_start, pd.Timestamp):
                group_start = group_start.to_pydatetime()
            if isinstance(group_end, pd.Timestamp):
                group_end = group_end.to_pydatetime()
            
            try:
                if group_start and group_end:
                    if group_start - timedelta(hours=6) <= alert_time <= group_end + timedelta(hours=6):
                        center_time = group_start + (group_end - group_start) / 2
                        time_diff = abs((alert_time - center_time).total_seconds())
                        if time_diff < min_time_diff:
                            min_time_diff = time_diff
                            best_match = group
            except Exception:
                pass
    
    if best_match:
        group = best_match
        participating_stations = group.get('participating_stations', [])
        location_status = "未定位"
        locations = st.session_state.get('coop_locations')
        if locations and group.get('group_id') in locations:
            loc = locations[group['group_id']]
            if loc.get('located'):
                location_status = "已定位"
            elif loc.get('method'):
                location_status = f"部分定位({loc.get('method', '')})"
        
        return {
            'group_id': group['group_id'],
            'station_count': len(participating_stations),
            'location_status': location_status
        }
    
    return None


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


def page_cooperative_tracing():
    st.header("🎯 多站点协同监测与异常溯源")
    
    stations_df = get_all_stations()
    if stations_df.empty:
        st.warning("暂无监测站点数据，请先导入数据。")
        return
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📍 步骤1: 选择监测站点")
    st.sidebar.info("请选择 3~8 个监测站点进行协同分析")
    
    station_options = []
    for _, row in stations_df.iterrows():
        label = f"{row.get('station_name', row['station_id'])} ({row['station_id']})"
        station_options.append((row['station_id'], label))
    
    default_selected = st.session_state.coop_selected_stations
    if not default_selected and len(station_options) >= 3:
        default_selected = [s[0] for s in station_options[:min(4, len(station_options))]]
    
    selected_ids = st.sidebar.multiselect(
        "选择监测站点 (3~8个)",
        options=[s[0] for s in station_options],
        format_func=lambda x: next((lbl for sid, lbl in station_options if sid == x), x),
        default=default_selected,
        key="coop_station_multiselect"
    )
    st.session_state.coop_selected_stations = selected_ids
    
    valid_count = len(selected_ids)
    if valid_count < 3:
        st.sidebar.warning(f"⚠️ 已选 {valid_count} 个，至少需 3 个")
    elif valid_count > 8:
        st.sidebar.warning(f"⚠️ 已选 {valid_count} 个，最多 8 个")
    else:
        st.sidebar.success(f"✅ 已选 {valid_count} 个站点")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ 步骤2: 分析参数")
    
    event_threshold = st.sidebar.slider(
        "事件检测阈值 (dB)",
        min_value=5, max_value=20, value=int(st.session_state.event_threshold), step=1,
        key="coop_event_threshold",
        help="Leq相对背景升高超过此阈值即标记为噪声事件"
    )
    spectrum_threshold = st.sidebar.slider(
        "频谱相似度阈值",
        min_value=0.3, max_value=0.95,
        value=st.session_state.coop_spectrum_threshold, step=0.05,
        key="coop_spec_threshold",
        help="余弦相似度高于此阈值认为事件频谱特征相似"
    )
    st.session_state.coop_spectrum_threshold = spectrum_threshold
    
    time_tolerance = st.sidebar.slider(
        "时间容差 (秒)",
        min_value=0.5, max_value=10.0,
        value=st.session_state.coop_time_tolerance, step=0.5,
        key="coop_time_tol",
        help="允许的时间匹配误差范围"
    )
    st.session_state.coop_time_tolerance = time_tolerance
    
    run_btn = st.sidebar.button("🚀 开始协同分析", type="primary", use_container_width=True)
    
    if len(selected_ids) < 3:
        st.info("👈 请在左侧边栏选择至少 3 个监测站点，然后点击「开始协同分析」按钮")
        return
    
    selected_stations = stations_df[stations_df['station_id'].isin(selected_ids)].copy()
    selected_stations = selected_stations.set_index('station_id').loc[selected_ids].reset_index()
    
    dist_matrix, delay_matrix, station_ids = compute_station_distance_matrix(selected_stations)
    n = len(station_ids)
    
    st.markdown("### 📐 站点邻接矩阵")
    
    col_dist, col_delay = st.columns(2)
    
    with col_dist:
        st.markdown("#### 直线距离 (米)")
        dist_data = []
        for i in range(n):
            row = {'站点': station_ids[i]}
            for j in range(n):
                row[station_ids[j]] = f"{dist_matrix[i, j]:.0f}" if i != j else "-"
            dist_data.append(row)
        dist_df = pd.DataFrame(dist_data).set_index('站点')
        st.dataframe(dist_df, use_container_width=True)
    
    with col_delay:
        st.markdown(f"#### 预估传播时延 (秒, 声速={SOUND_SPEED}m/s)")
        delay_data = []
        for i in range(n):
            row = {'站点': station_ids[i]}
            for j in range(n):
                row[station_ids[j]] = f"{delay_matrix[i, j]:.2f}" if i != j else "-"
            delay_data.append(row)
        delay_df = pd.DataFrame(delay_data).set_index('站点')
        st.dataframe(delay_df, use_container_width=True)
    
    cooperative_groups = None
    location_results = None
    
    if run_btn:
        with st.spinner("正在检测各站点噪声事件..."):
            all_events = detect_events_for_stations(selected_ids, threshold_db=float(event_threshold))
        
        total_events = sum(len(v) for v in all_events.values())
        st.info(f"📊 共检测到 {total_events} 个噪声事件")
        
        with st.spinner("正在匹配协同事件组..."):
            cooperative_groups = match_cooperative_events(
                all_events, delay_matrix, station_ids,
                spectrum_threshold=spectrum_threshold,
                time_tolerance=time_tolerance
            )
            st.session_state.coop_result = cooperative_groups
        
        with st.spinner("正在估计声源位置..."):
            location_results = {}
            for group in cooperative_groups:
                loc = estimate_source_location(group, selected_stations)
                location_results[group['group_id']] = loc
            st.session_state.coop_locations = location_results
    else:
        cooperative_groups = st.session_state.coop_result
        location_results = st.session_state.coop_locations
    
    if cooperative_groups is None:
        return
    
    st.markdown("---")
    st.markdown("### 🎯 步骤3: 溯源结果")
    
    if not cooperative_groups:
        st.warning("未检测到跨站点的协同噪声事件，建议尝试：\n"
                  "- 降低事件检测阈值\n"
                  "- 降低频谱相似度阈值\n"
                  "- 增大时间容差范围")
        return
    
    loc_count = sum(1 for loc in location_results.values() if loc.get('located') and loc.get('latitude'))
    two_station_count = sum(1 for g in cooperative_groups if len(g['participating_stations']) == 2)
    multi_station_count = sum(1 for g in cooperative_groups if len(g['participating_stations']) >= 3)
    
    col_r1, col_r2, col_r3, col_r4, col_r5 = st.columns(5)
    with col_r1:
        st.metric("协同事件组数", f"{len(cooperative_groups)} 组")
    with col_r2:
        st.metric("3+站精确定位", f"{loc_count} 组")
    with col_r3:
        st.metric("2站双曲线", f"{two_station_count} 组")
    with col_r4:
        all_stations_in = set()
        for g in cooperative_groups:
            all_stations_in.update(g['participating_stations'])
        st.metric("参与站点总数", f"{len(all_stations_in)} 个")
    with col_r5:
        if cooperative_groups:
            avg_sim = np.mean([g['avg_spectrum_similarity'] for g in cooperative_groups])
            st.metric("平均频谱相似度", f"{avg_sim:.3f}")
    
    st.markdown("---")
    
    if 'coop_selected_group' not in st.session_state:
        st.session_state.coop_selected_group = None
    
    groups_sorted = sorted(cooperative_groups, key=lambda g: g['earliest_time'])
    
    tab_map, tab_timeline, tab_table, tab_export = st.tabs([
        "🗺️ 溯源地图", "⏱️ 事件时间轴", "📋 统计表格", "📤 导出GeoJSON"
    ])
    
    with tab_map:
        st.markdown("#### 🔵 监测站点 | 🟠 2站双曲线组 | 🔴 3+站定位组 | ---- 站点连线")
        
        center_lat = selected_stations['latitude'].mean()
        center_lon = selected_stations['longitude'].mean()
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=13,
            tiles='OpenStreetMap'
        )
        
        for _, row in selected_stations.iterrows():
            sid = row['station_id']
            sname = row.get('station_name', sid)
            popup_html = f"""
            <b>{sname}</b><br>
            编号: {sid}<br>
            区域: {row.get('region', '未知')}
            """
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=folium.Popup(popup_html, max_width=300),
                icon=folium.Icon(color='blue', icon='info-sign'),
                tooltip=f"📍 {sname}"
            ).add_to(m)
        
        station_coords = {}
        for _, row in selected_stations.iterrows():
            station_coords[row['station_id']] = (row['latitude'], row['longitude'])
        
        for i in range(n):
            for j in range(i + 1, n):
                s1 = station_ids[i]
                s2 = station_ids[j]
                if s1 in station_coords and s2 in station_coords:
                    c1 = station_coords[s1]
                    c2 = station_coords[s2]
                    folium.PolyLine(
                        [c1, c2],
                        color='#888888',
                        weight=1,
                        opacity=0.4,
                        dash_array='5, 5',
                        tooltip=f"{s1} ↔ {s2}: {dist_matrix[i, j]:.0f}m / {delay_matrix[i, j]:.2f}s"
                    ).add_to(m)
        
        group_colors = ['#E53935', '#D81B60', '#8E24AA', '#5E35B1',
                        '#3949AB', '#1E88E5', '#00ACC1', '#00897B']
        two_station_color = '#FF9800'
        
        for gi, group in enumerate(cooperative_groups):
            gid = group['group_id']
            loc = location_results.get(gid, {})
            num_stations = len(group['participating_stations'])
            
            if num_stations >= 3:
                color = group_colors[gi % len(group_colors)]
            else:
                color = two_station_color
            
            for hyp in loc.get('hyperbolas', []):
                if hyp.get('points'):
                    folium.PolyLine(
                        hyp['points'],
                        color=color,
                        weight=3 if num_stations >= 3 else 2.5,
                        opacity=0.7 if num_stations >= 3 else 0.8,
                        dash_array='2, 4' if num_stations >= 3 else '6, 3',
                        tooltip=f"{gid} {'定位' if num_stations>=3 else '2站'}: {hyp['station_pair'][0]}→{hyp['station_pair'][1]}, Δd={hyp['delta_distance_m']:.0f}m"
                    ).add_to(m)
            
            if loc.get('latitude') and loc.get('longitude'):
                uncertainty = loc.get('uncertainty_m', 100)
                folium.Circle(
                    location=[loc['latitude'], loc['longitude']],
                    radius=uncertainty,
                    color=color,
                    weight=2,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.15,
                    popup=f"""
                    <b>{gid} 估计声源</b><br>
                    参与站点: {num_stations}个<br>
                    方位角: {loc.get('bearing_deg', 'N/A')}°<br>
                    距最早站: {loc.get('distance_from_earliest_m', 'N/A')}m<br>
                    不确定度: {uncertainty:.0f}m<br>
                    定位RMSE: {loc.get('rmse', 'N/A'):.1f}m<br>
                    站点: {', '.join(group['participating_stations'])}
                    """,
                    tooltip=f"{gid} 声源 (±{uncertainty:.0f}m, {num_stations}站)"
                ).add_to(m)
                
                folium.CircleMarker(
                    location=[loc['latitude'], loc['longitude']],
                    radius=8,
                    color='white',
                    weight=2,
                    fill=True,
                    fill_color=color,
                    fill_opacity=1.0,
                    popup=f"{gid} 声源中心点"
                ).add_to(m)
            elif num_stations == 2 and loc.get('hyperbolas'):
                s_pair = group['participating_stations']
                if len(s_pair) == 2 and s_pair[0] in station_coords and s_pair[1] in station_coords:
                    c1 = station_coords[s_pair[0]]
                    c2 = station_coords[s_pair[1]]
                    mid_lat = (c1[0] + c2[0]) / 2
                    mid_lon = (c1[1] + c2[1]) / 2
                    
                    td = group.get('time_diffs', {})
                    other_sid = s_pair[1] if s_pair[0] == group['earliest_station'] else s_pair[0]
                    delta_d = td.get(other_sid, 0) * SOUND_SPEED
                    
                    folium.CircleMarker(
                        location=[mid_lat, mid_lon],
                        radius=10,
                        color=color,
                        weight=3,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.5,
                        popup=f"""
                        <b>{gid} 2站协同事件</b><br>
                        站点: {s_pair[0]} ↔ {s_pair[1]}<br>
                        距离差: {delta_d:.0f}m<br>
                        仅绘制双曲线，无法精确定位<br>
                        平均峰值: {group['avg_peak_leq']:.1f}dB
                        """,
                        tooltip=f"{gid} 2站事件 | {s_pair[0]} & {s_pair[1]}"
                    ).add_to(m)
        
        all_bounds = []
        for _, row in selected_stations.iterrows():
            all_bounds.append([row['latitude'], row['longitude']])
        for loc in location_results.values():
            if loc.get('latitude') and loc.get('longitude'):
                all_bounds.append([loc['latitude'], loc['longitude']])
        if all_bounds:
            m.fit_bounds(all_bounds)
        
        st_folium(m, width='100%', height=600, returned_objects=[])
        
        st.caption("🟠 橙色圆圈 = 2站协同事件（仅提供双曲线，无法精确定位） | 🔴 红色圆圈 = 3站以上声源定位结果")
    
    with tab_timeline:
        st.markdown("#### 协同事件组时间轴 - 点击色块查看详情")
        st.caption("💡 点击图表中的色块即可选中对应事件组，下方自动展示该组详细信息")
        
        groups_sorted = sorted(cooperative_groups, key=lambda g: g['earliest_time'])
        
        if groups_sorted:
            import plotly.graph_objects as go
            
            group_colors_list = ['#E53935', '#D81B60', '#8E24AA', '#5E35B1',
                            '#3949AB', '#1E88E5', '#00ACC1', '#00897B']
            two_station_color = '#FF9800'
            
            group_color_map = {}
            for gi, group in enumerate(groups_sorted):
                gid = group['group_id']
                num_stations = len(group['participating_stations'])
                color_hex = group_colors_list[gi % len(group_colors_list)] if num_stations >= 3 else two_station_color
                group_color_map[gid] = color_hex
            
            gid_options = [g['group_id'] for g in groups_sorted]
            
            compare_options = gid_options
            compare_default = [g for g in st.session_state.coop_compare_groups if g in compare_options]
            
            st.markdown("##### 📊 多组对比分析 - 选择事件组（2-5个）")
            col_compare1, col_compare2 = st.columns([3, 1])
            with col_compare1:
                compare_selected = st.multiselect(
                    "选择要对比的事件组",
                    options=compare_options,
                    default=compare_default,
                    format_func=lambda x: f"{x} ({next((len(g['participating_stations']) for g in groups_sorted if g['group_id'] == x), 0)}站, {next((g['avg_peak_leq'] for g in groups_sorted if g['group_id'] == x), 0):.1f}dB)",
                    key="coop_compare_selector",
                    max_selections=5
                )
            with col_compare2:
                if st.button("🔄 重置选择", key="coop_compare_reset"):
                    st.session_state.coop_compare_groups = []
                    st.rerun()
            
            st.session_state.coop_compare_groups = compare_selected
            
            t_min = groups_sorted[0]['earliest_time'] - timedelta(minutes=30)
            t_max = groups_sorted[-1]['latest_time'] + timedelta(minutes=30)
            t0 = t_min
            
            y_labels = []
            fig = go.Figure()
            
            for gi, group in enumerate(reversed(groups_sorted)):
                gid = group['group_id']
                num_stations = len(group['participating_stations'])
                color_hex = group_color_map[gid]
                
                is_compared = gid in compare_selected
                
                if compare_selected:
                    opacity = 1.0 if is_compared else 0.3
                    line_width = 3 if is_compared else 1
                    line_color = '#000000' if is_compared else 'white'
                else:
                    opacity = 0.9
                    line_width = 1
                    line_color = 'white'
                
                start_min = (group['earliest_time'] - t0).total_seconds() / 60.0
                dur_min = max(0.5, (group['latest_time'] - group['earliest_time']).total_seconds() / 60.0)
                
                loc_tag = "精确定位" if num_stations >= 3 else "仅双曲线"
                stations_str = ", ".join(group['participating_stations'])
                
                y_label = f"{gid} [{num_stations}站]"
                y_labels.append(y_label)
                
                hover_text = (
                    f"<b>{gid}</b><br>"
                    f"站点数: {num_stations} ({loc_tag})<br>"
                    f"时间: {group['earliest_time'].strftime('%m-%d %H:%M:%S')}<br>"
                    f"持续: {dur_min:.1f} min<br>"
                    f"峰值Leq: {group['avg_peak_leq']:.1f} dB<br>"
                    f"参与站点: {stations_str}<br>"
                    f"频谱相似度: {group['avg_spectrum_similarity']:.3f}"
                )
                
                fig.add_trace(go.Bar(
                    y=[y_label],
                    x=[dur_min],
                    base=[start_min],
                    orientation='h',
                    marker_color=color_hex,
                    marker_line_color=line_color,
                    marker_line_width=line_width,
                    opacity=opacity,
                    hovertemplate=hover_text + '<extra></extra>',
                    customdata=[gid],
                    showlegend=False,
                    text=f"{gid}",
                    textposition='inside',
                    insidetextanchor='middle',
                    textfont=dict(color='white', size=11),
                ))
            
            fig.update_layout(
                height=max(300, len(groups_sorted) * 28 + 80),
                margin=dict(l=120, r=30, t=40, b=50),
                barmode='overlay',
                yaxis=dict(
                    tickmode='array',
                    tickvals=y_labels,
                    ticktext=y_labels,
                    tickfont=dict(size=11),
                ),
                xaxis=dict(
                    title=f'时间 (分钟，起始 {t0.strftime("%m-%d %H:%M")})',
                    tickfont=dict(size=10),
                ),
                plot_bgcolor='#fafafa',
                hoverlabel=dict(
                    bgcolor='white',
                    font_size=13,
                    font_family='sans-serif',
                ),
                dragmode='select',
            )
            
            selected_gid = None
            chart_result = st.plotly_chart(fig, on_select="rerun", key="coop_timeline_chart")
            
            selection_points = chart_result.get("selection", {}).get("points", [])
            if selection_points:
                for pt in selection_points:
                    custom = pt.get("customdata")
                    if custom:
                        selected_gid = custom
                        break
            
            def compute_comparison_data(selected_groups, groups_sorted, location_results, group_color_map):
                comparison_data = []
                
                all_num_stations = [len(g['participating_stations']) for g in groups_sorted]
                all_durations = [(g['latest_time'] - g['earliest_time']).total_seconds() / 60.0 for g in groups_sorted]
                all_leq = [g['avg_peak_leq'] for g in groups_sorted]
                all_sim = [g['avg_spectrum_similarity'] for g in groups_sorted]
                all_uncertainty = []
                for g in groups_sorted:
                    loc = location_results.get(g['group_id'], {})
                    unc = loc.get('uncertainty_m')
                    if unc is not None:
                        all_uncertainty.append(unc)
                
                max_stations = max(all_num_stations) if all_num_stations else 1
                max_duration = max(all_durations) if all_durations else 1
                max_leq = max(all_leq) if all_leq else 1
                max_sim = max(all_sim) if all_sim else 1
                max_uncertainty = max(all_uncertainty) if all_uncertainty else 1
                min_uncertainty = min(all_uncertainty) if all_uncertainty else 0
                
                weights = {
                    'stations': 0.3,
                    'duration': 0.1,
                    'leq': 0.2,
                    'similarity': 0.25,
                    'precision': 0.15
                }
                
                for gid in selected_groups:
                    group = next(g for g in groups_sorted if g['group_id'] == gid)
                    loc = location_results.get(gid, {})
                    
                    num_stations = len(group['participating_stations'])
                    duration_min = (group['latest_time'] - group['earliest_time']).total_seconds() / 60.0
                    avg_peak_leq = group['avg_peak_leq']
                    spectrum_similarity = group['avg_spectrum_similarity']
                    uncertainty = loc.get('uncertainty_m')
                    
                    norm_stations = num_stations / max_stations if max_stations > 0 else 0
                    norm_duration = duration_min / max_duration if max_duration > 0 else 0
                    norm_leq = avg_peak_leq / max_leq if max_leq > 0 else 0
                    norm_sim = spectrum_similarity / max_sim if max_sim > 0 else 0
                    
                    if uncertainty is not None and max_uncertainty > min_uncertainty:
                        norm_precision = 1 - (uncertainty - min_uncertainty) / (max_uncertainty - min_uncertainty)
                    elif uncertainty is not None:
                        norm_precision = 1
                    else:
                        norm_precision = 0
                    
                    norm_precision_for_radar = norm_precision
                    
                    composite_score = (
                        norm_stations * weights['stations'] +
                        norm_duration * weights['duration'] +
                        norm_leq * weights['leq'] +
                        norm_sim * weights['similarity'] +
                        norm_precision * weights['precision']
                    )
                    
                    comparison_data.append({
                        'group_id': gid,
                        'num_stations': num_stations,
                        'duration_min': duration_min,
                        'avg_peak_leq': avg_peak_leq,
                        'spectrum_similarity': spectrum_similarity,
                        'uncertainty': uncertainty,
                        'composite_score': composite_score,
                        'color': group_color_map.get(gid, '#333333'),
                        'norm_stations': norm_stations,
                        'norm_duration': norm_duration,
                        'norm_leq': norm_leq,
                        'norm_sim': norm_sim,
                        'norm_precision': norm_precision,
                        'norm_precision_for_radar': norm_precision_for_radar
                    })
                
                return comparison_data
            
            if len(compare_selected) >= 2:
                comparison_data = compute_comparison_data(
                    compare_selected, groups_sorted, location_results, group_color_map
                )
                st.session_state.coop_comparison_data = comparison_data
                st.session_state.coop_compare_selected = compare_selected
                
                all_have_uncertainty = all(d['uncertainty'] is not None for d in comparison_data)
                none_have_uncertainty = all(d['uncertainty'] is None for d in comparison_data)
                
                st.markdown("---")
                st.markdown("##### 📊 多组对比分析面板")
                
                radar_categories_full = ['参与站点数', '持续时长', '平均峰值Leq', '频谱相似度', '定位不确定度']
                radar_categories = ['参与站点数', '持续时长', '平均峰值Leq', '频谱相似度']
                if all_have_uncertainty:
                    radar_categories = radar_categories_full
                
                num_vars = len(radar_categories)
                
                radar_fig = go.Figure()
                
                for d in comparison_data:
                    gid = d['group_id']
                    is_highlighted = (gid == st.session_state.coop_selected_group)
                    line_width = 4 if is_highlighted else 2
                    
                    values = [
                        d['norm_stations'],
                        d['norm_duration'],
                        d['norm_leq'],
                        d['norm_sim'],
                    ]
                    if all_have_uncertainty:
                        values.append(d['norm_precision_for_radar'])
                    values += values[:1]
                    
                    angles = [n / float(num_vars) * 2 * 3.14159 for n in range(num_vars)]
                    angles += angles[:1]
                    
                    legend_name = gid
                    if none_have_uncertainty:
                        legend_name = f"{gid} (无定位数据)"
                    
                    radar_fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=radar_categories + [radar_categories[0]],
                        fill='toself',
                        name=legend_name,
                        line=dict(color=d['color'], width=line_width),
                        marker=dict(size=6, color=d['color']),
                        opacity=0.8,
                        customdata=[gid]
                    ))
                
                radar_title = '多维度对比雷达图'
                if none_have_uncertainty:
                    radar_title = '多维度对比雷达图（无定位不确定度数据）'
                
                radar_fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1],
                            ticktext=['0%', '20%', '40%', '60%', '80%', '100%'],
                            tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0]
                        ),
                        angularaxis=dict(
                            tickfont=dict(size=11)
                        )
                    ),
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.15,
                        xanchor="center",
                        x=0.5
                    ),
                    height=450,
                    margin=dict(l=20, r=20, t=30, b=100),
                    title=dict(
                        text=radar_title,
                        font=dict(size=14)
                    )
                )
                
                st.session_state.coop_radar_fig_json = radar_fig.to_json()
                
                col_radar, col_table = st.columns([1, 1])
                
                with col_radar:
                    radar_click = st.plotly_chart(
                        radar_fig, 
                        key="coop_radar_chart",
                        on_select="rerun",
                        use_container_width=True
                    )
                    if none_have_uncertainty:
                        st.info("ℹ️ 所选事件组均为2站组，无精确定位数据，雷达图暂不包含\"定位不确定度\"维度")
                
                with col_table:
                    st.markdown("**对比数据表格**")
                    st.caption("💡 点击表格中的组ID按钮可切换下方详情面板")
                    
                    for i, d in enumerate(comparison_data):
                        is_highlighted = (d['group_id'] == st.session_state.coop_selected_group)
                        bg_color = f"{d['color']}15" if is_highlighted else "#ffffff"
                        border_color = d['color'] if is_highlighted else "#e0e0e0"
                        weight = "bold" if is_highlighted else "normal"
                        
                        if i == 0:
                            header_cols = st.columns([1.2, 1, 1.2, 1.2, 1, 1.3, 1])
                            headers = ['组ID', '站点数', '持续(min)', '峰值(dB)', '频谱相似', '不确定度(m)', '综合评分']
                            for j, h in enumerate(headers):
                                with header_cols[j]:
                                    st.markdown(f"<div style='text-align:center; font-weight:bold; color:#1565C0; font-size:12px;'>{h}</div>", unsafe_allow_html=True)
                        
                        cols = st.columns([1.2, 1, 1.2, 1.2, 1, 1.3, 1])
                        with cols[0]:
                            if st.button(
                                f"📌 {d['group_id']}",
                                key=f"table_goto_{d['group_id']}",
                                help=f"点击查看{d['group_id']}详情",
                                use_container_width=True,
                                type="secondary" if not is_highlighted else "primary"
                            ):
                                st.session_state.coop_selected_group = d['group_id']
                                st.rerun()
                        with cols[1]:
                            st.markdown(f"<div style='text-align:center; font-weight:{weight};'>{d['num_stations']}</div>", unsafe_allow_html=True)
                        with cols[2]:
                            st.markdown(f"<div style='text-align:center; font-weight:{weight};'>{d['duration_min']:.1f}</div>", unsafe_allow_html=True)
                        with cols[3]:
                            st.markdown(f"<div style='text-align:center; font-weight:{weight};'>{d['avg_peak_leq']:.1f}</div>", unsafe_allow_html=True)
                        with cols[4]:
                            st.markdown(f"<div style='text-align:center; font-weight:{weight};'>{d['spectrum_similarity']:.3f}</div>", unsafe_allow_html=True)
                        with cols[5]:
                            unc_text = f"{d['uncertainty']:.0f}" if d['uncertainty'] else 'N/A'
                            st.markdown(f"<div style='text-align:center; font-weight:{weight};'>{unc_text}</div>", unsafe_allow_html=True)
                        with cols[6]:
                            st.markdown(f"<div style='text-align:center; font-weight:{weight}; color:#1976D2;'>{d['composite_score']:.3f}</div>", unsafe_allow_html=True)
                        
                        st.markdown(
                            f"<div style='height:1px; background:linear-gradient(to right, transparent, {border_color}, transparent); margin:2px 0;'></div>",
                            unsafe_allow_html=True
                        )
                
                col_export1, col_export2 = st.columns([1, 3])
                with col_export1:
                    try:
                        import plotly.io as pio
                        
                        prev_selected = st.session_state.get('coop_prev_pdf_groups', [])
                        if prev_selected != compare_selected:
                            st.session_state.coop_pdf_data = None
                            st.session_state.coop_pdf_error = None
                            st.session_state.coop_prev_pdf_groups = list(compare_selected)
                        
                        if st.button(
                            "📄 导出对比报告PDF", 
                            type="primary", 
                            key="export_compare_report_btn",
                            use_container_width=True
                        ):
                            try:
                                with st.spinner("正在生成雷达图截图..."):
                                    radar_bytes = pio.to_image(
                                        radar_fig, format='png', 
                                        width=800, height=600, scale=2
                                    )
                                
                                with st.spinner("正在生成PDF报告..."):
                                    pdf_comparison_data = []
                                    for d in comparison_data:
                                        d_copy = dict(d)
                                        d_copy['highlighted'] = (d['group_id'] == st.session_state.coop_selected_group)
                                        pdf_comparison_data.append(d_copy)
                                    
                                    pdf_colors = {d['group_id']: d['color'] for d in comparison_data}
                                    
                                    pdf_bytes = generate_comparison_report_pdf(
                                        pdf_comparison_data,
                                        pdf_colors,
                                        radar_bytes
                                    )
                                    
                                    gids_str = "_".join(compare_selected)
                                    st.session_state.coop_pdf_data = {
                                        'bytes': pdf_bytes,
                                        'filename': f"协同事件组对比报告_{gids_str}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                                    }
                                    st.session_state.coop_pdf_error = None
                            except Exception as e:
                                st.session_state.coop_pdf_error = f"PDF生成失败: {str(e)}"
                                st.session_state.coop_pdf_data = None
                        
                        if st.session_state.get('coop_pdf_data'):
                            pdf_data = st.session_state.coop_pdf_data
                            st.success("✅ 报告已生成，可以下载了！")
                            st.download_button(
                                label="⬇️ 下载对比报告",
                                data=pdf_data['bytes'],
                                file_name=pdf_data['filename'],
                                mime="application/pdf",
                                type="primary",
                                key="download_compare_pdf_final",
                                use_container_width=True
                            )
                            if st.button("重新生成", key="regenerate_pdf", use_container_width=True):
                                st.session_state.coop_pdf_data = None
                                st.rerun()
                        
                        if st.session_state.get('coop_pdf_error'):
                            st.error(f"❌ {st.session_state.coop_pdf_error}")
                            st.caption("💡 可能原因：kaleido 未安装。请尝试执行: pip install kaleido")
                            if st.button("清除错误提示", key="clear_pdf_error", use_container_width=True):
                                st.session_state.coop_pdf_error = None
                                st.rerun()
                    except Exception as import_err:
                        st.error(f"❌ 依赖缺失: {import_err}")
                        st.info("请确保已安装 plotly 和 kaleido 包")
                with col_export2:
                    st.caption("📝 PDF报告包含：雷达图截图、对比数据表格、自动生成的分析总结")
                    st.caption("💡 提示：点击'导出PDF'生成报告，生成后可点击'下载'按钮保存文件")
                
                st.markdown("---")
            elif len(compare_selected) < 2:
                if cooperative_groups and len(cooperative_groups) > 0:
                    st.info("ℹ️ 请至少选择2个事件组进行对比分析（最多选择5个）")
            
            if selected_gid and selected_gid in gid_options:
                default_idx = gid_options.index(selected_gid)
            elif st.session_state.coop_selected_group in gid_options:
                default_idx = gid_options.index(st.session_state.coop_selected_group)
            else:
                default_idx = 0
            
            st.markdown("---")
            
            selected_group = st.selectbox(
                "📋 选中事件组",
                options=gid_options,
                index=default_idx,
                format_func=lambda x: f"{x} - {next((g['earliest_time'].strftime('%m-%d %H:%M:%S') for g in groups_sorted if g['group_id'] == x), '')} ({next((len(g['participating_stations']) for g in groups_sorted if g['group_id'] == x), 0)}站)",
                key="coop_group_selector"
            )
            st.session_state.coop_selected_group = selected_group
            
            if len(compare_selected) >= 2 and selected_group in compare_selected:
                for d in comparison_data:
                    d['highlighted'] = (d['group_id'] == selected_group)
            
            if selected_group:
                group = next(g for g in groups_sorted if g['group_id'] == selected_group)
                loc = location_results.get(selected_group, {})
                num_stations = len(group['participating_stations'])
                
                if num_stations >= 3:
                    gi_orig = gid_options.index(selected_group)
                    highlight_color = group_colors_list[gi_orig % len(group_colors_list)]
                else:
                    highlight_color = two_station_color
                
                st.markdown(f'<div style="border-left:4px solid {highlight_color};padding-left:12px;margin:4px 0;">', unsafe_allow_html=True)
                
                col_d1, col_d2, col_d3, col_d4 = st.columns(4)
                with col_d1:
                    st.metric("参与站点数", f"{num_stations} 个",
                             help="3个及以上可进行精确定位")
                with col_d2:
                    st.metric("最早触发站", group['earliest_station'])
                with col_d3:
                    if loc.get('bearing_deg') is not None:
                        st.metric("估计方位角", f"{loc['bearing_deg']:.1f}°")
                    else:
                        st.metric("定位状态", "仅双曲线" if num_stations == 2 else "N/A")
                with col_d4:
                    if loc.get('distance_from_earliest_m') is not None:
                        st.metric("估计距离", f"{loc['distance_from_earliest_m']:.0f} m")
                    else:
                        st.metric("频谱相似度", f"{group['avg_spectrum_similarity']:.3f}")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("**各站点触发时间差 (相对最早站)**")
                td_rows = []
                for sid, td in sorted(group['time_diffs'].items(), key=lambda x: x[1]):
                    td_rows.append({
                        '站点': sid,
                        '到达时间差 (秒)': f"{td:.2f}",
                        '对应距离差 (m)': f"{td * SOUND_SPEED:.0f}",
                        '触发顺序': f"第 {list(group['time_diffs'].keys()).index(sid) + 1} 个"
                    })
                st.table(pd.DataFrame(td_rows))
                
                if num_stations == 2:
                    st.warning("⚠️ 该事件组仅有2个站点参与，仅能绘制TDOA等距差双曲线，声源可能位于双曲线上任意位置，无法精确定位。建议增加监测站点数量以提高定位精度。")
                elif loc.get('uncertainty_m') is not None:
                    st.caption(f"📍 定位不确定度: ±{loc['uncertainty_m']:.0f}m | 定位RMSE: {loc.get('rmse', 'N/A'):.1f} | 频谱相似度: {group['avg_spectrum_similarity']:.3f}")
                
                st.markdown("**参与站点的事件详情**")
                event_detail_rows = []
                for e in group['events']:
                    dur_h = int(e['duration_minutes'] // 60)
                    dur_m = int(e['duration_minutes'] % 60)
                    dur_str = f"{dur_h}h{dur_m}min" if dur_h > 0 else f"{dur_m}min"
                    e_icon = e.get('icon', '❓') if e.get('icon') else '❓'
                    source = e.get('source_name', '未知') if e.get('source_name') else '未知'
                    event_detail_rows.append({
                        '站点': e['station_id'],
                        '开始时间': e['start_time'].strftime('%Y-%m-%d %H:%M:%S'),
                        '结束时间': e['end_time'].strftime('%Y-%m-%d %H:%M:%S'),
                        '峰值Leq': f"{e['peak_leq']:.1f} dB",
                        '持续时长': dur_str,
                        '推测来源': f"{e_icon} {source}"
                    })
                st.table(pd.DataFrame(event_detail_rows))
    
    with tab_table:
        st.markdown("#### 协同事件组汇总统计表")
        
        table_rows = []
        for group in groups_sorted:
            gid = group['group_id']
            loc = location_results.get(gid, {})
            num_stations = len(group['participating_stations'])
            
            has_alert = False
            for sid in group['participating_stations']:
                if has_active_alerts(sid, lookback_hours=24):
                    has_alert = True
                    break
            alert_tag = " 🔔 有活跃告警" if has_alert else ""
            
            table_rows.append({
                '组ID': gid + alert_tag,
                '站点数': num_stations,
                '定位类型': '精确定位' if num_stations >= 3 else '仅双曲线',
                '最早触发站': group['earliest_station'],
                '最早时间': group['earliest_time'].strftime('%Y-%m-%d %H:%M:%S'),
                '持续时长 (min)': f"{(group['latest_time'] - group['earliest_time']).total_seconds() / 60:.1f}",
                '平均峰值Leq (dB)': f"{group['avg_peak_leq']:.1f}",
                '估计方位角 (°)': f"{loc['bearing_deg']:.1f}" if loc.get('bearing_deg') is not None else "N/A",
                '估计距离 (m)': f"{loc['distance_from_earliest_m']:.0f}" if loc.get('distance_from_earliest_m') is not None else "N/A",
                '定位不确定度 (m)': f"{loc['uncertainty_m']:.0f}" if loc.get('uncertainty_m') is not None else "N/A",
                '频谱相似度': f"{group['avg_spectrum_similarity']:.3f}"
            })
        
        result_df = pd.DataFrame(table_rows)
        
        def highlight_row(row):
            if row['定位类型'] == '仅双曲线':
                return ['background-color: #fff3e0; color: #e65100'] * len(row)
            return [''] * len(row)
        
        styled = result_df.style.apply(highlight_row, axis=1)
        st.dataframe(styled, use_container_width=True, hide_index=True, height=400)
        
        st.caption("🟠 橙色高亮行 = 2站事件组，仅能提供TDOA双曲线，无法精确定位")
    
    with tab_export:
        st.markdown("#### 📤 导出溯源结果为GeoJSON")
        st.info("导出的GeoJSON文件包含：估计声源位置点、不确定度圆、TDOA双曲线、监测站点位置，可直接在GIS软件中打开")
        
        geojson_data = export_traceability_geojson(
            cooperative_groups, selected_stations, location_results
        )
        
        geojson_str = json.dumps(geojson_data, ensure_ascii=False, indent=2)
        
        col_dl1, col_dl2 = st.columns([1, 2])
        with col_dl1:
            export_name = st.text_input("文件名", value=f"协同溯源结果_{datetime.now().strftime('%Y%m%d_%H%M')}")
            st.download_button(
                label="⬇️ 下载GeoJSON",
                data=geojson_str,
                file_name=f"{export_name}.geojson",
                mime="application/geo+json",
                type="primary",
                use_container_width=True
            )
        with col_dl2:
            st.metric("文件大小", f"{len(geojson_str.encode('utf-8')) / 1024:.1f} KB")
            st.metric("要素数量", f"{len(geojson_data['features'])} 个")
        
        with st.expander("👀 预览GeoJSON内容", expanded=False):
            st.code(geojson_str, language="json")


if __name__ == '__main__':
    main()