import numpy as np
import pandas as pd
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from data_models import get_data_statistics, ZONE_STANDARDS
from contour_utils import NOISE_LEVEL_RANGES

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

CHINESE_FONT_PATHS = [
    '/System/Library/Fonts/PingFang.ttc',
    '/System/Library/Fonts/STHeiti Medium.ttc',
    '/System/Library/Fonts/Hiragino Sans GB.ttc',
    '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    'C:/Windows/Fonts/simhei.ttf',
    'C:/Windows/Fonts/msyh.ttc'
]


def _register_chinese_font():
    for path in CHINESE_FONT_PATHS:
        try:
            pdfmetrics.registerFont(TTFont('ChineseFont', path))
            return True
        except Exception:
            continue
    return False


FONT_AVAILABLE = _register_chinese_font()
FONT_NAME = 'ChineseFont' if FONT_AVAILABLE else 'Helvetica'


def _create_style():
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Title'],
        fontName=FONT_NAME,
        fontSize=20,
        leading=28,
        alignment=1,
        spaceAfter=20
    )
    
    h2_style = ParagraphStyle(
        'H2Style',
        parent=styles['Heading2'],
        fontName=FONT_NAME,
        fontSize=14,
        leading=20,
        spaceBefore=15,
        spaceAfter=10,
        textColor=colors.HexColor('#1565C0')
    )
    
    h3_style = ParagraphStyle(
        'H3Style',
        parent=styles['Heading3'],
        fontName=FONT_NAME,
        fontSize=12,
        leading=16,
        spaceBefore=10,
        spaceAfter=6,
        textColor=colors.HexColor('#1976D2')
    )
    
    normal_style = ParagraphStyle(
        'NormalCN',
        parent=styles['Normal'],
        fontName=FONT_NAME,
        fontSize=10,
        leading=16,
        spaceAfter=6
    )
    
    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName=FONT_NAME,
        fontSize=9,
        leading=12,
        alignment=1,
        textColor=colors.white
    )
    
    return {
        'title': title_style,
        'h2': h2_style,
        'h3': h3_style,
        'normal': normal_style,
        'table_header': table_header_style
    }


def _fig_to_image(fig, width_cm: float = 16) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close(fig)
    
    original_width, original_height = fig.get_size_inches()
    aspect_ratio = original_height / original_width
    
    width_points = width_cm * cm
    height_points = width_points * aspect_ratio
    
    return Image(buf, width=width_points, height=height_points)


def generate_zone_compliance_chart(compliance_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if compliance_df.empty:
        ax.text(0.5, 0.5, '暂无功能区评价数据', ha='center', va='center',
                transform=ax.transAxes, fontsize=14, color='gray')
        return fig
    
    zones = compliance_df['功能区名称'].values
    compliance_rates = compliance_df['达标比例(%)'].values
    
    bar_colors = []
    for rate in compliance_rates:
        if rate >= 95:
            bar_colors.append('#4CAF50')
        elif rate >= 80:
            bar_colors.append('#FFC107')
        else:
            bar_colors.append('#F44336')
    
    bars = ax.bar(range(len(zones)), compliance_rates, color=bar_colors, edgecolor='white', linewidth=0.5)
    
    ax.set_xticks(range(len(zones)))
    ax.set_xticklabels(zones, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('达标比例 (%)', fontsize=11)
    ax.set_title('各功能区噪声达标率', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylim(0, 110)
    ax.axhline(y=90, color='#FF9800', linestyle='--', linewidth=1, alpha=0.7, label='90%目标线')
    ax.legend(loc='upper right', fontsize=9)
    
    for bar, rate in zip(bars, compliance_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    
    fig.tight_layout()
    return fig


def generate_area_distribution_pie(area_stats: Dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    
    labels = []
    values = []
    pie_colors = []
    
    for level in NOISE_LEVEL_RANGES:
        key = level['label']
        if key in area_stats and area_stats[key]['area_km2'] > 0:
            labels.append(key)
            values.append(area_stats[key]['area_km2'])
            pie_colors.append(level['color'])
    
    if not values:
        ax.text(0.5, 0.5, '暂无数据', ha='center', va='center',
                transform=ax.transAxes, fontsize=14, color='gray')
        return fig
    
    explode = [0.05] * len(values)
    
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, colors=pie_colors, autopct='%1.1f%%',
        explode=explode, startangle=90, pctdistance=0.85,
        textprops={'fontsize': 9}
    )
    
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')
    
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    ax.add_artist(centre_circle)
    
    ax.set_title('噪声等级面积分布', fontsize=13, fontweight='bold', pad=15)
    ax.axis('equal')
    
    total_area = sum(values)
    ax.text(0, 0, f'总面积\n{total_area:.2f} km²', ha='center', va='center',
            fontsize=11, fontweight='bold', color='#333')
    
    fig.tight_layout()
    return fig


def generate_heatmap_image(z_grid: np.ndarray, grid_lon: np.ndarray, grid_lat: np.ndarray) -> plt.Figure:
    from contour_utils import create_noise_colormap
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    cmap = create_noise_colormap()
    im = ax.imshow(z_grid, extent=[grid_lon[0], grid_lon[-1], grid_lat[0], grid_lat[-1]],
                   origin='lower', cmap=cmap, vmin=40, vmax=85, aspect='auto')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('等效连续声级 Leq [dB(A)]', fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    
    levels = list(range(45, 86, 5))
    cs = ax.contour(z_grid, levels=levels, extent=[grid_lon[0], grid_lon[-1], grid_lat[0], grid_lat[-1]],
                    origin='lower', colors='white', linewidths=0.8, alpha=0.7)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%d', alpha=0.9)
    
    ax.set_xlabel('经度', fontsize=11)
    ax.set_ylabel('纬度', fontsize=11)
    ax.set_title('噪声空间分布图', fontsize=13, fontweight='bold', pad=15)
    ax.tick_params(labelsize=9)
    
    fig.tight_layout()
    return fig


def generate_weekly_chart(weekly_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    
    if weekly_df.empty:
        ax.text(0.5, 0.5, '暂无周变化数据', ha='center', va='center',
                transform=ax.transAxes, fontsize=14, color='gray')
        return fig
    
    weekdays = weekly_df['weekday_name'].values
    means = weekly_df['mean'].values
    
    bar_colors = ['#2196F3'] * 5 + ['#4CAF50'] * 2
    bar_colors = bar_colors[:len(weekdays)]
    
    bars = ax.bar(range(len(weekdays)), means, color=bar_colors,
                  edgecolor='white', linewidth=0.5, alpha=0.85)
    
    ax.set_xticks(range(len(weekdays)))
    ax.set_xticklabels(weekdays, fontsize=10)
    ax.set_ylabel('平均 Leq [dB(A)]', fontsize=11)
    ax.set_title('一周噪声变化模式', fontsize=13, fontweight='bold', pad=15)
    
    y_min = min(means) - 3 if len(means) > 0 else 0
    y_max = max(means) + 3 if len(means) > 0 else 70
    ax.set_ylim(y_min, y_max)
    
    for bar, mean_val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{mean_val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    overall_mean = np.mean(means) if len(means) > 0 else 0
    ax.axhline(y=overall_mean, color='#FF5722', linestyle='--', linewidth=1.2,
               alpha=0.8, label=f'周均值 {overall_mean:.1f} dB')
    ax.legend(loc='upper right', fontsize=9)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    
    fig.tight_layout()
    return fig


def generate_hourly_chart(hourly_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 5))
    
    if hourly_df.empty:
        ax.text(0.5, 0.5, '暂无24小时变化数据', ha='center', va='center',
                transform=ax.transAxes, fontsize=14, color='gray')
        return fig
    
    hours = hourly_df['hour'].values
    means = hourly_df['mean'].values
    
    ax.plot(hours, means, 'o-', color='#1976D2', linewidth=2, markersize=6,
            markerfacecolor='white', markeredgewidth=2, label='平均 Leq')
    
    peak_mask = hourly_df.get('is_peak', pd.Series([False] * len(hours))).values
    if np.any(peak_mask):
        peak_hours = hours[peak_mask]
        peak_means = means[peak_mask]
        ax.scatter(peak_hours, peak_means, s=120, color='#F44336', marker='*',
                   zorder=5, label=f'高峰时段 (n={len(peak_hours)}h)')
    
    daily_mean = float(hourly_df.get('daily_mean', np.mean(means)).iloc[0]) if len(hourly_df) > 0 else 0
    threshold = float(hourly_df.get('threshold', daily_mean + 5).iloc[0]) if len(hourly_df) > 0 else daily_mean + 5
    
    ax.axhline(y=daily_mean, color='#4CAF50', linestyle='--', linewidth=1,
               alpha=0.8, label=f'日均值 {daily_mean:.1f} dB')
    ax.axhline(y=threshold, color='#FF9800', linestyle=':', linewidth=1,
               alpha=0.8, label=f'高峰阈值 {threshold:.1f} dB')
    
    ax.fill_between(hours, means, alpha=0.15, color='#1976D2')
    
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel('时刻 (小时)', fontsize=11)
    ax.set_ylabel('平均 Leq [dB(A)]', fontsize=11)
    ax.set_title('24小时噪声变化曲线', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlim(-0.5, 23.5)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    
    fig.tight_layout()
    return fig


def generate_monthly_trend_chart(monthly_df: pd.DataFrame, trend_info: Dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 5))
    
    if monthly_df.empty:
        ax.text(0.5, 0.5, '暂无月度趋势数据', ha='center', va='center',
                transform=ax.transAxes, fontsize=14, color='gray')
        return fig
    
    labels = monthly_df['year_month_str'].values
    means = monthly_df['mean'].values
    x = np.arange(len(labels))
    
    ax.plot(x, means, 'o-', color='#7B1FA2', linewidth=2, markersize=7,
            markerfacecolor='white', markeredgewidth=2, label='月均 Leq')
    
    if trend_info.get('predicted'):
        predicted = np.array(trend_info['predicted'])
        if len(predicted) == len(x):
            trend_text = trend_info.get('trend', '')
            trend_color = trend_info.get('trend_color', '#666')
            ax.plot(x, predicted, '--', color=trend_color, linewidth=1.8, alpha=0.9,
                    label=f'趋势线 ({trend_text})')
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('月均 Leq [dB(A)]', fontsize=11)
    ax.set_title('月度噪声趋势分析', fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=9)
    
    if 'monthly_change_dB' in trend_info:
        change = trend_info['monthly_change_dB']
        if abs(change) >= 0.05:
            direction = '上升' if change > 0 else '下降'
            ax.text(0.02, 0.98, f'月变化率: {change:+.2f} dB/月\n{direction}趋势',
                    transform=ax.transAxes, fontsize=10, va='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=trend_info.get('trend_color', '#90CAF9'),
                              alpha=0.3))
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    
    fig.tight_layout()
    return fig


def generate_compliance_table(compliance_df: pd.DataFrame, styles: Dict) -> Table:
    if compliance_df.empty:
        data = [['暂无功能区评价数据']]
        t = Table(data, colWidths=[160 * mm])
        return t
    
    headers = [styles['table_header'](col) for col in compliance_df.columns]
    
    table_data = [headers]
    for _, row in compliance_df.iterrows():
        table_data.append([str(val) for val in row.values])
    
    n_cols = len(compliance_df.columns)
    col_widths = [160 * mm / n_cols] * n_cols
    
    t = Table(table_data, colWidths=col_widths, repeatRows=1)
    
    style_commands = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1565C0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, -1), FONT_NAME),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F5F5F5')]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]
    
    for i in range(1, len(table_data)):
        try:
            rate_col = list(compliance_df.columns).index('达标比例(%)')
            rate = float(table_data[i][rate_col])
            if rate < 80:
                style_commands.append(('BACKGROUND', (rate_col, i), (rate_col, i), colors.HexColor('#FFCDD2')))
                style_commands.append(('TEXTCOLOR', (rate_col, i), (rate_col, i), colors.HexColor('#C62828')))
            elif rate < 95:
                style_commands.append(('BACKGROUND', (rate_col, i), (rate_col, i), colors.HexColor('#FFECB3')))
        except (ValueError, IndexError):
            pass
    
    t.setStyle(TableStyle(style_commands))
    return t


def generate_report_pdf(
    compliance_df: pd.DataFrame,
    area_stats: Dict,
    z_grid: Optional[np.ndarray],
    grid_lon: Optional[np.ndarray],
    grid_lat: Optional[np.ndarray],
    weekly_df: Optional[pd.DataFrame],
    hourly_df: Optional[pd.DataFrame],
    monthly_df: Optional[pd.DataFrame],
    trend_info: Optional[Dict],
    source_analysis: Optional[List[Dict]],
    recommendations: Optional[List[str]] = None
) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=20 * mm, bottomMargin=20 * mm)
    
    styles = _create_style()
    elements = []
    
    elements.append(Paragraph('城市声环境质量评估报告', styles['title']))
    elements.append(Paragraph(f'报告生成日期: {datetime.now().strftime("%Y年%m月%d日 %H:%M")}', styles['normal']))
    elements.append(Spacer(1, 10))
    
    elements.append(Paragraph('一、监测概况', styles['h2']))
    
    stats = get_data_statistics()
    overview_data = [
        ['监测站点数量', f"{stats['station_count']} 个"],
        ['有效监测数据', f"{stats['valid_measurements']} 条"],
        ['数据有效率', f"{stats['data_efficiency']:.1f} %"],
        ['监测覆盖天数', f"{stats['days_count']} 天"],
    ]
    if stats['time_start'] and stats['time_end']:
        overview_data.append(['监测时段', f"{stats['time_start']} 至 {stats['time_end']}"])
    
    overview_table = Table(overview_data, colWidths=[50 * mm, 100 * mm])
    overview_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), FONT_NAME),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#E3F2FD')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
    ]))
    elements.append(overview_table)
    elements.append(Spacer(1, 15))
    
    elements.append(Paragraph('二、各功能区达标评价', styles['h2']))
    
    elements.append(Paragraph('2.1 达标率汇总表', styles['h3']))
    comp_table = generate_compliance_table(compliance_df, styles)
    elements.append(comp_table)
    elements.append(Spacer(1, 10))
    
    elements.append(Paragraph('2.2 各功能区达标率对比', styles['h3']))
    if not compliance_df.empty:
        fig1 = generate_zone_compliance_chart(compliance_df)
        elements.append(_fig_to_image(fig1))
    elements.append(Spacer(1, 10))
    
    elements.append(Paragraph('三、超标统计分析', styles['h2']))
    
    if area_stats:
        exceed_key = '超标(>70dB)'
        if exceed_key in area_stats:
            exceed_area = area_stats[exceed_key]['area_km2']
            exceed_pct = area_stats[exceed_key]['percentage']
            total_area = area_stats.get('_total_area_km2', 0)
            
            stat_text = f'''
            评估区域总面积: <b>{total_area:.4f} km²</b><br/>
            超标区域面积(>70dB): <b>{exceed_area:.4f} km²</b><br/>
            超标面积占比: <b>{exceed_pct:.2f}%</b>
            '''
            elements.append(Paragraph(stat_text, styles['normal']))
        
        elements.append(Paragraph('3.1 噪声等级面积分布', styles['h3']))
        fig2 = generate_area_distribution_pie(area_stats)
        elements.append(_fig_to_image(fig2, width_cm=14))
    elements.append(Spacer(1, 10))
    
    elements.append(PageBreak())
    elements.append(Paragraph('四、噪声空间分布', styles['h2']))
    
    if z_grid is not None and grid_lon is not None and grid_lat is not None:
        fig3 = generate_heatmap_image(z_grid, grid_lon, grid_lat)
        elements.append(_fig_to_image(fig3, width_cm=16))
    else:
        elements.append(Paragraph('暂无空间插值数据', styles['normal']))
    elements.append(Spacer(1, 10))
    
    elements.append(Paragraph('五、时间维度分析', styles['h2']))
    
    elements.append(Paragraph('5.1 一周变化模式', styles['h3']))
    if weekly_df is not None:
        fig4 = generate_weekly_chart(weekly_df)
        elements.append(_fig_to_image(fig4, width_cm=16))
    elements.append(Spacer(1, 8))
    
    elements.append(Paragraph('5.2 24小时变化曲线', styles['h3']))
    if hourly_df is not None:
        fig5 = generate_hourly_chart(hourly_df)
        elements.append(_fig_to_image(fig5, width_cm=16))
    elements.append(Spacer(1, 8))
    
    elements.append(Paragraph('5.3 月度变化趋势', styles['h3']))
    if monthly_df is not None:
        fig6 = generate_monthly_trend_chart(monthly_df, trend_info or {})
        elements.append(_fig_to_image(fig6, width_cm=16))
    elements.append(Spacer(1, 10))
    
    elements.append(PageBreak())
    elements.append(Paragraph('六、声源识别分析', styles['h2']))
    
    if source_analysis:
        for station_src in source_analysis:
            station_name = station_src.get('station_name', '未知站点')
            src = station_src.get('source', {})
            primary = src.get('primary_source_name', '未知')
            conf = src.get('confidence', 0)
            
            elements.append(Paragraph(f'<b>{station_name}</b>', styles['h3']))
            source_text = f'''
            主要噪声源: <b>{primary}</b> (置信度: {conf:.1f}%)<br/>
            平均 Leq: {station_src.get('avg_leq', 'N/A')} dB(A)
            '''
            elements.append(Paragraph(source_text, styles['normal']))
            elements.append(Spacer(1, 5))
    else:
        elements.append(Paragraph('暂无声源识别数据', styles['normal']))
    elements.append(Spacer(1, 10))
    
    elements.append(Paragraph('七、改善建议', styles['h2']))
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            elements.append(Paragraph(f'{i}. {rec}', styles['normal']))
    else:
        default_recs = [
            '加强重点超标区域的噪声监测频次',
            '针对交通噪声，可考虑设置声屏障或优化交通组织',
            '对工业噪声源进行隔声降噪改造',
            '加强施工噪声管理，严格控制夜间施工',
            '建立噪声环境质量预警机制',
            '定期开展噪声环境质量评估'
        ]
        for i, rec in enumerate(default_recs, 1):
            elements.append(Paragraph(f'{i}. {rec}', styles['normal']))
    
    elements.append(Spacer(1, 30))
    elements.append(Paragraph('— 报告结束 —', styles['normal']))
    
    doc.build(elements)
    
    return buf.getvalue()


def generate_radar_chart(comparison_data: List[Dict], group_colors: Dict[str, str]) -> plt.Figure:
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    
    categories = ['参与站点数', '持续时长', '平均峰值Leq', '频谱相似度', '定位不确定度']
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    max_values = {
        '参与站点数': 10,
        '持续时长': 60,
        '平均峰值Leq': 100,
        '频谱相似度': 1,
        '定位不确定度': 500
    }
    
    for data in comparison_data:
        gid = data['group_id']
        color = group_colors.get(gid, '#333333')
        
        values = [
            min(data['num_stations'] / max_values['参与站点数'], 1),
            min(data['duration_min'] / max_values['持续时长'], 1),
            min(data['avg_peak_leq'] / max_values['平均峰值Leq'], 1),
            data['spectrum_similarity'],
            min(data['uncertainty'] / max_values['定位不确定度'], 1) if data['uncertainty'] else 0
        ]
        values += values[:1]
        
        line_width = 3 if data.get('highlighted', False) else 2
        ax.plot(angles, values, 'o-', linewidth=line_width, color=color, label=gid, markersize=6)
        ax.fill(angles, values, color=color, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10, fontname=FONT_NAME)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9, prop={'family': FONT_NAME})
    ax.set_title('协同事件组多维度对比雷达图', fontsize=13, fontweight='bold', pad=20, fontname=FONT_NAME)
    
    fig.tight_layout()
    return fig


def generate_comparison_summary(comparison_data: List[Dict]) -> str:
    if len(comparison_data) < 2:
        return "请至少选择2个事件组进行对比分析。"
    
    gids = [d['group_id'] for d in comparison_data]
    summary_parts = []
    
    station_score_best = max(comparison_data, key=lambda x: x['num_stations'])
    duration_best = max(comparison_data, key=lambda x: x['duration_min'])
    leq_best = max(comparison_data, key=lambda x: x['avg_peak_leq'])
    sim_best = max(comparison_data, key=lambda x: x['spectrum_similarity'])
    
    uncertainty_data = [d for d in comparison_data if d['uncertainty'] is not None]
    if uncertainty_data:
        uncertainty_best = min(uncertainty_data, key=lambda x: x['uncertainty'])
        summary_parts.append(f"{uncertainty_best['group_id']}号事件组定位精度最高（不确定度{uncertainty_best['uncertainty']:.0f}m）")
    
    summary_parts.append(f"{station_score_best['group_id']}号事件组站点覆盖度最广（{station_score_best['num_stations']}个站点）")
    summary_parts.append(f"{duration_best['group_id']}号事件组持续时间最长（{duration_best['duration_min']:.1f}分钟）")
    summary_parts.append(f"{leq_best['group_id']}号事件组噪声峰值最高（{leq_best['avg_peak_leq']:.1f}dB）")
    summary_parts.append(f"{sim_best['group_id']}号事件组频谱相似度最高（{sim_best['spectrum_similarity']:.3f}）")
    
    overall_best = max(comparison_data, key=lambda x: x['composite_score'])
    overall_worst = min(comparison_data, key=lambda x: x['composite_score'])
    
    summary = (
        f"本次对比分析共涉及 {len(comparison_data)} 个协同事件组：{', '.join(gids)}。\n\n"
        f"【主要发现】\n"
        + "".join(f"• {p}\n" for p in summary_parts) +
        f"\n【综合评估】\n"
        f"• {overall_best['group_id']}号事件组综合评分最高（{overall_best['composite_score']:.3f}），在整体表现上最为突出。\n"
        f"• {overall_worst['group_id']}号事件组综合评分相对较低（{overall_worst['composite_score']:.3f}），建议重点关注。\n\n"
        f"【建议】\n"
        f"• 优先对综合评分高的事件组进行溯源分析，其协同特征更为显著。\n"
        f"• 对持续时间长、噪声峰值高的事件组，应加强监测并采取相应的降噪措施。"
    )
    
    return summary


def generate_comparison_report_pdf(
    comparison_data: List[Dict],
    group_colors: Dict[str, str],
    radar_fig_bytes: bytes
) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=20 * mm, bottomMargin=20 * mm)
    
    styles = _create_style()
    elements = []
    
    elements.append(Paragraph('协同事件组对比分析报告', styles['title']))
    elements.append(Paragraph(f'报告生成日期: {datetime.now().strftime("%Y年%m月%d日 %H:%M")}', styles['normal']))
    elements.append(Spacer(1, 10))
    
    gids = [d['group_id'] for d in comparison_data]
    elements.append(Paragraph(f'对比事件组: {", ".join(gids)}', styles['normal']))
    elements.append(Spacer(1, 15))
    
    elements.append(Paragraph('一、多维度对比雷达图', styles['h2']))
    
    radar_img = io.BytesIO(radar_fig_bytes)
    radar_img.seek(0)
    elements.append(Image(radar_img, width=14 * cm, height=14 * cm))
    elements.append(Spacer(1, 10))
    
    elements.append(Paragraph('二、对比数据表格', styles['h2']))
    
    table_headers = ['组ID', '参与站点数', '持续时长(min)', '平均峰值Leq(dB)', '频谱相似度', '定位不确定度(m)', '综合评分']
    table_data = [[styles['table_header'](h) for h in table_headers]]
    
    for d in comparison_data:
        row = [
            d['group_id'],
            str(d['num_stations']),
            f"{d['duration_min']:.1f}",
            f"{d['avg_peak_leq']:.1f}",
            f"{d['spectrum_similarity']:.3f}",
            f"{d['uncertainty']:.0f}" if d['uncertainty'] else 'N/A',
            f"{d['composite_score']:.3f}"
        ]
        table_data.append(row)
    
    n_cols = len(table_headers)
    col_widths = [160 * mm / n_cols] * n_cols
    
    t = Table(table_data, colWidths=col_widths, repeatRows=1)
    
    style_commands = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1565C0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, -1), FONT_NAME),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F5F5F5')]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]
    
    t.setStyle(TableStyle(style_commands))
    elements.append(t)
    elements.append(Spacer(1, 15))
    
    elements.append(Paragraph('三、分析总结', styles['h2']))
    
    summary = generate_comparison_summary(comparison_data)
    for line in summary.split('\n'):
        if line.strip():
            elements.append(Paragraph(line.strip(), styles['normal']))
    
    elements.append(Spacer(1, 30))
    elements.append(Paragraph('— 报告结束 —', styles['normal']))
    
    doc.build(elements)
    
    return buf.getvalue()
