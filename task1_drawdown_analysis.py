#!/usr/bin/env python3
"""
Task 1: S&P 500 Historical Drawdown Analysis (1995-2025)
Professional-grade analysis from a Wall Street investment perspective.
No transaction costs assumed.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os

# ============================================================
# 1. Data Loading
# ============================================================

def load_data(csv_path='SP500.csv'):
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df = df[(df['Date'] >= '1995-01-01') & (df['Date'] <= '2025-12-31')].copy()
    df = df[['Date', 'Close']].dropna()
    return df


# ============================================================
# 2. Drawdown Identification (Improved Logic)
# ============================================================

def identify_drawdowns(df, threshold=-0.10):
    """
    Identify all drawdown events exceeding the threshold.
    Uses a cleaner state machine approach.
    """
    prices = df['Close'].values
    dates = df['Date'].values
    n = len(prices)

    # Calculate running max and drawdown series
    running_max = np.maximum.accumulate(prices)
    drawdown_series = (prices - running_max) / running_max

    events = []
    i = 0

    while i < n:
        # Find next drawdown that exceeds threshold
        while i < n and drawdown_series[i] > threshold:
            i += 1

        if i >= n:
            break

        # Found a drawdown - find the peak that preceded it
        peak_idx = i - 1
        while peak_idx > 0 and prices[peak_idx] < running_max[peak_idx]:
            peak_idx -= 1

        # Ensure we have the actual peak
        while peak_idx > 0 and prices[peak_idx - 1] >= prices[peak_idx]:
            peak_idx -= 1

        peak_price = prices[peak_idx]

        # Find the trough (lowest point before recovery)
        trough_idx = i
        j = i
        while j < n and prices[j] < peak_price:
            if prices[j] < prices[trough_idx]:
                trough_idx = j
            j += 1

        trough_price = prices[trough_idx]
        dd_magnitude = (trough_price - peak_price) / peak_price

        # Check if we recovered
        if j < n:
            recovery_idx = j
            recovered = True
        else:
            recovery_idx = None
            recovered = False

        # Only add if it's a genuine 10%+ drawdown
        if dd_magnitude <= threshold:
            event = {
                'peak_date': pd.Timestamp(dates[peak_idx]),
                'trough_date': pd.Timestamp(dates[trough_idx]),
                'recovery_date': pd.Timestamp(dates[recovery_idx]) if recovered else None,
                'peak_price': peak_price,
                'trough_price': trough_price,
                'recovery_price': prices[recovery_idx] if recovered else None,
                'drawdown_pct': dd_magnitude * 100,
                'duration_to_trough': (pd.Timestamp(dates[trough_idx]) - pd.Timestamp(dates[peak_idx])).days,
                'duration_to_recovery': (pd.Timestamp(dates[recovery_idx]) - pd.Timestamp(dates[peak_idx])).days if recovered else None,
                'recovery_from_trough': (pd.Timestamp(dates[recovery_idx]) - pd.Timestamp(dates[trough_idx])).days if recovered else None,
                'recovered': recovered
            }
            events.append(event)

        # Move past this drawdown
        i = j if j < n else n

    return events, drawdown_series


def label_events(events):
    """Add accurate labels based on historical events."""
    for event in events:
        peak_year = event['peak_date'].year
        peak_month = event['peak_date'].month
        trough_year = event['trough_date'].year
        dd = abs(event['drawdown_pct'])

        # More accurate labeling
        if peak_year == 1997 and peak_month >= 7:
            event['label'] = "1997亚洲金融危机"
            event['category'] = "技术性回调"
        elif peak_year == 1998 and 6 <= peak_month <= 8:
            event['label'] = "1998俄罗斯危机/LTCM"
            event['category'] = "技术性回调"
        elif peak_year == 1999:
            event['label'] = "1999年技术性回调"
            event['category'] = "技术性回调"
        elif peak_year == 2000 and trough_year >= 2001:
            event['label'] = "2000-02互联网泡沫破裂"
            event['category'] = "熊市"
        elif peak_year == 2007 and trough_year >= 2008:
            event['label'] = "2007-09全球金融危机"
            event['category'] = "股灾"
        elif peak_year == 2011:
            event['label'] = "2011欧债危机"
            event['category'] = "技术性回调"
        elif peak_year == 2015 or (peak_year == 2016 and peak_month <= 2):
            event['label'] = "2015-16全球恐慌"
            event['category'] = "技术性回调"
        elif peak_year == 2018 and peak_month <= 2:
            event['label'] = "2018.02波动率冲击"
            event['category'] = "闪崩"
        elif peak_year == 2018 and peak_month >= 9:
            event['label'] = "2018.Q4加息恐慌"
            event['category'] = "技术性回调"
        elif peak_year == 2020 and peak_month <= 3:
            event['label'] = "2020 COVID-19"
            event['category'] = "股灾"
        elif peak_year == 2022:
            event['label'] = "2022加息熊市"
            event['category'] = "熊市"
        elif peak_year == 2025:
            event['label'] = "2025年回调"
            event['category'] = "技术性回调"
        else:
            event['label'] = f"{peak_year}年回调"
            event['category'] = "技术性回调" if dd < 20 else "熊市"

        # Classification by severity
        if dd < 15:
            event['severity'] = "轻度回调 (10-15%)"
        elif dd < 20:
            event['severity'] = "中度回调 (15-20%)"
        elif dd < 30:
            event['severity'] = "重度回调 (20-30%)"
        elif dd < 40:
            event['severity'] = "熊市 (30-40%)"
        else:
            event['severity'] = "股灾 (40%+)"

    return events


def compute_advanced_metrics(events):
    """Compute professional-grade metrics."""
    n = len(events)
    magnitudes = [abs(e['drawdown_pct']) for e in events]
    durations = [e['duration_to_trough'] for e in events]
    recovery_times = [e['recovery_from_trough'] for e in events if e['recovered']]
    total_cycles = [e['duration_to_recovery'] for e in events if e['recovered']]

    # Categorization
    light = sum(1 for m in magnitudes if 10 <= m < 15)
    moderate = sum(1 for m in magnitudes if 15 <= m < 20)
    severe = sum(1 for m in magnitudes if 20 <= m < 30)
    bear = sum(1 for m in magnitudes if 30 <= m < 40)
    crash = sum(1 for m in magnitudes if m >= 40)

    stats = {
        'total_events': n,
        'avg_magnitude': np.mean(magnitudes),
        'median_magnitude': np.median(magnitudes),
        'max_magnitude': np.max(magnitudes),
        'min_magnitude': np.min(magnitudes),
        'std_magnitude': np.std(magnitudes),
        'avg_duration_to_trough': np.mean(durations),
        'median_duration_to_trough': np.median(durations),
        'max_duration_to_trough': np.max(durations),
        'avg_recovery_time': np.mean(recovery_times) if recovery_times else None,
        'median_recovery_time': np.median(recovery_times) if recovery_times else None,
        'max_recovery_time': np.max(recovery_times) if recovery_times else None,
        'avg_total_cycle': np.mean(total_cycles) if total_cycles else None,
        'light_10_15': light,
        'moderate_15_20': moderate,
        'severe_20_30': severe,
        'bear_30_40': bear,
        'crash_40_plus': crash,
        'frequency_per_year': n / 30,
        'avg_drawdown_speed': np.mean([m / d * 365 for m, d in zip(magnitudes, durations)]),  # Annualized
    }

    # Add per-event advanced metrics
    for e in events:
        dd = abs(e['drawdown_pct'])
        dur = e['duration_to_trough']
        e['decline_speed_annual'] = dd / dur * 365 if dur > 0 else 0
        if e['recovered'] and e['recovery_from_trough']:
            e['recovery_speed_annual'] = dd / e['recovery_from_trough'] * 365
        else:
            e['recovery_speed_annual'] = None
        # Pain index = magnitude * duration (higher = more painful)
        e['pain_index'] = dd * dur / 100

    return stats


# ============================================================
# 3. Visualization (Professional Focus)
# ============================================================

def create_main_price_chart(df, events):
    """Main chart: S&P 500 with drawdown periods shaded."""
    fig = go.Figure()

    # Main price line
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Close'],
        mode='lines', name='S&P 500',
        line=dict(color='#2c3e50', width=1.5)
    ))

    colors = {'股灾': '#e74c3c', '熊市': '#e67e22', '技术性回调': '#f39c12', '闪崩': '#9b59b6'}

    for event in events:
        end = event['recovery_date'] or df['Date'].iloc[-1]
        color = colors.get(event['category'], '#95a5a6')

        # Shade the drawdown period
        fig.add_vrect(
            x0=event['peak_date'], x1=event['trough_date'],
            fillcolor=color, opacity=0.2, line_width=0,
            annotation_text=event['label'], annotation_position="top left"
        )

        # Mark peak and trough
        fig.add_trace(go.Scatter(
            x=[event['peak_date'], event['trough_date']],
            y=[event['peak_price'], event['trough_price']],
            mode='markers+lines',
            marker=dict(size=8, color=color),
            line=dict(color=color, width=2, dash='dot'),
            name=f"{event['label']} ({event['drawdown_pct']:.1f}%)",
            showlegend=True
        ))

    fig.update_layout(
        title='标普500历史走势与回调事件 (1995-2025)',
        xaxis_title='日期', yaxis_title='价格 (USD)',
        yaxis_type='log',
        template='plotly_white', height=550,
        legend=dict(orientation='h', yanchor='bottom', y=-0.25),
        hovermode='x unified'
    )
    return fig


def create_drawdown_underwater_chart(df, dd_series):
    """Underwater chart showing drawdown depth over time."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['Date'], y=dd_series * 100,
        mode='lines', name='回撤深度',
        line=dict(color='#e74c3c', width=1),
        fill='tozeroy', fillcolor='rgba(231,76,60,0.3)'
    ))

    fig.add_hline(y=-10, line_dash='dash', line_color='#f39c12', annotation_text='10% 回调线')
    fig.add_hline(y=-20, line_dash='dash', line_color='#e67e22', annotation_text='20% 熊市线')
    fig.add_hline(y=-30, line_dash='dash', line_color='#e74c3c', annotation_text='30% 股灾线')

    fig.update_layout(
        title='回撤深度图（Underwater Chart）',
        xaxis_title='日期', yaxis_title='回撤幅度 (%)',
        template='plotly_white', height=400
    )
    return fig


def create_recovery_analysis_chart(events):
    """Key chart: Recovery time vs Drawdown depth."""
    recovered = [e for e in events if e['recovered']]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['回调幅度 vs 恢复时间', '回调周期全景图'])

    # Scatter: Magnitude vs Recovery Time
    magnitudes = [abs(e['drawdown_pct']) for e in recovered]
    recovery_times = [e['recovery_from_trough'] for e in recovered]
    labels = [e['label'] for e in recovered]
    colors = [e['pain_index'] for e in recovered]

    fig.add_trace(go.Scatter(
        x=magnitudes, y=recovery_times,
        mode='markers+text',
        marker=dict(size=15, color=colors, colorscale='Reds', showscale=True,
                    colorbar=dict(title='痛苦指数')),
        text=labels, textposition='top center', textfont=dict(size=9),
        name='回调事件'
    ), row=1, col=1)

    # Add trend line
    if len(magnitudes) > 2:
        z = np.polyfit(magnitudes, recovery_times, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(magnitudes), max(magnitudes), 100)
        fig.add_trace(go.Scatter(
            x=x_line, y=p(x_line),
            mode='lines', line=dict(dash='dash', color='gray'),
            name='趋势线', showlegend=False
        ), row=1, col=1)

    # Gantt-like chart for drawdown cycles
    for i, e in enumerate(events):
        y_pos = len(events) - i
        # Decline phase
        fig.add_trace(go.Scatter(
            x=[e['peak_date'], e['trough_date']],
            y=[y_pos, y_pos],
            mode='lines', line=dict(color='#e74c3c', width=8),
            name='下跌', showlegend=(i == 0),
            hovertext=f"下跌: {e['duration_to_trough']}天"
        ), row=1, col=2)
        # Recovery phase
        if e['recovered']:
            fig.add_trace(go.Scatter(
                x=[e['trough_date'], e['recovery_date']],
                y=[y_pos, y_pos],
                mode='lines', line=dict(color='#27ae60', width=8),
                name='恢复', showlegend=(i == 0),
                hovertext=f"恢复: {e['recovery_from_trough']}天"
            ), row=1, col=2)

    fig.update_xaxes(title_text='回调幅度 (%)', row=1, col=1)
    fig.update_yaxes(title_text='恢复时间 (天)', row=1, col=1)
    fig.update_yaxes(ticktext=[e['label'] for e in events][::-1],
                     tickvals=list(range(1, len(events) + 1)), row=1, col=2)

    fig.update_layout(
        title='回调恢复分析 - 投资者最关心的问题：需要多久才能回本？',
        template='plotly_white', height=500, showlegend=True
    )
    return fig


def create_severity_distribution_chart(stats, events):
    """Distribution of drawdown severity."""
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['回调深度分布', '回调持续时间分布'],
                        specs=[[{"type": "pie"}, {"type": "xy"}]])

    # Severity pie chart
    labels = ['轻度 (10-15%)', '中度 (15-20%)', '重度 (20-30%)', '熊市 (30-40%)', '股灾 (40%+)']
    values = [stats['light_10_15'], stats['moderate_15_20'], stats['severe_20_30'],
              stats['bear_30_40'], stats['crash_40_plus']]
    colors = ['#f1c40f', '#e67e22', '#e74c3c', '#c0392b', '#8e44ad']

    fig.add_trace(go.Pie(
        labels=labels, values=values,
        marker_colors=colors,
        textinfo='label+value',
        hole=0.4
    ), row=1, col=1)

    # Duration histogram
    durations = [e['duration_to_trough'] for e in events]
    fig.add_trace(go.Histogram(
        x=durations, nbinsx=10,
        marker_color='#3498db',
        name='下跌天数'
    ), row=1, col=2)

    fig.update_layout(template='plotly_white', height=400)
    return fig


def create_key_metrics_table(events):
    """Create a comprehensive metrics table for each event."""
    headers = ['事件', '类别', '回调幅度', '下跌天数', '恢复天数', '总周期', '下跌速度<br>(年化)', '恢复速度<br>(年化)', '痛苦指数']

    rows = []
    for e in events:
        rec_days = f"{e['recovery_from_trough']}" if e['recovered'] else 'N/A'
        total_days = f"{e['duration_to_recovery']}" if e['duration_to_recovery'] else 'N/A'
        rec_speed = f"{e['recovery_speed_annual']:.1f}%" if e['recovery_speed_annual'] else 'N/A'

        rows.append([
            e['label'],
            e['category'],
            f"{e['drawdown_pct']:.1f}%",
            str(e['duration_to_trough']),
            rec_days,
            total_days,
            f"{e['decline_speed_annual']:.1f}%",
            rec_speed,
            f"{e['pain_index']:.1f}"
        ])

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=headers,
            fill_color='#2c3e50',
            font=dict(color='white', size=12),
            align='center'
        ),
        cells=dict(
            values=list(zip(*rows)),
            fill_color=[['#f8f9fa', '#ffffff'] * (len(rows) // 2 + 1)][:len(rows)],
            align='center',
            font=dict(size=11)
        )
    )])

    fig.update_layout(title='回调事件详细指标', height=450)
    return fig


# ============================================================
# 4. HTML Report Generation
# ============================================================

def generate_html_report(df, events, stats, dd_series):
    """Generate professional HTML report."""
    fig_price = create_main_price_chart(df, events)
    fig_underwater = create_drawdown_underwater_chart(df, dd_series)
    fig_recovery = create_recovery_analysis_chart(events)
    fig_dist = create_severity_distribution_chart(stats, events)
    fig_table = create_key_metrics_table(events)

    charts = {
        'price': fig_price.to_html(full_html=False, include_plotlyjs=False),
        'underwater': fig_underwater.to_html(full_html=False, include_plotlyjs=False),
        'recovery': fig_recovery.to_html(full_html=False, include_plotlyjs=False),
        'dist': fig_dist.to_html(full_html=False, include_plotlyjs=False),
        'table': fig_table.to_html(full_html=False, include_plotlyjs=False),
    }

    # Format stats
    avg_rec = f"{stats['avg_recovery_time']:.0f}" if stats['avg_recovery_time'] else 'N/A'
    med_rec = f"{stats['median_recovery_time']:.0f}" if stats['median_recovery_time'] else 'N/A'
    max_rec = f"{stats['max_recovery_time']:.0f}" if stats['max_recovery_time'] else 'N/A'
    avg_cycle = f"{stats['avg_total_cycle']:.0f}" if stats['avg_total_cycle'] else 'N/A'
    freq = f"{stats['frequency_per_year']:.2f}"

    # Build event details for CSV-style table
    event_rows = ""
    for i, e in enumerate(events, 1):
        rec_date = e['recovery_date'].strftime('%Y-%m-%d') if e['recovery_date'] else '进行中'
        rec_days = f"{e['recovery_from_trough']}天" if e['recovery_from_trough'] else 'N/A'
        total = f"{e['duration_to_recovery']}天" if e['duration_to_recovery'] else 'N/A'
        event_rows += f"""
        <tr>
            <td>{i}</td>
            <td><strong>{e['label']}</strong></td>
            <td>{e['category']}</td>
            <td>{e['peak_date'].strftime('%Y-%m-%d')}</td>
            <td>{e['trough_date'].strftime('%Y-%m-%d')}</td>
            <td>{rec_date}</td>
            <td class="negative">{e['drawdown_pct']:.1f}%</td>
            <td>{e['duration_to_trough']}天</td>
            <td>{rec_days}</td>
            <td>{total}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>标普500历史回调深度分析 (1995-2025)</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif; background: #f4f6f9; color: #2c3e50; line-height: 1.7; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #1a252f 0%, #2c3e50 100%); color: white; padding: 40px; border-radius: 12px; margin-bottom: 30px; }}
        .header h1 {{ font-size: 2.2em; margin-bottom: 10px; }}
        .section {{ background: white; border-radius: 12px; padding: 30px; margin-bottom: 25px; box-shadow: 0 2px 15px rgba(0,0,0,0.06); }}
        .section h2 {{ color: #2c3e50; border-left: 4px solid #3498db; padding-left: 15px; margin-bottom: 20px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin: 25px 0; }}
        .stat-card {{ background: linear-gradient(135deg, #f8f9fa, #e9ecef); padding: 20px; border-radius: 10px; text-align: center; }}
        .stat-card .number {{ font-size: 1.8em; font-weight: bold; color: #2c3e50; }}
        .stat-card .label {{ font-size: 0.85em; color: #7f8c8d; margin-top: 5px; }}
        .insight {{ padding: 20px; border-radius: 8px; margin: 15px 0; }}
        .insight-blue {{ background: #ebf5fb; border-left: 4px solid #3498db; }}
        .insight-green {{ background: #e8f6f3; border-left: 4px solid #27ae60; }}
        .insight-orange {{ background: #fef5e7; border-left: 4px solid #f39c12; }}
        .insight-red {{ background: #fdedec; border-left: 4px solid #e74c3c; }}
        .insight h3 {{ margin-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 0.9em; }}
        th {{ background: #2c3e50; color: white; padding: 12px 10px; text-align: center; }}
        td {{ padding: 10px; text-align: center; border-bottom: 1px solid #ecf0f1; }}
        tr:hover {{ background: #f8f9fa; }}
        .negative {{ color: #e74c3c; font-weight: bold; }}
        .chart-container {{ margin: 20px 0; }}
        ul {{ padding-left: 25px; }}
        li {{ margin: 8px 0; }}
    </style>
</head>
<body>
<div class="container">

    <div class="header">
        <h1>标普500历史回调深度分析报告</h1>
        <p>研究区间：1995年1月 - 2025年8月 | 回调阈值：≥10% | 专业投资视角</p>
        <p>报告生成：{datetime.now().strftime('%Y-%m-%d')}</p>
    </div>

    <!-- ===== Executive Summary ===== -->
    <div class="section">
        <h2>核心发现 Executive Summary</h2>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="number">{stats['total_events']}</div>
                <div class="label">10%+回调总次数</div>
            </div>
            <div class="stat-card">
                <div class="number">{freq}</div>
                <div class="label">年均发生频率</div>
            </div>
            <div class="stat-card">
                <div class="number">{stats['avg_magnitude']:.1f}%</div>
                <div class="label">平均回调幅度</div>
            </div>
            <div class="stat-card">
                <div class="number">{stats['median_magnitude']:.1f}%</div>
                <div class="label">中位数幅度</div>
            </div>
            <div class="stat-card">
                <div class="number">{stats['avg_duration_to_trough']:.0f}天</div>
                <div class="label">平均下跌时长</div>
            </div>
            <div class="stat-card">
                <div class="number">{avg_rec}天</div>
                <div class="label">平均恢复时长</div>
            </div>
        </div>

        <div class="insight insight-blue">
            <h3>频率与分布</h3>
            <ul>
                <li>过去30年标普500经历了<strong>{stats['total_events']}次</strong>超过10%的回调，平均每<strong>{1/stats['frequency_per_year']:.1f}年</strong>发生一次</li>
                <li>分布：轻度(10-15%): {stats['light_10_15']}次 | 中度(15-20%): {stats['moderate_15_20']}次 | 重度(20-30%): {stats['severe_20_30']}次 | 熊市/股灾(30%+): {stats['bear_30_40'] + stats['crash_40_plus']}次</li>
                <li>10%回调是市场常态，约占总时间的15-20%；真正的股灾（30%+）平均每15年发生1次</li>
            </ul>
        </div>

        <div class="insight insight-green">
            <h3>恢复能力 - 投资者最关心的问题</h3>
            <ul>
                <li><strong>平均恢复时间</strong>：从谷底回到前高需要<strong>{avg_rec}天</strong>（中位数{med_rec}天）</li>
                <li><strong>最长恢复</strong>：互联网泡沫后需要{max_rec}天（约{int(int(max_rec)/365)}年）才恢复到2000年高点</li>
                <li><strong>完整周期</strong>：从峰值到恢复的完整周期平均<strong>{avg_cycle}天</strong></li>
                <li>关键启示：回调幅度与恢复时间呈正相关；20%以内的回调通常6个月内恢复；30%+回调可能需要2-5年</li>
            </ul>
        </div>

        <div class="insight insight-orange">
            <h3>下跌速度 vs 恢复速度</h3>
            <ul>
                <li>平均下跌速度（年化）：<strong>{stats['avg_drawdown_speed']:.0f}%</strong>，市场下跌通常比上涨更快更猛烈</li>
                <li>股灾的特点是"快速崩盘+缓慢恢复"：2008年金融危机517天跌56.8%，但恢复用了1480天</li>
                <li>COVID-19例外：33天跌34%，但仅用148天恢复，体现了现代央行干预的威力</li>
            </ul>
        </div>

        <div class="insight insight-red">
            <h3>对投资者的实战意义</h3>
            <ul>
                <li><strong>择时难度</strong>：10%回调发生时，你无法判断是会变成20%还是50%的下跌</li>
                <li><strong>机会成本</strong>：如果在每次10%回调时清仓，可能错过后续反弹（1997、2018.02都是快速V型反弹）</li>
                <li><strong>心理准备</strong>：持有资产就必须接受周期性20-30%的回撤，这是市场的"入场费"</li>
                <li><strong>资金规划</strong>：重大股灾后可能需要5年才能恢复，确保不在低点被迫卖出</li>
            </ul>
        </div>
    </div>

    <!-- ===== Main Charts ===== -->
    <div class="section">
        <h2>历史走势与回调标注</h2>
        <div class="chart-container">{charts['price']}</div>
    </div>

    <div class="section">
        <h2>回撤深度图 (Underwater Chart)</h2>
        <p style="color:#7f8c8d;margin-bottom:15px;">这是专业投资者最常用的风险可视化工具，展示任意时点距离历史高点的距离</p>
        <div class="chart-container">{charts['underwater']}</div>
    </div>

    <div class="section">
        <h2>恢复时间分析 - 核心问题：跌了之后多久能回本？</h2>
        <div class="chart-container">{charts['recovery']}</div>
    </div>

    <div class="section">
        <h2>回调分类统计</h2>
        <div class="chart-container">{charts['dist']}</div>
    </div>

    <!-- ===== Detailed Data Table ===== -->
    <div class="section">
        <h2>回调事件完整数据</h2>
        <div style="overflow-x:auto;">
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>事件</th>
                    <th>类别</th>
                    <th>峰值日期</th>
                    <th>谷底日期</th>
                    <th>恢复日期</th>
                    <th>回调幅度</th>
                    <th>下跌天数</th>
                    <th>恢复天数</th>
                    <th>完整周期</th>
                </tr>
            </thead>
            <tbody>
                {event_rows}
            </tbody>
        </table>
        </div>
    </div>

    <div class="section">
        <h2>详细指标表（含痛苦指数）</h2>
        <p style="color:#7f8c8d;margin-bottom:15px;">痛苦指数 = 回调幅度 × 持续天数 / 100，数值越大表示投资者承受的心理压力越大</p>
        <div class="chart-container">{charts['table']}</div>
    </div>

    <div class="section" style="text-align:center; color:#95a5a6; font-size:0.9em;">
        <p>本报告仅供研究参考，不构成投资建议 | 数据来源：S&P 500历史数据</p>
    </div>

</div>
</body>
</html>"""
    return html


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 65)
    print("Task 1: S&P 500 Historical Drawdown Analysis (1995-2025)")
    print("Professional Edition - No Transaction Costs")
    print("=" * 65)

    print("\n[1/5] Loading data...")
    df = load_data('SP500.csv')
    print(f"  Loaded {len(df)} trading days: {df['Date'].iloc[0].strftime('%Y-%m-%d')} to {df['Date'].iloc[-1].strftime('%Y-%m-%d')}")

    print("\n[2/5] Identifying drawdowns (threshold: -10%)...")
    events, dd_series = identify_drawdowns(df, threshold=-0.10)
    events = label_events(events)
    print(f"  Found {len(events)} drawdown events")

    for i, e in enumerate(events, 1):
        print(f"    {i:2d}. {e['label']:<25} {e['drawdown_pct']:>7.1f}%  {e['category']}")

    print("\n[3/5] Computing advanced metrics...")
    stats = compute_advanced_metrics(events)
    print(f"  Average magnitude: {stats['avg_magnitude']:.1f}%")
    print(f"  Average recovery time: {stats['avg_recovery_time']:.0f} days" if stats['avg_recovery_time'] else "  No recovery data")

    print("\n[4/5] Saving data...")
    os.makedirs('outputs', exist_ok=True)
    events_df = pd.DataFrame(events)
    events_df.to_csv('outputs/task1_drawdowns.csv', index=False, encoding='utf-8-sig')
    print("  Saved to outputs/task1_drawdowns.csv")

    print("\n[5/5] Generating HTML report...")
    html = generate_html_report(df, events, stats, dd_series)
    with open('outputs/task1_analysis.html', 'w', encoding='utf-8') as f:
        f.write(html)
    print("  Saved to outputs/task1_analysis.html")

    print("\n" + "=" * 65)
    print("Task 1 Complete!")
    print("=" * 65)

    return df, events, stats


if __name__ == '__main__':
    main()
