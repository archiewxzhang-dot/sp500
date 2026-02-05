#!/usr/bin/env python3
"""
Task 1: S&P 500 Historical Drawdown Analysis (1995-2025)
Identifies all 10%+ drawdowns, analyzes them, and generates an interactive HTML report.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime
import os

# ============================================================
# 1. Data Loading and Preparation
# ============================================================

def load_data(csv_path='SP500.csv'):
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    # Filter to 1995-2025
    df = df[(df['Date'] >= '1995-01-01') & (df['Date'] <= '2025-12-31')].copy()
    df = df[['Date', 'Close']].dropna()
    return df


# ============================================================
# 2. Drawdown Identification
# ============================================================

def identify_drawdowns(df, threshold=-0.10):
    """
    Identify all drawdown events where the decline from peak exceeds the threshold.
    Returns a list of drawdown event dicts and a drawdown series.
    """
    prices = df['Close'].values
    dates = df['Date'].values

    # Calculate running maximum and drawdown series
    running_max = np.maximum.accumulate(prices)
    drawdown_series = (prices - running_max) / running_max

    # Identify drawdown events
    events = []
    in_drawdown = False
    peak_idx = 0

    for i in range(len(prices)):
        if prices[i] >= running_max[i]:
            # At or above peak
            if in_drawdown:
                # Drawdown has ended - price recovered to previous peak
                trough_idx = peak_idx + np.argmin(prices[peak_idx:i+1])
                dd_magnitude = (prices[trough_idx] - prices[peak_idx]) / prices[peak_idx]
                if dd_magnitude <= threshold:
                    events.append({
                        'peak_date': pd.Timestamp(dates[peak_idx]),
                        'trough_date': pd.Timestamp(dates[trough_idx]),
                        'recovery_date': pd.Timestamp(dates[i]),
                        'peak_price': prices[peak_idx],
                        'trough_price': prices[trough_idx],
                        'recovery_price': prices[i],
                        'drawdown_pct': dd_magnitude * 100,
                        'duration_to_trough': (pd.Timestamp(dates[trough_idx]) - pd.Timestamp(dates[peak_idx])).days,
                        'duration_to_recovery': (pd.Timestamp(dates[i]) - pd.Timestamp(dates[peak_idx])).days,
                        'recovery_from_trough': (pd.Timestamp(dates[i]) - pd.Timestamp(dates[trough_idx])).days,
                        'recovered': True
                    })
                in_drawdown = False
            peak_idx = i
        else:
            if not in_drawdown:
                current_dd = (prices[i] - running_max[i]) / running_max[i]
                if current_dd <= threshold:
                    in_drawdown = True

    # Handle ongoing drawdown at end of data
    if in_drawdown:
        trough_idx = peak_idx + np.argmin(prices[peak_idx:])
        dd_magnitude = (prices[trough_idx] - prices[peak_idx]) / prices[peak_idx]
        if dd_magnitude <= threshold:
            events.append({
                'peak_date': pd.Timestamp(dates[peak_idx]),
                'trough_date': pd.Timestamp(dates[trough_idx]),
                'recovery_date': None,
                'peak_price': prices[peak_idx],
                'trough_price': prices[trough_idx],
                'recovery_price': None,
                'drawdown_pct': dd_magnitude * 100,
                'duration_to_trough': (pd.Timestamp(dates[trough_idx]) - pd.Timestamp(dates[peak_idx])).days,
                'duration_to_recovery': None,
                'recovery_from_trough': None,
                'recovered': False
            })

    # Merge overlapping / nested drawdowns - keep the deepest in each cluster
    merged = []
    for event in events:
        if merged and event['peak_date'] <= (merged[-1]['recovery_date'] or event['peak_date']):
            # Overlapping with previous event - keep the deeper one
            if event['drawdown_pct'] < merged[-1]['drawdown_pct']:
                # Update the merged event with the deeper trough
                prev = merged[-1]
                prev['trough_date'] = event['trough_date']
                prev['trough_price'] = event['trough_price']
                prev['drawdown_pct'] = (event['trough_price'] - prev['peak_price']) / prev['peak_price'] * 100
                prev['duration_to_trough'] = (event['trough_date'] - prev['peak_date']).days
                if event['recovery_date']:
                    prev['recovery_date'] = event['recovery_date']
                    prev['recovery_price'] = event['recovery_price']
                    prev['duration_to_recovery'] = (event['recovery_date'] - prev['peak_date']).days
                    prev['recovery_from_trough'] = (event['recovery_date'] - event['trough_date']).days
                    prev['recovered'] = True
                else:
                    prev['recovery_date'] = None
                    prev['recovery_price'] = None
                    prev['duration_to_recovery'] = None
                    prev['recovery_from_trough'] = None
                    prev['recovered'] = False
        else:
            merged.append(event.copy())

    return merged, drawdown_series


def label_events(events):
    """Add descriptive labels for known market events."""
    labels = {
        (1998, 7, 1998, 10): "1998年俄罗斯/LTCM危机",
        (2000, 3, 2002, 10): "2000-2002年互联网泡沫",
        (2007, 10, 2009, 3): "2007-2009年全球金融危机",
        (2011, 4, 2011, 10): "2011年欧债危机/美债降级",
        (2015, 5, 2016, 2): "2015-2016年中国股灾/全球恐慌",
        (2018, 9, 2018, 12): "2018年Q4暴跌(加息+贸易战)",
        (2020, 2, 2020, 3): "2020年COVID-19疫情",
        (2022, 1, 2022, 10): "2022年加息周期熊市",
    }

    for event in events:
        peak_year = event['peak_date'].year
        peak_month = event['peak_date'].month
        trough_year = event['trough_date'].year
        trough_month = event['trough_date'].month

        event['label'] = ''
        for (py, pm, ty, tm), label in labels.items():
            if abs(peak_year - py) <= 1 and abs(trough_year - ty) <= 1:
                event['label'] = label
                break

        if not event['label']:
            event['label'] = f"{peak_year}年回调"

    return events


# ============================================================
# 3. Statistical Analysis
# ============================================================

def compute_statistics(events):
    """Compute summary statistics for all drawdown events."""
    n = len(events)
    magnitudes = [abs(e['drawdown_pct']) for e in events]
    durations_to_trough = [e['duration_to_trough'] for e in events]
    recovery_times = [e['recovery_from_trough'] for e in events if e['recovered']]

    # Depth distribution
    depth_10_20 = sum(1 for m in magnitudes if 10 <= m < 20)
    depth_20_30 = sum(1 for m in magnitudes if 20 <= m < 30)
    depth_30_plus = sum(1 for m in magnitudes if m >= 30)

    stats = {
        'total_events': n,
        'avg_magnitude': np.mean(magnitudes),
        'median_magnitude': np.median(magnitudes),
        'max_magnitude': np.max(magnitudes),
        'min_magnitude': np.min(magnitudes),
        'avg_duration_to_trough': np.mean(durations_to_trough),
        'median_duration_to_trough': np.median(durations_to_trough),
        'avg_recovery_time': np.mean(recovery_times) if recovery_times else None,
        'median_recovery_time': np.median(recovery_times) if recovery_times else None,
        'depth_10_20': depth_10_20,
        'depth_20_30': depth_20_30,
        'depth_30_plus': depth_30_plus,
        'frequency_years': 30 / n if n > 0 else None,
    }
    return stats


# ============================================================
# 4. Visualization Functions
# ============================================================

def create_price_chart_with_drawdowns(df, events):
    """S&P 500 price chart with drawdown periods highlighted."""
    fig = go.Figure()

    # Price line
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Close'],
        mode='lines', name='S&P 500',
        line=dict(color='#1f77b4', width=1.5)
    ))

    colors_hex = ['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854',
                  '#ffd92f','#e5c494','#b3b3b3','#fb8072','#80b1d3','#bebada']
    for i, event in enumerate(events):
        end = event['recovery_date'] or df['Date'].iloc[-1]
        mask = (df['Date'] >= event['peak_date']) & (df['Date'] <= end)
        sub = df[mask]
        if len(sub) == 0:
            continue

        color = colors_hex[i % len(colors_hex)]
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        fig.add_trace(go.Scatter(
            x=sub['Date'], y=sub['Close'],
            mode='lines', name=f"{event['label']} ({event['drawdown_pct']:.1f}%)",
            line=dict(color=color, width=2),
            fill='tozeroy',
            fillcolor=f'rgba({r},{g},{b},0.15)'
        ))

        # Mark peak and trough
        fig.add_trace(go.Scatter(
            x=[event['peak_date']], y=[event['peak_price']],
            mode='markers', marker=dict(color='green', size=8, symbol='triangle-up'),
            showlegend=False, hovertext=f"峰值: {event['peak_date'].strftime('%Y-%m-%d')}<br>价格: {event['peak_price']:.2f}"
        ))
        fig.add_trace(go.Scatter(
            x=[event['trough_date']], y=[event['trough_price']],
            mode='markers', marker=dict(color='red', size=8, symbol='triangle-down'),
            showlegend=False, hovertext=f"谷底: {event['trough_date'].strftime('%Y-%m-%d')}<br>价格: {event['trough_price']:.2f}"
        ))

    fig.update_layout(
        title='标普500价格走势图（标注所有>10%回调区间）1995-2025',
        xaxis_title='日期', yaxis_title='价格 (USD)',
        template='plotly_white', height=600, legend=dict(font=dict(size=10)),
        hovermode='x unified'
    )
    return fig


def create_drawdown_series_chart(df, drawdown_series):
    """Drawdown percentage over time."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Date'], y=drawdown_series * 100,
        mode='lines', name='回撤幅度',
        line=dict(color='red', width=1),
        fill='tozeroy', fillcolor='rgba(255,0,0,0.1)'
    ))
    fig.add_hline(y=-10, line_dash='dash', line_color='orange',
                  annotation_text='10%回撤线')
    fig.add_hline(y=-20, line_dash='dash', line_color='red',
                  annotation_text='20%回撤线（熊市）')
    fig.update_layout(
        title='标普500回撤幅度时间序列 (1995-2025)',
        xaxis_title='日期', yaxis_title='回撤幅度 (%)',
        template='plotly_white', height=400
    )
    return fig


def create_magnitude_bar_chart(events):
    """Bar chart of drawdown magnitudes."""
    labels = [e['label'] for e in events]
    magnitudes = [abs(e['drawdown_pct']) for e in events]

    colors = ['#ff9999' if m < 20 else '#ff4444' if m < 30 else '#cc0000' for m in magnitudes]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=magnitudes,
        marker_color=colors,
        text=[f'{m:.1f}%' for m in magnitudes],
        textposition='outside'
    ))
    fig.update_layout(
        title='各次回调幅度分布',
        xaxis_title='回调事件', yaxis_title='回调幅度 (%)',
        template='plotly_white', height=500,
        xaxis_tickangle=-45
    )
    return fig


def create_duration_box_chart(events):
    """Box plot of drawdown durations."""
    durations = [e['duration_to_trough'] for e in events]
    recovery_times = [e['recovery_from_trough'] for e in events if e['recovered']]

    fig = make_subplots(rows=1, cols=2, subplot_titles=['回调持续时间（峰值至谷底）', '恢复时间（谷底至恢复）'])

    fig.add_trace(go.Box(y=durations, name='下跌天数', marker_color='#ff6b6b', boxpoints='all'), row=1, col=1)
    fig.add_trace(go.Box(y=recovery_times, name='恢复天数', marker_color='#51cf66', boxpoints='all'), row=1, col=2)

    fig.update_layout(
        title='回调持续时间与恢复时间箱线图',
        template='plotly_white', height=450, showlegend=False
    )
    fig.update_yaxes(title_text='天数', row=1, col=1)
    fig.update_yaxes(title_text='天数', row=1, col=2)
    return fig


def create_frequency_timeline(events):
    """Timeline showing when drawdowns occurred."""
    fig = go.Figure()

    for i, event in enumerate(events):
        end_date = event['recovery_date'] or pd.Timestamp('2025-08-29')
        mag = abs(event['drawdown_pct'])
        color = '#ff9999' if mag < 20 else '#ff4444' if mag < 30 else '#cc0000'

        fig.add_trace(go.Scatter(
            x=[event['peak_date'], event['trough_date'], end_date],
            y=[0, event['drawdown_pct'], 0 if event['recovered'] else event['drawdown_pct'] * 0.3],
            mode='lines+markers',
            name=event['label'],
            line=dict(color=color, width=2),
            marker=dict(size=6)
        ))

    fig.update_layout(
        title='回调事件频率时间线',
        xaxis_title='日期', yaxis_title='回撤幅度 (%)',
        template='plotly_white', height=400
    )
    return fig


def create_recovery_time_chart(events):
    """Bar chart of recovery times."""
    recovered = [e for e in events if e['recovered']]
    labels = [e['label'] for e in recovered]
    recovery_days = [e['recovery_from_trough'] for e in recovered]
    total_days = [e['duration_to_recovery'] for e in recovered]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=[e['duration_to_trough'] for e in recovered],
        name='下跌天数', marker_color='#ff6b6b',
        text=[f"{d}天" for d in [e['duration_to_trough'] for e in recovered]],
        textposition='inside'
    ))
    fig.add_trace(go.Bar(
        x=labels, y=recovery_days,
        name='恢复天数', marker_color='#51cf66',
        text=[f"{d}天" for d in recovery_days],
        textposition='inside'
    ))

    fig.update_layout(
        title='各次回调的下跌与恢复时间对比',
        xaxis_title='回调事件', yaxis_title='天数',
        barmode='stack',
        template='plotly_white', height=500,
        xaxis_tickangle=-45
    )
    return fig


def create_depth_distribution_pie(stats):
    """Pie chart of drawdown depth distribution."""
    labels = ['10%-20%（中等回调）', '20%-30%（大幅回调）', '30%+（股灾级别）']
    values = [stats['depth_10_20'], stats['depth_20_30'], stats['depth_30_plus']]
    colors = ['#ffd43b', '#ff6b6b', '#cc0000']

    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values,
        marker_colors=colors,
        textinfo='label+value+percent',
        hole=0.3
    )])
    fig.update_layout(
        title='回调深度分布',
        template='plotly_white', height=400
    )
    return fig


# ============================================================
# 5. Report Generation
# ============================================================

def generate_html_report(df, events, stats, drawdown_series):
    """Generate a complete HTML report."""

    # Create all charts
    fig_price = create_price_chart_with_drawdowns(df, events)
    fig_dd_series = create_drawdown_series_chart(df, drawdown_series)
    fig_magnitude = create_magnitude_bar_chart(events)
    fig_box = create_duration_box_chart(events)
    fig_timeline = create_frequency_timeline(events)
    fig_recovery = create_recovery_time_chart(events)
    fig_pie = create_depth_distribution_pie(stats)

    # Build event table
    event_rows = ""
    for i, e in enumerate(events, 1):
        rec_date = e['recovery_date'].strftime('%Y-%m-%d') if e['recovery_date'] else '未恢复'
        rec_days = f"{e['recovery_from_trough']}天" if e['recovery_from_trough'] else 'N/A'
        total_days = f"{e['duration_to_recovery']}天" if e['duration_to_recovery'] else 'N/A'
        event_rows += f"""
        <tr>
            <td>{i}</td>
            <td>{e['label']}</td>
            <td>{e['peak_date'].strftime('%Y-%m-%d')}</td>
            <td>{e['trough_date'].strftime('%Y-%m-%d')}</td>
            <td>{rec_date}</td>
            <td style="color:red;font-weight:bold">{e['drawdown_pct']:.1f}%</td>
            <td>{e['peak_price']:.2f}</td>
            <td>{e['trough_price']:.2f}</td>
            <td>{e['duration_to_trough']}天</td>
            <td>{rec_days}</td>
            <td>{total_days}</td>
        </tr>
        """

    avg_recovery = f"{stats['avg_recovery_time']:.0f}" if stats['avg_recovery_time'] else 'N/A'
    median_recovery = f"{stats['median_recovery_time']:.0f}" if stats['median_recovery_time'] else 'N/A'
    freq_years = f"{stats['frequency_years']:.1f}" if stats['frequency_years'] else 'N/A'

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>标普500历史回调分析报告 (1995-2025)</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f7fa; color: #333; line-height: 1.6; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); color: white; padding: 40px; border-radius: 12px; margin-bottom: 30px; }}
        .header h1 {{ font-size: 2em; margin-bottom: 10px; }}
        .header p {{ font-size: 1.1em; opacity: 0.9; }}
        .summary-box {{ background: white; border-radius: 12px; padding: 30px; margin-bottom: 30px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }}
        .summary-box h2 {{ color: #1a1a2e; border-bottom: 3px solid #0f3460; padding-bottom: 10px; margin-bottom: 20px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .stat-card {{ background: linear-gradient(135deg, #f8f9fa, #e9ecef); padding: 20px; border-radius: 10px; text-align: center; border-left: 4px solid #0f3460; }}
        .stat-card .number {{ font-size: 2em; font-weight: bold; color: #0f3460; }}
        .stat-card .label {{ font-size: 0.9em; color: #666; margin-top: 5px; }}
        .chart-container {{ background: white; border-radius: 12px; padding: 20px; margin-bottom: 30px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 0.9em; }}
        th {{ background: #1a1a2e; color: white; padding: 12px 8px; text-align: center; }}
        td {{ padding: 10px 8px; text-align: center; border-bottom: 1px solid #eee; }}
        tr:hover {{ background: #f8f9fa; }}
        .conclusion {{ background: #e8f5e9; border-left: 5px solid #4caf50; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .conclusion h3 {{ color: #2e7d32; margin-bottom: 10px; }}
        .warning {{ background: #fff3e0; border-left: 5px solid #ff9800; padding: 15px; border-radius: 8px; margin: 15px 0; }}
        .key-finding {{ background: #e3f2fd; border-left: 5px solid #2196f3; padding: 15px; border-radius: 8px; margin: 15px 0; }}
        ul {{ padding-left: 20px; }}
        li {{ margin: 8px 0; }}
    </style>
</head>
<body>
<div class="container">

    <div class="header">
        <h1>标普500历史回调分析报告</h1>
        <p>研究区间：1995年 - 2025年 | 回调阈值：>10% | 数据来源：S&P 500历史数据</p>
        <p>报告生成日期：{datetime.now().strftime('%Y-%m-%d')}</p>
    </div>

    <!-- ===== 研究结论 ===== -->
    <div class="summary-box">
        <h2>研究结论</h2>

        <div class="key-finding">
            <h3>核心发现</h3>
            <p>过去30年（1995-2025年），标普500共发生了 <strong>{stats['total_events']}次</strong> 超过10%的市场回调。
            平均约每 <strong>{freq_years}年</strong> 发生一次10%+的显著回调。</p>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="number">{stats['total_events']}</div>
                <div class="label">10%+回调事件总数</div>
            </div>
            <div class="stat-card">
                <div class="number">{stats['avg_magnitude']:.1f}%</div>
                <div class="label">平均回调幅度</div>
            </div>
            <div class="stat-card">
                <div class="number">{stats['avg_duration_to_trough']:.0f}天</div>
                <div class="label">平均下跌持续时间</div>
            </div>
            <div class="stat-card">
                <div class="number">{avg_recovery}天</div>
                <div class="label">平均恢复时间</div>
            </div>
            <div class="stat-card">
                <div class="number">{stats['max_magnitude']:.1f}%</div>
                <div class="label">最大回调幅度</div>
            </div>
            <div class="stat-card">
                <div class="number">{freq_years}年</div>
                <div class="label">平均发生频率</div>
            </div>
        </div>

        <div class="conclusion">
            <h3>回调典型特征</h3>
            <ul>
                <li><strong>频率</strong>：平均每{freq_years}年出现一次超过10%的回调，说明10%+回调是市场常态而非异常。</li>
                <li><strong>幅度</strong>：平均回调幅度为{stats['avg_magnitude']:.1f}%，中位数为{stats['median_magnitude']:.1f}%。其中{stats['depth_10_20']}次为中等回调(10-20%)，{stats['depth_20_30']}次为大幅回调(20-30%)，{stats['depth_30_plus']}次为股灾级别(30%+)。</li>
                <li><strong>持续时间</strong>：从峰值到谷底平均需要{stats['avg_duration_to_trough']:.0f}天（中位数{stats['median_duration_to_trough']:.0f}天）。</li>
                <li><strong>恢复能力</strong>：从谷底恢复到前高平均需要{avg_recovery}天（中位数{median_recovery}天）。市场展现了极强的恢复韧性——所有已完成的回调最终都实现了完全恢复。</li>
            </ul>
        </div>

        <div class="conclusion">
            <h3>市场恢复能力分析</h3>
            <ul>
                <li>标普500在过去30年中经历了互联网泡沫（-49.1%）、全球金融危机（-56.8%）、COVID-19（-33.9%）等重大股灾，但每次都成功恢复并创出新高。</li>
                <li>小幅回调（10-20%）通常在数月内恢复；重大股灾可能需要数年。</li>
                <li>回调幅度越深，恢复时间通常越长，但长期来看市场始终向上。</li>
            </ul>
        </div>

        <div class="warning">
            <h3>对投资者的启示</h3>
            <ul>
                <li><strong>保持耐心</strong>：10%+回调平均每{freq_years}年发生一次，是正常的市场行为，不必恐慌。</li>
                <li><strong>逢低布局</strong>：历史数据显示每次重大回调后都是绝佳的买入机会。</li>
                <li><strong>风险管理</strong>：尽管市场终会恢复，但30%+的深度回调可能持续数年，需做好资金管理。</li>
                <li><strong>长期投资</strong>：买入持有策略在过去30年表现优异，频繁择时反而可能错失良机。</li>
            </ul>
        </div>
    </div>

    <!-- ===== 详细数据表 ===== -->
    <div class="summary-box">
        <h2>回调事件详细数据</h2>
        <div style="overflow-x: auto;">
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>事件名称</th>
                    <th>峰值日期</th>
                    <th>谷底日期</th>
                    <th>恢复日期</th>
                    <th>回调幅度</th>
                    <th>峰值价格</th>
                    <th>谷底价格</th>
                    <th>下跌天数</th>
                    <th>恢复天数</th>
                    <th>总周期</th>
                </tr>
            </thead>
            <tbody>
                {event_rows}
            </tbody>
        </table>
        </div>
    </div>

    <!-- ===== Charts ===== -->
    <div class="chart-container">
        <div id="price_chart"></div>
    </div>
    <div class="chart-container">
        <div id="dd_series_chart"></div>
    </div>
    <div class="chart-container">
        <div id="magnitude_chart"></div>
    </div>
    <div class="chart-container">
        <div id="pie_chart"></div>
    </div>
    <div class="chart-container">
        <div id="box_chart"></div>
    </div>
    <div class="chart-container">
        <div id="timeline_chart"></div>
    </div>
    <div class="chart-container">
        <div id="recovery_chart"></div>
    </div>

    <div class="summary-box" style="text-align:center; color:#999; font-size:0.9em;">
        <p>本报告仅供学术研究和教育目的使用，不构成投资建议。</p>
        <p>数据来源：S&P 500 历史数据 | 分析区间：1995-2025</p>
    </div>

</div>

<script>
    {fig_price.to_json()}
    Plotly.newPlot('price_chart', {fig_price.to_json()}.data, {fig_price.to_json()}.layout);

    Plotly.newPlot('dd_series_chart', {fig_dd_series.to_json()}.data, {fig_dd_series.to_json()}.layout);

    Plotly.newPlot('magnitude_chart', {fig_magnitude.to_json()}.data, {fig_magnitude.to_json()}.layout);

    Plotly.newPlot('pie_chart', {fig_pie.to_json()}.data, {fig_pie.to_json()}.layout);

    Plotly.newPlot('box_chart', {fig_box.to_json()}.data, {fig_box.to_json()}.layout);

    Plotly.newPlot('timeline_chart', {fig_timeline.to_json()}.data, {fig_timeline.to_json()}.layout);

    Plotly.newPlot('recovery_chart', {fig_recovery.to_json()}.data, {fig_recovery.to_json()}.layout);
</script>

</body>
</html>
"""
    return html


def generate_html_report_v2(df, events, stats, drawdown_series):
    """Generate HTML report using Plotly's to_html for chart embedding (more reliable)."""
    fig_price = create_price_chart_with_drawdowns(df, events)
    fig_dd_series = create_drawdown_series_chart(df, drawdown_series)
    fig_magnitude = create_magnitude_bar_chart(events)
    fig_box = create_duration_box_chart(events)
    fig_timeline = create_frequency_timeline(events)
    fig_recovery = create_recovery_time_chart(events)
    fig_pie = create_depth_distribution_pie(stats)

    charts_html = {
        'price': fig_price.to_html(full_html=False, include_plotlyjs=False),
        'dd_series': fig_dd_series.to_html(full_html=False, include_plotlyjs=False),
        'magnitude': fig_magnitude.to_html(full_html=False, include_plotlyjs=False),
        'box': fig_box.to_html(full_html=False, include_plotlyjs=False),
        'timeline': fig_timeline.to_html(full_html=False, include_plotlyjs=False),
        'recovery': fig_recovery.to_html(full_html=False, include_plotlyjs=False),
        'pie': fig_pie.to_html(full_html=False, include_plotlyjs=False),
    }

    event_rows = ""
    for i, e in enumerate(events, 1):
        rec_date = e['recovery_date'].strftime('%Y-%m-%d') if e['recovery_date'] else '未恢复'
        rec_days = f"{e['recovery_from_trough']}天" if e['recovery_from_trough'] else 'N/A'
        total_days = f"{e['duration_to_recovery']}天" if e['duration_to_recovery'] else 'N/A'
        event_rows += f"""
        <tr>
            <td>{i}</td>
            <td>{e['label']}</td>
            <td>{e['peak_date'].strftime('%Y-%m-%d')}</td>
            <td>{e['trough_date'].strftime('%Y-%m-%d')}</td>
            <td>{rec_date}</td>
            <td style="color:red;font-weight:bold">{e['drawdown_pct']:.1f}%</td>
            <td>{e['peak_price']:.2f}</td>
            <td>{e['trough_price']:.2f}</td>
            <td>{e['duration_to_trough']}天</td>
            <td>{rec_days}</td>
            <td>{total_days}</td>
        </tr>"""

    avg_recovery = f"{stats['avg_recovery_time']:.0f}" if stats['avg_recovery_time'] else 'N/A'
    median_recovery = f"{stats['median_recovery_time']:.0f}" if stats['median_recovery_time'] else 'N/A'
    freq_years = f"{stats['frequency_years']:.1f}" if stats['frequency_years'] else 'N/A'

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>标普500历史回调分析报告 (1995-2025)</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f7fa; color: #333; line-height: 1.6; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); color: white; padding: 40px; border-radius: 12px; margin-bottom: 30px; }}
        .header h1 {{ font-size: 2em; margin-bottom: 10px; }}
        .header p {{ font-size: 1.1em; opacity: 0.9; }}
        .summary-box {{ background: white; border-radius: 12px; padding: 30px; margin-bottom: 30px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }}
        .summary-box h2 {{ color: #1a1a2e; border-bottom: 3px solid #0f3460; padding-bottom: 10px; margin-bottom: 20px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .stat-card {{ background: linear-gradient(135deg, #f8f9fa, #e9ecef); padding: 20px; border-radius: 10px; text-align: center; border-left: 4px solid #0f3460; }}
        .stat-card .number {{ font-size: 2em; font-weight: bold; color: #0f3460; }}
        .stat-card .label {{ font-size: 0.9em; color: #666; margin-top: 5px; }}
        .chart-container {{ background: white; border-radius: 12px; padding: 20px; margin-bottom: 30px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 0.9em; }}
        th {{ background: #1a1a2e; color: white; padding: 12px 8px; text-align: center; }}
        td {{ padding: 10px 8px; text-align: center; border-bottom: 1px solid #eee; }}
        tr:hover {{ background: #f8f9fa; }}
        .conclusion {{ background: #e8f5e9; border-left: 5px solid #4caf50; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .conclusion h3 {{ color: #2e7d32; margin-bottom: 10px; }}
        .warning {{ background: #fff3e0; border-left: 5px solid #ff9800; padding: 15px; border-radius: 8px; margin: 15px 0; }}
        .key-finding {{ background: #e3f2fd; border-left: 5px solid #2196f3; padding: 15px; border-radius: 8px; margin: 15px 0; }}
        ul {{ padding-left: 20px; }}
        li {{ margin: 8px 0; }}
    </style>
</head>
<body>
<div class="container">

    <div class="header">
        <h1>标普500历史回调分析报告</h1>
        <p>研究区间：1995年 - 2025年 | 回调阈值：>10% | 数据来源：S&P 500历史数据</p>
        <p>报告生成日期：{datetime.now().strftime('%Y-%m-%d')}</p>
    </div>

    <!-- ===== 研究结论 ===== -->
    <div class="summary-box">
        <h2>研究结论</h2>

        <div class="key-finding">
            <h3>核心发现</h3>
            <p>过去30年（1995-2025年），标普500共发生了 <strong>{stats['total_events']}次</strong> 超过10%的市场回调。
            平均约每 <strong>{freq_years}年</strong> 发生一次10%+的显著回调。</p>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="number">{stats['total_events']}</div>
                <div class="label">10%+回调事件总数</div>
            </div>
            <div class="stat-card">
                <div class="number">{stats['avg_magnitude']:.1f}%</div>
                <div class="label">平均回调幅度</div>
            </div>
            <div class="stat-card">
                <div class="number">{stats['avg_duration_to_trough']:.0f}天</div>
                <div class="label">平均下跌持续时间</div>
            </div>
            <div class="stat-card">
                <div class="number">{avg_recovery}天</div>
                <div class="label">平均恢复时间</div>
            </div>
            <div class="stat-card">
                <div class="number">{stats['max_magnitude']:.1f}%</div>
                <div class="label">最大回调幅度</div>
            </div>
            <div class="stat-card">
                <div class="number">{freq_years}年</div>
                <div class="label">平均发生频率</div>
            </div>
        </div>

        <div class="conclusion">
            <h3>回调典型特征</h3>
            <ul>
                <li><strong>频率</strong>：平均每{freq_years}年出现一次超过10%的回调，说明10%+回调是市场常态而非异常。</li>
                <li><strong>幅度</strong>：平均回调幅度为{stats['avg_magnitude']:.1f}%，中位数为{stats['median_magnitude']:.1f}%。其中{stats['depth_10_20']}次为中等回调(10-20%)，{stats['depth_20_30']}次为大幅回调(20-30%)，{stats['depth_30_plus']}次为股灾级别(30%+)。</li>
                <li><strong>持续时间</strong>：从峰值到谷底平均需要{stats['avg_duration_to_trough']:.0f}天（中位数{stats['median_duration_to_trough']:.0f}天）。</li>
                <li><strong>恢复能力</strong>：从谷底恢复到前高平均需要{avg_recovery}天（中位数{median_recovery}天）。市场展现了极强的恢复韧性——所有已完成的回调最终都实现了完全恢复。</li>
            </ul>
        </div>

        <div class="conclusion">
            <h3>市场恢复能力分析</h3>
            <ul>
                <li>标普500在过去30年中经历了互联网泡沫、全球金融危机、COVID-19等重大股灾，但每次都成功恢复并创出新高。</li>
                <li>小幅回调（10-20%）通常在数月内恢复；重大股灾可能需要数年。</li>
                <li>回调幅度越深，恢复时间通常越长，但长期来看市场始终向上。</li>
            </ul>
        </div>

        <div class="warning">
            <h3>对投资者的启示</h3>
            <ul>
                <li><strong>保持耐心</strong>：10%+回调平均每{freq_years}年发生一次，是正常的市场行为，不必恐慌。</li>
                <li><strong>逢低布局</strong>：历史数据显示每次重大回调后都是绝佳的买入机会。</li>
                <li><strong>风险管理</strong>：尽管市场终会恢复，但30%+的深度回调可能持续数年，需做好资金管理。</li>
                <li><strong>长期投资</strong>：买入持有策略在过去30年表现优异，频繁择时反而可能错失良机。</li>
            </ul>
        </div>
    </div>

    <!-- ===== 详细数据表 ===== -->
    <div class="summary-box">
        <h2>回调事件详细数据</h2>
        <div style="overflow-x: auto;">
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>事件名称</th>
                    <th>峰值日期</th>
                    <th>谷底日期</th>
                    <th>恢复日期</th>
                    <th>回调幅度</th>
                    <th>峰值价格</th>
                    <th>谷底价格</th>
                    <th>下跌天数</th>
                    <th>恢复天数</th>
                    <th>总周期</th>
                </tr>
            </thead>
            <tbody>
                {event_rows}
            </tbody>
        </table>
        </div>
    </div>

    <!-- ===== Charts ===== -->
    <div class="chart-container">{charts_html['price']}</div>
    <div class="chart-container">{charts_html['dd_series']}</div>
    <div class="chart-container">{charts_html['magnitude']}</div>
    <div class="chart-container">{charts_html['pie']}</div>
    <div class="chart-container">{charts_html['box']}</div>
    <div class="chart-container">{charts_html['timeline']}</div>
    <div class="chart-container">{charts_html['recovery']}</div>

    <div class="summary-box" style="text-align:center; color:#999; font-size:0.9em;">
        <p>本报告仅供学术研究和教育目的使用，不构成投资建议。</p>
        <p>数据来源：S&P 500 历史数据 | 分析区间：1995-2025</p>
    </div>

</div>
</body>
</html>"""
    return html


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Task 1: S&P 500 Historical Drawdown Analysis (1995-2025)")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading data...")
    df = load_data('SP500.csv')
    print(f"  Data loaded: {len(df)} trading days from {df['Date'].iloc[0].strftime('%Y-%m-%d')} to {df['Date'].iloc[-1].strftime('%Y-%m-%d')}")

    # Identify drawdowns
    print("\n[2/5] Identifying drawdowns (threshold: -10%)...")
    events, drawdown_series = identify_drawdowns(df, threshold=-0.10)
    events = label_events(events)
    print(f"  Found {len(events)} drawdown events exceeding 10%")

    for i, e in enumerate(events, 1):
        print(f"    {i}. {e['label']}: {e['drawdown_pct']:.1f}% | Peak: {e['peak_date'].strftime('%Y-%m-%d')} | Trough: {e['trough_date'].strftime('%Y-%m-%d')}")

    # Compute statistics
    print("\n[3/5] Computing statistics...")
    stats = compute_statistics(events)
    print(f"  Average drawdown: {stats['avg_magnitude']:.1f}%")
    print(f"  Average duration to trough: {stats['avg_duration_to_trough']:.0f} days")
    print(f"  Depth distribution: 10-20%: {stats['depth_10_20']}, 20-30%: {stats['depth_20_30']}, 30%+: {stats['depth_30_plus']}")

    # Save drawdowns CSV
    print("\n[4/5] Saving drawdown events to CSV...")
    os.makedirs('outputs', exist_ok=True)
    events_df = pd.DataFrame(events)
    events_df.to_csv('outputs/task1_drawdowns.csv', index=False, encoding='utf-8-sig')
    print(f"  Saved to outputs/task1_drawdowns.csv")

    # Generate HTML report
    print("\n[5/5] Generating HTML report...")
    html = generate_html_report_v2(df, events, stats, drawdown_series)
    with open('outputs/task1_analysis.html', 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  Saved to outputs/task1_analysis.html")

    print("\n" + "=" * 60)
    print("Task 1 Complete!")
    print("=" * 60)

    return df, events, stats


if __name__ == '__main__':
    main()
