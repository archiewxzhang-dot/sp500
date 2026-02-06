#!/usr/bin/env python3
"""
Task 2: Dynamic Position Management Strategy Backtesting
5 Strategies based on 10% drawdown signal, no transaction costs.

Strategies:
1. Buy & Hold - Always 100%
2. Fixed 50% - Reduce to 50% at 10% drawdown, return to 100% at new high
3. Fixed 30% - Reduce to 30% at 10% drawdown, return to 100% at new high
4. Fixed 50% v2 - Reduce to 50%, add 10% for every 20% recovery from trough
5. Fixed 30% v2 - Reduce to 30%, gradually add back on recovery
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
    df = df[['Date', 'Close']].dropna()
    return df


def get_backtest_period(df):
    """Use period from 2009-03-10 (post GFC trough) to present."""
    start_date = '2009-03-10'
    return df[df['Date'] >= start_date].copy().reset_index(drop=True)


# ============================================================
# 2. Performance Metrics
# ============================================================

def compute_performance_metrics(portfolio_values, dates, risk_free_rate=0.02):
    """Compute comprehensive performance metrics."""
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
    n_years = (dates[-1] - dates[0]).days / 365.25
    annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    # Volatility
    annual_vol = np.std(daily_returns) * np.sqrt(252)

    # Sharpe Ratio
    daily_rf = risk_free_rate / 252
    excess_returns = daily_returns - daily_rf
    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0

    # Maximum Drawdown
    running_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - running_max) / running_max
    max_dd = np.min(drawdowns)

    # Calmar Ratio
    calmar = annualized_return / abs(max_dd) if max_dd != 0 else 0

    # Sortino Ratio
    downside_returns = daily_returns[daily_returns < daily_rf]
    downside_vol = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 1e-10
    sortino = (annualized_return - risk_free_rate) / downside_vol

    # Win Rate
    win_rate = np.sum(daily_returns > 0) / len(daily_returns) if len(daily_returns) > 0 else 0

    return {
        'total_return': total_return * 100,
        'annualized_return': annualized_return * 100,
        'annual_volatility': annual_vol * 100,
        'max_drawdown': max_dd * 100,
        'sharpe_ratio': sharpe,
        'calmar_ratio': calmar,
        'sortino_ratio': sortino,
        'win_rate': win_rate * 100,
        'final_value': portfolio_values[-1],
    }


# ============================================================
# 3. Strategy Implementations
# ============================================================

def strategy_buy_and_hold(prices, dates, initial_capital=100000):
    """
    Strategy 1: Buy and Hold
    Always 100% invested, no changes.
    """
    n = len(prices)
    shares = initial_capital / prices[0]
    portfolio = shares * prices
    positions = np.ones(n)
    trades = []

    return portfolio, trades, positions


def strategy_fixed_50(prices, dates, initial_capital=100000):
    """
    Strategy 2: Fixed 50%
    - When index drops 10% from peak: reduce to 50% position
    - When index recovers to previous peak: return to 100%
    """
    n = len(prices)
    portfolio = np.zeros(n)
    positions = np.zeros(n)
    portfolio[0] = initial_capital
    positions[0] = 1.0

    trades = []
    peak = prices[0]
    shares = initial_capital / prices[0]
    cash = 0.0
    current_pos = 1.0
    in_drawdown_mode = False

    for i in range(1, n):
        # Update peak if we're at new high
        if prices[i] >= peak:
            peak = prices[i]
            # If we were in drawdown mode, return to 100%
            if in_drawdown_mode and current_pos < 1.0:
                total_value = shares * prices[i] + cash
                shares = total_value / prices[i]
                cash = 0.0
                trades.append({
                    'date': dates[i], 'action': 'INCREASE',
                    'old_position': current_pos, 'new_position': 1.0,
                    'price': prices[i], 'reason': '恢复至前高',
                    'portfolio_value': total_value, 'strategy': 'Fixed 50%'
                })
                current_pos = 1.0
                in_drawdown_mode = False

        # Check for 10% drawdown signal
        dd = (prices[i] - peak) / peak
        if dd <= -0.10 and not in_drawdown_mode:
            # Trigger: reduce to 50%
            in_drawdown_mode = True
            total_value = shares * prices[i] + cash
            shares = 0.50 * total_value / prices[i]
            cash = 0.50 * total_value
            trades.append({
                'date': dates[i], 'action': 'REDUCE',
                'old_position': current_pos, 'new_position': 0.50,
                'price': prices[i], 'reason': f'触发10%回调信号 (DD={dd*100:.1f}%)',
                'portfolio_value': total_value, 'strategy': 'Fixed 50%'
            })
            current_pos = 0.50

        portfolio[i] = shares * prices[i] + cash
        positions[i] = current_pos

    return portfolio, trades, positions


def strategy_fixed_30(prices, dates, initial_capital=100000):
    """
    Strategy 3: Fixed 30%
    - When index drops 10% from peak: reduce to 30% position
    - When index recovers to previous peak: return to 100%
    """
    n = len(prices)
    portfolio = np.zeros(n)
    positions = np.zeros(n)
    portfolio[0] = initial_capital
    positions[0] = 1.0

    trades = []
    peak = prices[0]
    shares = initial_capital / prices[0]
    cash = 0.0
    current_pos = 1.0
    in_drawdown_mode = False

    for i in range(1, n):
        # Update peak if we're at new high
        if prices[i] >= peak:
            peak = prices[i]
            if in_drawdown_mode and current_pos < 1.0:
                total_value = shares * prices[i] + cash
                shares = total_value / prices[i]
                cash = 0.0
                trades.append({
                    'date': dates[i], 'action': 'INCREASE',
                    'old_position': current_pos, 'new_position': 1.0,
                    'price': prices[i], 'reason': '恢复至前高',
                    'portfolio_value': total_value, 'strategy': 'Fixed 30%'
                })
                current_pos = 1.0
                in_drawdown_mode = False

        # Check for 10% drawdown signal
        dd = (prices[i] - peak) / peak
        if dd <= -0.10 and not in_drawdown_mode:
            in_drawdown_mode = True
            total_value = shares * prices[i] + cash
            shares = 0.30 * total_value / prices[i]
            cash = 0.70 * total_value
            trades.append({
                'date': dates[i], 'action': 'REDUCE',
                'old_position': current_pos, 'new_position': 0.30,
                'price': prices[i], 'reason': f'触发10%回调信号 (DD={dd*100:.1f}%)',
                'portfolio_value': total_value, 'strategy': 'Fixed 30%'
            })
            current_pos = 0.30

        portfolio[i] = shares * prices[i] + cash
        positions[i] = current_pos

    return portfolio, trades, positions


def strategy_fixed_50_v2(prices, dates, initial_capital=100000):
    """
    Strategy 4: Fixed 50% v2
    - When index drops 10% from peak: reduce to 50% position
    - Add 10% position for every 20% recovery from trough (until 100%)
    """
    n = len(prices)
    portfolio = np.zeros(n)
    positions = np.zeros(n)
    portfolio[0] = initial_capital
    positions[0] = 1.0

    trades = []
    peak = prices[0]
    trough = prices[0]
    shares = initial_capital / prices[0]
    cash = 0.0
    current_pos = 1.0
    in_drawdown_mode = False
    last_add_level = 0  # Track which recovery level we last added at

    for i in range(1, n):
        # Update peak if at new high
        if prices[i] >= peak:
            peak = prices[i]
            trough = prices[i]
            in_drawdown_mode = False
            last_add_level = 0
            if current_pos < 1.0:
                total_value = shares * prices[i] + cash
                shares = total_value / prices[i]
                cash = 0.0
                trades.append({
                    'date': dates[i], 'action': 'INCREASE',
                    'old_position': current_pos, 'new_position': 1.0,
                    'price': prices[i], 'reason': '恢复至前高',
                    'portfolio_value': total_value, 'strategy': 'Fixed 50% v2'
                })
                current_pos = 1.0

        # Update trough
        if prices[i] < trough:
            trough = prices[i]
            last_add_level = 0  # Reset add level when new trough

        dd = (prices[i] - peak) / peak

        # Check for 10% drawdown signal
        if dd <= -0.10 and not in_drawdown_mode:
            in_drawdown_mode = True
            total_value = shares * prices[i] + cash
            shares = 0.50 * total_value / prices[i]
            cash = 0.50 * total_value
            trades.append({
                'date': dates[i], 'action': 'REDUCE',
                'old_position': current_pos, 'new_position': 0.50,
                'price': prices[i], 'reason': f'触发10%回调信号 (DD={dd*100:.1f}%)',
                'portfolio_value': total_value, 'strategy': 'Fixed 50% v2'
            })
            current_pos = 0.50
            trough = prices[i]

        # Check for recovery-based adding (only in drawdown mode)
        if in_drawdown_mode and current_pos < 1.0 and prices[i] > trough:
            recovery_pct = (prices[i] - trough) / trough
            # Every 20% recovery, add 10%
            add_levels = int(recovery_pct / 0.20)
            if add_levels > last_add_level:
                add_amount = (add_levels - last_add_level) * 0.10
                new_pos = min(1.0, current_pos + add_amount)
                if new_pos > current_pos:
                    total_value = shares * prices[i] + cash
                    shares = new_pos * total_value / prices[i]
                    cash = (1 - new_pos) * total_value
                    trades.append({
                        'date': dates[i], 'action': 'INCREASE',
                        'old_position': current_pos, 'new_position': new_pos,
                        'price': prices[i], 'reason': f'从底部反弹{recovery_pct*100:.1f}%',
                        'portfolio_value': total_value, 'strategy': 'Fixed 50% v2'
                    })
                    current_pos = new_pos
                    last_add_level = add_levels

        portfolio[i] = shares * prices[i] + cash
        positions[i] = current_pos

    return portfolio, trades, positions


def strategy_fixed_30_v2(prices, dates, initial_capital=100000):
    """
    Strategy 5: Fixed 30% v2
    - When index drops 10% from peak: reduce to 30% position
    - Add 14% position for every 20% recovery from trough (until 100%)
    (70% to add back, over ~5 levels of 20% recovery each = 14% per level)
    """
    n = len(prices)
    portfolio = np.zeros(n)
    positions = np.zeros(n)
    portfolio[0] = initial_capital
    positions[0] = 1.0

    trades = []
    peak = prices[0]
    trough = prices[0]
    shares = initial_capital / prices[0]
    cash = 0.0
    current_pos = 1.0
    in_drawdown_mode = False
    last_add_level = 0

    for i in range(1, n):
        # Update peak
        if prices[i] >= peak:
            peak = prices[i]
            trough = prices[i]
            in_drawdown_mode = False
            last_add_level = 0
            if current_pos < 1.0:
                total_value = shares * prices[i] + cash
                shares = total_value / prices[i]
                cash = 0.0
                trades.append({
                    'date': dates[i], 'action': 'INCREASE',
                    'old_position': current_pos, 'new_position': 1.0,
                    'price': prices[i], 'reason': '恢复至前高',
                    'portfolio_value': total_value, 'strategy': 'Fixed 30% v2'
                })
                current_pos = 1.0

        # Update trough
        if prices[i] < trough:
            trough = prices[i]
            last_add_level = 0

        dd = (prices[i] - peak) / peak

        # Check for 10% drawdown signal
        if dd <= -0.10 and not in_drawdown_mode:
            in_drawdown_mode = True
            total_value = shares * prices[i] + cash
            shares = 0.30 * total_value / prices[i]
            cash = 0.70 * total_value
            trades.append({
                'date': dates[i], 'action': 'REDUCE',
                'old_position': current_pos, 'new_position': 0.30,
                'price': prices[i], 'reason': f'触发10%回调信号 (DD={dd*100:.1f}%)',
                'portfolio_value': total_value, 'strategy': 'Fixed 30% v2'
            })
            current_pos = 0.30
            trough = prices[i]

        # Check for recovery-based adding
        if in_drawdown_mode and current_pos < 1.0 and prices[i] > trough:
            recovery_pct = (prices[i] - trough) / trough
            # Every 20% recovery, add 14% (to get from 30% to 100% over ~5 levels)
            add_levels = int(recovery_pct / 0.20)
            if add_levels > last_add_level:
                add_amount = (add_levels - last_add_level) * 0.14
                new_pos = min(1.0, current_pos + add_amount)
                if new_pos > current_pos:
                    total_value = shares * prices[i] + cash
                    shares = new_pos * total_value / prices[i]
                    cash = (1 - new_pos) * total_value
                    trades.append({
                        'date': dates[i], 'action': 'INCREASE',
                        'old_position': current_pos, 'new_position': new_pos,
                        'price': prices[i], 'reason': f'从底部反弹{recovery_pct*100:.1f}%',
                        'portfolio_value': total_value, 'strategy': 'Fixed 30% v2'
                    })
                    current_pos = new_pos
                    last_add_level = add_levels

        portfolio[i] = shares * prices[i] + cash
        positions[i] = current_pos

    return portfolio, trades, positions


# ============================================================
# 4. Visualization
# ============================================================

COLORS = {
    'Buy & Hold': '#2c3e50',
    'Fixed 50%': '#3498db',
    'Fixed 30%': '#e74c3c',
    'Fixed 50% v2': '#27ae60',
    'Fixed 30% v2': '#9b59b6'
}


def create_cumulative_returns_chart(dates, results):
    """Cumulative returns comparison."""
    fig = go.Figure()

    for name, data in results.items():
        normalized = data['portfolio'] / data['portfolio'][0] * 100
        fig.add_trace(go.Scatter(
            x=dates, y=normalized,
            mode='lines', name=name,
            line=dict(color=COLORS.get(name, '#333'), width=2)
        ))

    fig.update_layout(
        title='累计收益曲线对比（初始$100,000 = 100）',
        xaxis_title='日期', yaxis_title='投资组合价值（基准=100）',
        template='plotly_white', height=500,
        hovermode='x unified',
        legend=dict(x=0.01, y=0.99)
    )
    return fig


def create_drawdown_comparison_chart(dates, results):
    """Drawdown comparison."""
    fig = go.Figure()

    for name, data in results.items():
        running_max = np.maximum.accumulate(data['portfolio'])
        dd = (data['portfolio'] - running_max) / running_max * 100
        fig.add_trace(go.Scatter(
            x=dates, y=dd,
            mode='lines', name=name,
            line=dict(color=COLORS.get(name, '#333'), width=1.5),
            fill='tozeroy' if name == 'Buy & Hold' else None,
            fillcolor='rgba(44,62,80,0.1)' if name == 'Buy & Hold' else None
        ))

    fig.update_layout(
        title='回撤对比 - 各策略的最大亏损暴露',
        xaxis_title='日期', yaxis_title='回撤 (%)',
        template='plotly_white', height=450,
        hovermode='x unified'
    )
    return fig


def create_position_chart(dates, results):
    """Position allocation over time."""
    fig = go.Figure()

    for name, data in results.items():
        if name == 'Buy & Hold':
            continue
        fig.add_trace(go.Scatter(
            x=dates, y=data['positions'] * 100,
            mode='lines', name=name,
            line=dict(color=COLORS.get(name, '#333'), width=1.5)
        ))

    fig.update_layout(
        title='仓位变化时间序列 - 各策略的仓位调整',
        xaxis_title='日期', yaxis_title='仓位 (%)',
        template='plotly_white', height=400,
        hovermode='x unified',
        yaxis=dict(range=[0, 105])
    )
    return fig


def create_annual_returns_chart(dates, results):
    """Annual returns comparison."""
    fig = go.Figure()

    years = sorted(set(d.year for d in dates))

    for name, data in results.items():
        portfolio = data['portfolio']
        annual_returns = []
        year_labels = []

        for year in years:
            year_mask = [d.year == year for d in dates]
            year_indices = [i for i, m in enumerate(year_mask) if m]
            if len(year_indices) >= 2:
                ret = (portfolio[year_indices[-1]] / portfolio[year_indices[0]] - 1) * 100
                annual_returns.append(ret)
                year_labels.append(str(year))

        fig.add_trace(go.Bar(
            x=year_labels, y=annual_returns,
            name=name, marker_color=COLORS.get(name, '#333')
        ))

    fig.update_layout(
        title='年度收益对比',
        xaxis_title='年份', yaxis_title='收益率 (%)',
        barmode='group',
        template='plotly_white', height=500
    )
    return fig


def create_risk_return_scatter(results):
    """Risk-return scatter plot."""
    fig = go.Figure()

    for name, data in results.items():
        m = data['metrics']
        fig.add_trace(go.Scatter(
            x=[m['annual_volatility']], y=[m['annualized_return']],
            mode='markers+text', name=name,
            marker=dict(size=20, color=COLORS.get(name, '#333')),
            text=[name], textposition='top center', textfont=dict(size=11)
        ))

    # Add diagonal lines for Sharpe ratios
    max_vol = max(d['metrics']['annual_volatility'] for d in results.values()) + 2
    for sr in [0.5, 0.75, 1.0]:
        fig.add_trace(go.Scatter(
            x=[0, max_vol], y=[2, 2 + sr * max_vol],
            mode='lines', line=dict(dash='dot', color='gray', width=1),
            showlegend=False, hoverinfo='skip'
        ))

    fig.update_layout(
        title='风险收益散点图（对角虚线为夏普比率等高线）',
        xaxis_title='年化波动率 (%)', yaxis_title='年化收益率 (%)',
        template='plotly_white', height=500
    )
    return fig


def create_metrics_radar_chart(results):
    """Radar chart comparing key metrics."""
    categories = ['年化收益', '夏普比率', '卡尔马比率', '索提诺比率', '回撤控制']

    fig = go.Figure()

    for name, data in results.items():
        m = data['metrics']
        # Normalize to 0-1 scale
        values = [
            min(m['annualized_return'] / 20, 1.0),
            min(max(m['sharpe_ratio'], 0) / 1.5, 1.0),
            min(max(m['calmar_ratio'], 0) / 1.0, 1.0),
            min(max(m['sortino_ratio'], 0) / 2.0, 1.0),
            1 + m['max_drawdown'] / 50,  # Convert (smaller DD = better)
        ]
        values.append(values[0])  # Close radar

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            name=name,
            line=dict(color=COLORS.get(name, '#333'), width=2),
            fill='toself', fillcolor=COLORS.get(name, '#333').replace(')', ',0.1)').replace('rgb', 'rgba'),
        ))

    fig.update_layout(
        title='策略综合评分雷达图',
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        template='plotly_white', height=500
    )
    return fig


def create_trade_timeline_chart(dates, prices, results):
    """Show trades on price chart for each strategy."""
    strategies = [k for k in results.keys() if k != 'Buy & Hold']

    fig = make_subplots(rows=len(strategies), cols=1, shared_xaxes=True,
                        subplot_titles=strategies, vertical_spacing=0.06)

    for idx, sname in enumerate(strategies, 1):
        data = results[sname]

        # Price line
        fig.add_trace(go.Scatter(
            x=dates, y=prices,
            mode='lines', name='S&P 500',
            line=dict(color='#bdc3c7', width=1),
            showlegend=(idx == 1)
        ), row=idx, col=1)

        # Trade markers
        for trade in data['trades']:
            color = '#27ae60' if trade['action'] == 'INCREASE' else '#e74c3c'
            symbol = 'triangle-up' if trade['action'] == 'INCREASE' else 'triangle-down'
            fig.add_trace(go.Scatter(
                x=[trade['date']], y=[trade['price']],
                mode='markers',
                marker=dict(color=color, size=10, symbol=symbol),
                showlegend=False,
                hovertext=f"{trade['action']}: {trade['old_position']:.0%} → {trade['new_position']:.0%}<br>{trade['reason']}"
            ), row=idx, col=1)

    fig.update_layout(
        height=250 * len(strategies),
        template='plotly_white',
        title='调仓时点标注（绿色=加仓，红色=减仓）',
        showlegend=False
    )
    return fig


# ============================================================
# 5. HTML Report
# ============================================================

def generate_html_report(dates, prices, results):
    """Generate comprehensive HTML report."""

    fig_cumulative = create_cumulative_returns_chart(dates, results)
    fig_drawdown = create_drawdown_comparison_chart(dates, results)
    fig_position = create_position_chart(dates, results)
    fig_annual = create_annual_returns_chart(dates, results)
    fig_scatter = create_risk_return_scatter(results)
    fig_radar = create_metrics_radar_chart(results)
    fig_trades = create_trade_timeline_chart(dates, prices, results)

    charts = {
        'cumulative': fig_cumulative.to_html(full_html=False, include_plotlyjs=False),
        'drawdown': fig_drawdown.to_html(full_html=False, include_plotlyjs=False),
        'position': fig_position.to_html(full_html=False, include_plotlyjs=False),
        'annual': fig_annual.to_html(full_html=False, include_plotlyjs=False),
        'scatter': fig_scatter.to_html(full_html=False, include_plotlyjs=False),
        'radar': fig_radar.to_html(full_html=False, include_plotlyjs=False),
        'trades': fig_trades.to_html(full_html=False, include_plotlyjs=False),
    }

    # Performance table
    perf_rows = ""
    for name, data in results.items():
        m = data['metrics']
        n_trades = len(data['trades'])
        perf_rows += f"""
        <tr>
            <td style="font-weight:bold;color:{COLORS.get(name, '#333')}">{name}</td>
            <td>{m['total_return']:.1f}%</td>
            <td>{m['annualized_return']:.1f}%</td>
            <td>{m['annual_volatility']:.1f}%</td>
            <td class="negative">{m['max_drawdown']:.1f}%</td>
            <td>{m['sharpe_ratio']:.2f}</td>
            <td>{m['calmar_ratio']:.2f}</td>
            <td>{m['sortino_ratio']:.2f}</td>
            <td>${m['final_value']:,.0f}</td>
            <td>{n_trades}</td>
        </tr>"""

    # Find best strategies
    best_return = max(results.items(), key=lambda x: x[1]['metrics']['annualized_return'])
    best_sharpe = max(results.items(), key=lambda x: x[1]['metrics']['sharpe_ratio'])
    best_dd = max(results.items(), key=lambda x: x[1]['metrics']['max_drawdown'])
    best_calmar = max(results.items(), key=lambda x: x[1]['metrics']['calmar_ratio'])

    start_date = dates[0].strftime('%Y-%m-%d')
    end_date = dates[-1].strftime('%Y-%m-%d')

    # Trade summary
    trade_summary = ""
    for name, data in results.items():
        if name == 'Buy & Hold':
            continue
        n_trades = len(data['trades'])
        reduces = sum(1 for t in data['trades'] if t['action'] == 'REDUCE')
        increases = sum(1 for t in data['trades'] if t['action'] == 'INCREASE')
        trade_summary += f"<li><strong>{name}</strong>: 共{n_trades}次调仓（减仓{reduces}次，加仓{increases}次）</li>"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>标普500仓位管理策略回测报告</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif; background: #f4f6f9; color: #2c3e50; line-height: 1.7; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%); color: white; padding: 40px; border-radius: 12px; margin-bottom: 30px; }}
        .header h1 {{ font-size: 2.2em; margin-bottom: 10px; }}
        .section {{ background: white; border-radius: 12px; padding: 30px; margin-bottom: 25px; box-shadow: 0 2px 15px rgba(0,0,0,0.06); }}
        .section h2 {{ color: #2c3e50; border-left: 4px solid #3498db; padding-left: 15px; margin-bottom: 20px; }}
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
        .strategy-box {{ background: #f8f9fa; border-radius: 8px; padding: 15px; margin: 10px 0; border-left: 4px solid; }}
        .strategy-box h4 {{ margin-bottom: 8px; }}
    </style>
</head>
<body>
<div class="container">

    <div class="header">
        <h1>标普500动态仓位管理策略回测报告</h1>
        <p>回测区间：{start_date} 至 {end_date} | 初始资金：$100,000 | 无交易费用</p>
        <p>回调信号定义：指数从高点下跌10% | 报告生成：{datetime.now().strftime('%Y-%m-%d')}</p>
    </div>

    <!-- ===== Executive Summary ===== -->
    <div class="section">
        <h2>核心结论 Executive Summary</h2>

        <div class="insight insight-blue">
            <h3>最佳策略推荐</h3>
            <ul>
                <li><strong>最高年化收益</strong>：{best_return[0]}（{best_return[1]['metrics']['annualized_return']:.1f}%）</li>
                <li><strong>最佳风险调整收益（夏普比率）</strong>：{best_sharpe[0]}（{best_sharpe[1]['metrics']['sharpe_ratio']:.2f}）</li>
                <li><strong>最小最大回撤</strong>：{best_dd[0]}（{best_dd[1]['metrics']['max_drawdown']:.1f}%）</li>
                <li><strong>最佳卡尔马比率</strong>：{best_calmar[0]}（{best_calmar[1]['metrics']['calmar_ratio']:.2f}）</li>
            </ul>
        </div>

        <div class="insight insight-green">
            <h3>关键发现</h3>
            <ul>
                <li><strong>收益方面</strong>：Buy & Hold在长牛市中获得最高绝对收益，但承担最大回撤风险</li>
                <li><strong>风险控制</strong>：减仓策略（Fixed 30%/50%）有效降低最大回撤，代价是牺牲部分上涨收益</li>
                <li><strong>渐进加仓（v2策略）</strong>：相比简单等待回到前高，在反弹过程中逐步加仓能更好地捕捉复苏行情</li>
                <li><strong>权衡取舍</strong>：没有完美策略，需根据个人风险承受能力选择</li>
            </ul>
        </div>

        <div class="insight insight-orange">
            <h3>策略对比核心数据</h3>
            <p>以下对比基于{start_date}至{end_date}的完整回测期间：</p>
            <ul>
                <li>Buy & Hold累计收益 <strong>{results['Buy & Hold']['metrics']['total_return']:.1f}%</strong>，最大回撤 <strong>{results['Buy & Hold']['metrics']['max_drawdown']:.1f}%</strong></li>
                <li>Fixed 50%累计收益 <strong>{results['Fixed 50%']['metrics']['total_return']:.1f}%</strong>，最大回撤 <strong>{results['Fixed 50%']['metrics']['max_drawdown']:.1f}%</strong></li>
                <li>Fixed 30%累计收益 <strong>{results['Fixed 30%']['metrics']['total_return']:.1f}%</strong>，最大回撤 <strong>{results['Fixed 30%']['metrics']['max_drawdown']:.1f}%</strong></li>
            </ul>
        </div>

        <div class="insight insight-red">
            <h3>风险提示与实战建议</h3>
            <ul>
                <li><strong>回测局限性</strong>：历史数据不能保证未来表现。2009-2025是美股史上最长牛市之一，可能高估Buy & Hold优势</li>
                <li><strong>择时风险</strong>：10%信号触发时，无法确定是否会继续下跌。可能出现"刚减仓就反弹"的情况</li>
                <li><strong>执行纪律</strong>：策略成功的关键是严格执行，避免情绪化操作</li>
                <li><strong>个性化调整</strong>：可根据个人情况调整减仓比例和加仓节奏</li>
            </ul>
        </div>
    </div>

    <!-- ===== Strategy Definitions ===== -->
    <div class="section">
        <h2>策略定义</h2>

        <div class="strategy-box" style="border-color:#2c3e50">
            <h4 style="color:#2c3e50">1. Buy & Hold（基准）</h4>
            <p>始终持有100%仓位，不做任何调整。作为所有策略的对比基准。</p>
        </div>

        <div class="strategy-box" style="border-color:#3498db">
            <h4 style="color:#3498db">2. Fixed 50%</h4>
            <p><strong>触发条件</strong>：指数从高点下跌达到10%<br>
            <strong>操作</strong>：减仓至50%仓位<br>
            <strong>恢复条件</strong>：指数回到前期高点时恢复100%仓位</p>
        </div>

        <div class="strategy-box" style="border-color:#e74c3c">
            <h4 style="color:#e74c3c">3. Fixed 30%</h4>
            <p><strong>触发条件</strong>：指数从高点下跌达到10%<br>
            <strong>操作</strong>：减仓至30%仓位<br>
            <strong>恢复条件</strong>：指数回到前期高点时恢复100%仓位</p>
        </div>

        <div class="strategy-box" style="border-color:#27ae60">
            <h4 style="color:#27ae60">4. Fixed 50% v2（渐进加仓版）</h4>
            <p><strong>触发条件</strong>：指数从高点下跌达到10%<br>
            <strong>初始操作</strong>：减仓至50%仓位<br>
            <strong>加仓规则</strong>：市场从底部每反弹20%，加仓10%（如反弹20%→60%仓位，反弹40%→70%仓位...）<br>
            <strong>恢复条件</strong>：逐步加仓或回到前高时达到100%</p>
        </div>

        <div class="strategy-box" style="border-color:#9b59b6">
            <h4 style="color:#9b59b6">5. Fixed 30% v2（渐进加仓版）</h4>
            <p><strong>触发条件</strong>：指数从高点下跌达到10%<br>
            <strong>初始操作</strong>：减仓至30%仓位<br>
            <strong>加仓规则</strong>：市场从底部每反弹20%，加仓14%（逐步恢复70%的仓位缺口）<br>
            <strong>恢复条件</strong>：逐步加仓或回到前高时达到100%</p>
        </div>
    </div>

    <!-- ===== Performance Table ===== -->
    <div class="section">
        <h2>策略表现对比</h2>
        <div style="overflow-x:auto;">
        <table>
            <thead>
                <tr>
                    <th>策略</th>
                    <th>累计收益</th>
                    <th>年化收益</th>
                    <th>年化波动率</th>
                    <th>最大回撤</th>
                    <th>夏普比率</th>
                    <th>卡尔马比率</th>
                    <th>索提诺比率</th>
                    <th>最终价值</th>
                    <th>调仓次数</th>
                </tr>
            </thead>
            <tbody>
                {perf_rows}
            </tbody>
        </table>
        </div>

        <h3 style="margin-top:20px;">调仓统计</h3>
        <ul>
            {trade_summary}
        </ul>
    </div>

    <!-- ===== Charts ===== -->
    <div class="section">
        <h2>累计收益曲线</h2>
        <div class="chart-container">{charts['cumulative']}</div>
    </div>

    <div class="section">
        <h2>回撤对比</h2>
        <p style="color:#7f8c8d;margin-bottom:15px;">回撤是衡量策略风险的核心指标，反映投资者可能面临的最大亏损</p>
        <div class="chart-container">{charts['drawdown']}</div>
    </div>

    <div class="section">
        <h2>仓位变化</h2>
        <div class="chart-container">{charts['position']}</div>
    </div>

    <div class="section">
        <h2>风险收益分析</h2>
        <div class="chart-container">{charts['scatter']}</div>
    </div>

    <div class="section">
        <h2>年度收益对比</h2>
        <div class="chart-container">{charts['annual']}</div>
    </div>

    <div class="section">
        <h2>综合评分雷达图</h2>
        <div class="chart-container">{charts['radar']}</div>
    </div>

    <div class="section">
        <h2>调仓时点详图</h2>
        <div class="chart-container">{charts['trades']}</div>
    </div>

    <div class="section" style="text-align:center; color:#95a5a6; font-size:0.9em;">
        <p>本报告仅供研究参考，不构成投资建议 | 数据来源：S&P 500历史数据</p>
        <p>回测区间：{start_date} 至 {end_date} | 无交易费用假设</p>
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
    print("Task 2: Dynamic Position Management Strategy Backtesting")
    print("5 Strategies | 10% Drawdown Signal | No Transaction Costs")
    print("=" * 65)

    print("\n[1/5] Loading data...")
    df_full = load_data('SP500.csv')
    df = get_backtest_period(df_full)
    print(f"  Backtest: {df['Date'].iloc[0].strftime('%Y-%m-%d')} to {df['Date'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"  Trading days: {len(df)}")

    prices = df['Close'].values
    dates = df['Date'].values
    dates_ts = pd.to_datetime(dates)

    print("\n[2/5] Running strategies...")
    results = {}

    print("  1. Buy & Hold...")
    p, t, pos = strategy_buy_and_hold(prices, dates_ts)
    results['Buy & Hold'] = {'portfolio': p, 'trades': t, 'positions': pos,
                             'metrics': compute_performance_metrics(p, dates_ts)}

    print("  2. Fixed 50%...")
    p, t, pos = strategy_fixed_50(prices, dates_ts)
    results['Fixed 50%'] = {'portfolio': p, 'trades': t, 'positions': pos,
                            'metrics': compute_performance_metrics(p, dates_ts)}

    print("  3. Fixed 30%...")
    p, t, pos = strategy_fixed_30(prices, dates_ts)
    results['Fixed 30%'] = {'portfolio': p, 'trades': t, 'positions': pos,
                            'metrics': compute_performance_metrics(p, dates_ts)}

    print("  4. Fixed 50% v2...")
    p, t, pos = strategy_fixed_50_v2(prices, dates_ts)
    results['Fixed 50% v2'] = {'portfolio': p, 'trades': t, 'positions': pos,
                               'metrics': compute_performance_metrics(p, dates_ts)}

    print("  5. Fixed 30% v2...")
    p, t, pos = strategy_fixed_30_v2(prices, dates_ts)
    results['Fixed 30% v2'] = {'portfolio': p, 'trades': t, 'positions': pos,
                               'metrics': compute_performance_metrics(p, dates_ts)}

    print("\n[3/5] Performance Summary:")
    print(f"  {'Strategy':<15} {'Total':>10} {'Annual':>10} {'MaxDD':>10} {'Sharpe':>8} {'Trades':>8}")
    print("  " + "-" * 65)
    for name, data in results.items():
        m = data['metrics']
        n_trades = len(data['trades'])
        print(f"  {name:<15} {m['total_return']:>9.1f}% {m['annualized_return']:>9.1f}% {m['max_drawdown']:>9.1f}% {m['sharpe_ratio']:>7.2f} {n_trades:>8}")

    print("\n[4/5] Saving data...")
    os.makedirs('outputs', exist_ok=True)

    perf_data = [{**data['metrics'], 'strategy': name} for name, data in results.items()]
    pd.DataFrame(perf_data).to_csv('outputs/task2_performance.csv', index=False, encoding='utf-8-sig')
    print("  Saved outputs/task2_performance.csv")

    all_trades = []
    for name, data in results.items():
        all_trades.extend(data['trades'])
    if all_trades:
        pd.DataFrame(all_trades).to_csv('outputs/task2_trades.csv', index=False, encoding='utf-8-sig')
    print(f"  Saved outputs/task2_trades.csv ({len(all_trades)} trades)")

    print("\n[5/5] Generating HTML report...")
    html = generate_html_report(dates_ts, prices, results)
    with open('outputs/task2_backtest.html', 'w', encoding='utf-8') as f:
        f.write(html)
    print("  Saved outputs/task2_backtest.html")

    print("\n" + "=" * 65)
    print("Task 2 Complete!")
    print("=" * 65)


if __name__ == '__main__':
    main()
