#!/usr/bin/env python3
"""
Task 2: Dynamic Position Management Strategy Backtesting
Compares multiple position management strategies against buy-and-hold benchmark.
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
    """
    Use the period from March 2009 (end of GFC) to Aug 2025.
    The README says 'from end of last major crash to present'.
    """
    start_date = '2009-03-10'
    df_bt = df[df['Date'] >= start_date].copy().reset_index(drop=True)
    return df_bt


# ============================================================
# 2. Helper Functions
# ============================================================

def compute_daily_returns(prices):
    return np.diff(prices) / prices[:-1]


def compute_drawdown_from_peak(prices):
    """Compute running drawdown from peak for a price series."""
    running_max = np.maximum.accumulate(prices)
    dd = (prices - running_max) / running_max
    return dd


def compute_performance_metrics(portfolio_values, dates, initial_capital=100000, risk_free_rate=0.02):
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

TRANSACTION_COST = 0.001  # 0.1%


def strategy_buy_and_hold(prices, dates, initial_capital=100000):
    """Benchmark: Always 100% invested."""
    n = len(prices)
    portfolio = np.zeros(n)
    portfolio[0] = initial_capital
    position = 1.0  # 100% position

    trades = []
    shares = initial_capital / prices[0]
    cash = 0.0

    for i in range(1, n):
        daily_ret = (prices[i] - prices[i-1]) / prices[i-1]
        portfolio[i] = shares * prices[i] + cash

    return portfolio, trades, np.ones(n)  # positions always 1.0


def strategy_a_fixed_reduction(prices, dates, initial_capital=100000):
    """
    Strategy A - Fixed Reduction Method:
    - Drawdown 5%: reduce to 80%
    - Drawdown 10%: reduce to 60%
    - Drawdown 15%: reduce to 40%
    - Drawdown 20%+: reduce to 20%
    - When price recovers above 95% of peak: back to 100%
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

    for i in range(1, n):
        if prices[i] > peak:
            peak = prices[i]

        dd = (prices[i] - peak) / peak

        # Determine target position
        if dd > -0.05:
            target_pos = 1.0
        elif dd > -0.10:
            target_pos = 0.80
        elif dd > -0.15:
            target_pos = 0.60
        elif dd > -0.20:
            target_pos = 0.40
        else:
            target_pos = 0.20

        # Adjust position if changed
        if abs(target_pos - current_pos) > 0.01:
            total_value = shares * prices[i] + cash
            cost = abs(target_pos - current_pos) * total_value * TRANSACTION_COST
            total_value -= cost
            shares = (target_pos * total_value) / prices[i]
            cash = total_value * (1 - target_pos)
            trades.append({
                'date': dates[i],
                'action': 'REDUCE' if target_pos < current_pos else 'INCREASE',
                'old_position': current_pos,
                'new_position': target_pos,
                'price': prices[i],
                'drawdown_pct': dd * 100,
                'portfolio_value': total_value,
                'transaction_cost': cost,
                'strategy': 'Strategy A'
            })
            current_pos = target_pos

        portfolio[i] = shares * prices[i] + cash
        positions[i] = current_pos

    return portfolio, trades, positions


def strategy_b_gradual_rebalance(prices, dates, initial_capital=100000):
    """
    Strategy B - Gradual Rebalancing Method:
    - Start 100%
    - When drawdown starts (>2%), immediately reduce to 50%
    - After trough detected (price starts rising from >10% drawdown), gradually add back
    - Add 10% position for every 5% recovery from trough
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
    in_drawdown = False
    recovering = False

    for i in range(1, n):
        if prices[i] > peak:
            peak = prices[i]
            trough = prices[i]
            in_drawdown = False
            recovering = False

        dd = (prices[i] - peak) / peak

        if prices[i] < trough:
            trough = prices[i]
            recovering = False

        # Detect recovery from trough
        if trough < peak and prices[i] > trough:
            recovery_from_trough = (prices[i] - trough) / trough
        else:
            recovery_from_trough = 0

        target_pos = current_pos

        if dd < -0.02 and not in_drawdown and not recovering:
            # Drawdown starts, reduce to 50%
            in_drawdown = True
            target_pos = 0.50
        elif in_drawdown and dd < -0.10 and recovery_from_trough > 0.05:
            # Start recovering
            recovering = True
            in_drawdown = False
            # Gradually add back based on recovery
            recovery_pct = recovery_from_trough
            add_back = min(0.50, (recovery_pct / 0.05) * 0.10)
            target_pos = min(1.0, 0.50 + add_back)
        elif recovering:
            recovery_pct = recovery_from_trough
            add_back = min(0.50, (recovery_pct / 0.05) * 0.10)
            target_pos = min(1.0, 0.50 + add_back)
        elif dd > -0.02:
            target_pos = 1.0

        target_pos = max(0.0, min(1.0, target_pos))

        if abs(target_pos - current_pos) > 0.05:
            total_value = shares * prices[i] + cash
            cost = abs(target_pos - current_pos) * total_value * TRANSACTION_COST
            total_value -= cost
            shares = (target_pos * total_value) / prices[i]
            cash = total_value * (1 - target_pos)
            trades.append({
                'date': dates[i],
                'action': 'REDUCE' if target_pos < current_pos else 'INCREASE',
                'old_position': current_pos,
                'new_position': target_pos,
                'price': prices[i],
                'drawdown_pct': dd * 100,
                'portfolio_value': total_value,
                'transaction_cost': cost,
                'strategy': 'Strategy B'
            })
            current_pos = target_pos

        portfolio[i] = shares * prices[i] + cash
        positions[i] = current_pos

    return portfolio, trades, positions


def strategy_c_moving_stop(prices, dates, initial_capital=100000):
    """
    Strategy C - Dynamic Hedging with Moving Stop:
    - Use trailing stop: if price falls below 200-day MA, reduce position
    - Use volatility-based position sizing
    - Position = min(1.0, target_vol / realized_vol)
    - If below 200MA: further reduce by 50%
    """
    n = len(prices)
    portfolio = np.zeros(n)
    positions = np.zeros(n)
    portfolio[0] = initial_capital
    positions[0] = 1.0

    trades = []
    shares = initial_capital / prices[0]
    cash = 0.0
    current_pos = 1.0
    target_vol = 0.15  # Target 15% annual volatility

    for i in range(1, n):
        # Compute 200-day MA
        lookback_200 = max(0, i - 200)
        ma200 = np.mean(prices[lookback_200:i+1])

        # Compute 20-day realized volatility
        lookback_20 = max(1, i - 20)
        if i > lookback_20:
            recent_returns = np.diff(prices[lookback_20:i+1]) / prices[lookback_20:i]
            realized_vol = np.std(recent_returns) * np.sqrt(252) if len(recent_returns) > 1 else 0.15
        else:
            realized_vol = 0.15

        # Vol-based position sizing
        vol_pos = min(1.0, target_vol / max(realized_vol, 0.01))

        # MA filter
        if prices[i] < ma200:
            target_pos = vol_pos * 0.50  # Half position below 200MA
        else:
            target_pos = vol_pos

        target_pos = max(0.0, min(1.0, target_pos))

        if abs(target_pos - current_pos) > 0.05:
            total_value = shares * prices[i] + cash
            cost = abs(target_pos - current_pos) * total_value * TRANSACTION_COST
            total_value -= cost
            shares = (target_pos * total_value) / prices[i]
            cash = total_value * (1 - target_pos)
            trades.append({
                'date': dates[i],
                'action': 'REDUCE' if target_pos < current_pos else 'INCREASE',
                'old_position': current_pos,
                'new_position': target_pos,
                'price': prices[i],
                'drawdown_pct': 0,
                'portfolio_value': total_value,
                'transaction_cost': cost,
                'strategy': 'Strategy C'
            })
            current_pos = target_pos

        portfolio[i] = shares * prices[i] + cash
        positions[i] = current_pos

    return portfolio, trades, positions


def strategy_d_momentum_mean_reversion(prices, dates, initial_capital=100000):
    """
    Strategy D - Custom Multi-Factor Strategy:
    - Momentum signal: 12-month return (positive = bullish)
    - Mean reversion signal: RSI-based (oversold = buy, overbought = cautious)
    - Trend signal: price vs 50-day and 200-day MA
    - Combined scoring system to determine position
    """
    n = len(prices)
    portfolio = np.zeros(n)
    positions = np.zeros(n)
    portfolio[0] = initial_capital
    positions[0] = 1.0

    trades = []
    shares = initial_capital / prices[0]
    cash = 0.0
    current_pos = 1.0

    for i in range(1, n):
        score = 0.0  # -1 to +1 scale

        # Factor 1: 252-day momentum
        if i >= 252:
            momentum = (prices[i] / prices[i-252]) - 1
            if momentum > 0.10:
                score += 0.3
            elif momentum > 0:
                score += 0.15
            elif momentum > -0.10:
                score -= 0.1
            else:
                score -= 0.3

        # Factor 2: RSI (14-day)
        if i >= 14:
            changes = np.diff(prices[max(0, i-14):i+1])
            gains = np.mean(changes[changes > 0]) if np.any(changes > 0) else 0
            losses = -np.mean(changes[changes < 0]) if np.any(changes < 0) else 0.001
            rs = gains / max(losses, 0.001)
            rsi = 100 - (100 / (1 + rs))

            if rsi < 30:  # Oversold - contrarian buy
                score += 0.2
            elif rsi < 40:
                score += 0.1
            elif rsi > 70:  # Overbought - cautious
                score -= 0.1
            elif rsi > 80:
                score -= 0.2

        # Factor 3: Trend (50MA and 200MA)
        if i >= 200:
            ma50 = np.mean(prices[i-50:i+1])
            ma200 = np.mean(prices[i-200:i+1])

            if prices[i] > ma50 > ma200:
                score += 0.3  # Strong uptrend
            elif prices[i] > ma200:
                score += 0.15  # Above long-term trend
            elif prices[i] < ma50 < ma200:
                score -= 0.3  # Strong downtrend
            else:
                score -= 0.1

        # Factor 4: Short-term volatility adjustment
        if i >= 20:
            recent_returns = np.diff(prices[i-20:i+1]) / prices[i-20:i]
            vol = np.std(recent_returns) * np.sqrt(252)
            if vol > 0.30:  # High volatility
                score -= 0.15
            elif vol > 0.25:
                score -= 0.05

        # Convert score to position (map [-1, 1] to [0.2, 1.0])
        target_pos = max(0.2, min(1.0, 0.6 + score * 0.5))

        if abs(target_pos - current_pos) > 0.05:
            total_value = shares * prices[i] + cash
            cost = abs(target_pos - current_pos) * total_value * TRANSACTION_COST
            total_value -= cost
            shares = (target_pos * total_value) / prices[i]
            cash = total_value * (1 - target_pos)
            trades.append({
                'date': dates[i],
                'action': 'REDUCE' if target_pos < current_pos else 'INCREASE',
                'old_position': current_pos,
                'new_position': target_pos,
                'price': prices[i],
                'drawdown_pct': 0,
                'portfolio_value': total_value,
                'transaction_cost': cost,
                'strategy': 'Strategy D'
            })
            current_pos = target_pos

        portfolio[i] = shares * prices[i] + cash
        positions[i] = current_pos

    return portfolio, trades, positions


# ============================================================
# 4. Visualization Functions
# ============================================================

def create_cumulative_returns_chart(dates, results):
    """Cumulative return comparison chart."""
    fig = go.Figure()
    colors = {'Buy & Hold': '#1f77b4', 'Strategy A': '#ff7f0e',
              'Strategy B': '#2ca02c', 'Strategy C': '#d62728', 'Strategy D': '#9467bd'}

    for name, data in results.items():
        normalized = data['portfolio'] / data['portfolio'][0] * 100
        fig.add_trace(go.Scatter(
            x=dates, y=normalized,
            mode='lines', name=name,
            line=dict(color=colors.get(name, '#333'), width=2)
        ))

    fig.update_layout(
        title='累计收益曲线对比（初始$100,000）',
        xaxis_title='日期', yaxis_title='投资组合价值（标准化=100）',
        template='plotly_white', height=500,
        hovermode='x unified',
        legend=dict(x=0.01, y=0.99)
    )
    return fig


def create_drawdown_comparison_chart(dates, results):
    """Drawdown comparison chart."""
    fig = go.Figure()
    colors = {'Buy & Hold': '#1f77b4', 'Strategy A': '#ff7f0e',
              'Strategy B': '#2ca02c', 'Strategy C': '#d62728', 'Strategy D': '#9467bd'}

    for name, data in results.items():
        dd = compute_drawdown_from_peak(data['portfolio']) * 100
        fig.add_trace(go.Scatter(
            x=dates, y=dd,
            mode='lines', name=name,
            line=dict(color=colors.get(name, '#333'), width=1.5),
            fill='tozeroy' if name == 'Buy & Hold' else None
        ))

    fig.update_layout(
        title='回撤对比图',
        xaxis_title='日期', yaxis_title='回撤 (%)',
        template='plotly_white', height=450,
        hovermode='x unified'
    )
    return fig


def create_position_chart(dates, results):
    """Position allocation over time."""
    fig = go.Figure()
    colors = {'Strategy A': '#ff7f0e', 'Strategy B': '#2ca02c',
              'Strategy C': '#d62728', 'Strategy D': '#9467bd'}

    for name, data in results.items():
        if name == 'Buy & Hold':
            continue
        fig.add_trace(go.Scatter(
            x=dates, y=data['positions'] * 100,
            mode='lines', name=name,
            line=dict(color=colors.get(name, '#333'), width=1)
        ))

    fig.update_layout(
        title='仓位变化时间序列',
        xaxis_title='日期', yaxis_title='仓位 (%)',
        template='plotly_white', height=400,
        hovermode='x unified'
    )
    return fig


def create_rolling_returns_chart(dates, results, window=252):
    """Rolling annual returns comparison."""
    fig = go.Figure()
    colors = {'Buy & Hold': '#1f77b4', 'Strategy A': '#ff7f0e',
              'Strategy B': '#2ca02c', 'Strategy C': '#d62728', 'Strategy D': '#9467bd'}

    for name, data in results.items():
        if len(data['portfolio']) > window:
            rolling_ret = (data['portfolio'][window:] / data['portfolio'][:-window] - 1) * 100
            fig.add_trace(go.Scatter(
                x=dates[window:], y=rolling_ret,
                mode='lines', name=name,
                line=dict(color=colors.get(name, '#333'), width=1.5)
            ))

    fig.update_layout(
        title=f'滚动{window}日年化收益率对比',
        xaxis_title='日期', yaxis_title='滚动收益率 (%)',
        template='plotly_white', height=400,
        hovermode='x unified'
    )
    return fig


def create_risk_return_scatter(results):
    """Risk-return scatter plot."""
    fig = go.Figure()
    colors = {'Buy & Hold': '#1f77b4', 'Strategy A': '#ff7f0e',
              'Strategy B': '#2ca02c', 'Strategy C': '#d62728', 'Strategy D': '#9467bd'}

    for name, data in results.items():
        m = data['metrics']
        fig.add_trace(go.Scatter(
            x=[m['annual_volatility']], y=[m['annualized_return']],
            mode='markers+text', name=name,
            marker=dict(size=15, color=colors.get(name, '#333')),
            text=[name], textposition='top center'
        ))

    fig.update_layout(
        title='风险收益散点图',
        xaxis_title='年化波动率 (%)', yaxis_title='年化收益率 (%)',
        template='plotly_white', height=450
    )
    return fig


def create_annual_returns_chart(dates, results):
    """Annual returns comparison bar chart."""
    fig = go.Figure()
    colors = {'Buy & Hold': '#1f77b4', 'Strategy A': '#ff7f0e',
              'Strategy B': '#2ca02c', 'Strategy C': '#d62728', 'Strategy D': '#9467bd'}

    for name, data in results.items():
        portfolio = data['portfolio']
        # Compute annual returns
        years = sorted(set(d.year for d in dates))
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
            name=name, marker_color=colors.get(name, '#333')
        ))

    fig.update_layout(
        title='年度收益对比柱状图',
        xaxis_title='年份', yaxis_title='收益率 (%)',
        barmode='group',
        template='plotly_white', height=500
    )
    return fig


def create_radar_chart(results):
    """Strategy performance radar chart."""
    categories = ['年化收益率', '夏普比率', '卡尔马比率', '胜率', '最大回撤控制']

    fig = go.Figure()
    colors = {'Buy & Hold': '#1f77b4', 'Strategy A': '#ff7f0e',
              'Strategy B': '#2ca02c', 'Strategy C': '#d62728', 'Strategy D': '#9467bd'}

    for name, data in results.items():
        m = data['metrics']

        # Normalize each metric to 0-1 scale for radar
        values = [
            min(m['annualized_return'] / 20, 1.0),  # Normalize annual return (cap at 20%)
            min(max(m['sharpe_ratio'], 0) / 2.0, 1.0),  # Normalize sharpe (cap at 2.0)
            min(max(m['calmar_ratio'], 0) / 2.0, 1.0),  # Normalize calmar (cap at 2.0)
            m['win_rate'] / 100,  # Already 0-100
            1 + m['max_drawdown'] / 100,  # Convert max DD (smaller is better)
        ]
        values.append(values[0])  # Close the radar

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            name=name,
            line=dict(color=colors.get(name, '#333'))
        ))

    fig.update_layout(
        title='策略表现综合评分雷达图',
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        template='plotly_white', height=500
    )
    return fig


def create_trade_annotation_chart(dates, results, prices):
    """Price chart with trade annotations for each strategy."""
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=['Strategy A 调仓时点', 'Strategy B 调仓时点',
                                        'Strategy C 调仓时点', 'Strategy D 调仓时点'],
                        vertical_spacing=0.05)

    strategy_names = ['Strategy A', 'Strategy B', 'Strategy C', 'Strategy D']
    colors = {'Strategy A': '#ff7f0e', 'Strategy B': '#2ca02c',
              'Strategy C': '#d62728', 'Strategy D': '#9467bd'}

    for idx, sname in enumerate(strategy_names, 1):
        if sname not in results:
            continue
        data = results[sname]

        fig.add_trace(go.Scatter(
            x=dates, y=prices,
            mode='lines', name='S&P 500',
            line=dict(color='#ccc', width=1),
            showlegend=(idx == 1)
        ), row=idx, col=1)

        # Mark buy and sell points
        for trade in data['trades']:
            color = 'green' if trade['action'] == 'INCREASE' else 'red'
            symbol = 'triangle-up' if trade['action'] == 'INCREASE' else 'triangle-down'
            fig.add_trace(go.Scatter(
                x=[trade['date']], y=[trade['price']],
                mode='markers',
                marker=dict(color=color, size=6, symbol=symbol),
                showlegend=False,
                hovertext=f"{trade['action']}: {trade['old_position']:.0%} -> {trade['new_position']:.0%}"
            ), row=idx, col=1)

    fig.update_layout(
        height=1200, template='plotly_white',
        title='各策略调仓时点标注'
    )
    return fig


def create_pnl_distribution_chart(results):
    """P&L distribution histogram for each strategy."""
    fig = make_subplots(rows=1, cols=len(results),
                        subplot_titles=list(results.keys()))

    colors = {'Buy & Hold': '#1f77b4', 'Strategy A': '#ff7f0e',
              'Strategy B': '#2ca02c', 'Strategy C': '#d62728', 'Strategy D': '#9467bd'}

    for idx, (name, data) in enumerate(results.items(), 1):
        daily_returns = np.diff(data['portfolio']) / data['portfolio'][:-1] * 100
        fig.add_trace(go.Histogram(
            x=daily_returns, nbinsx=100,
            name=name, marker_color=colors.get(name, '#333'),
            opacity=0.7
        ), row=1, col=idx)

    fig.update_layout(
        title='日收益率分布直方图',
        template='plotly_white', height=400
    )
    return fig


# ============================================================
# 5. Report Generation
# ============================================================

def generate_html_report(dates, prices, results):
    """Generate comprehensive HTML backtest report."""

    # Create all charts
    fig_cumulative = create_cumulative_returns_chart(dates, results)
    fig_drawdown = create_drawdown_comparison_chart(dates, results)
    fig_position = create_position_chart(dates, results)
    fig_rolling = create_rolling_returns_chart(dates, results)
    fig_scatter = create_risk_return_scatter(results)
    fig_annual = create_annual_returns_chart(dates, results)
    fig_radar = create_radar_chart(results)
    fig_trades = create_trade_annotation_chart(dates, results, prices)
    fig_pnl = create_pnl_distribution_chart(results)

    charts = {
        'cumulative': fig_cumulative.to_html(full_html=False, include_plotlyjs=False),
        'drawdown': fig_drawdown.to_html(full_html=False, include_plotlyjs=False),
        'position': fig_position.to_html(full_html=False, include_plotlyjs=False),
        'rolling': fig_rolling.to_html(full_html=False, include_plotlyjs=False),
        'scatter': fig_scatter.to_html(full_html=False, include_plotlyjs=False),
        'annual': fig_annual.to_html(full_html=False, include_plotlyjs=False),
        'radar': fig_radar.to_html(full_html=False, include_plotlyjs=False),
        'trades': fig_trades.to_html(full_html=False, include_plotlyjs=False),
        'pnl': fig_pnl.to_html(full_html=False, include_plotlyjs=False),
    }

    # Build performance comparison table
    perf_rows = ""
    for name, data in results.items():
        m = data['metrics']
        perf_rows += f"""
        <tr>
            <td style="font-weight:bold">{name}</td>
            <td>{m['total_return']:.1f}%</td>
            <td>{m['annualized_return']:.1f}%</td>
            <td>{m['annual_volatility']:.1f}%</td>
            <td style="color:red">{m['max_drawdown']:.1f}%</td>
            <td>{m['sharpe_ratio']:.2f}</td>
            <td>{m['calmar_ratio']:.2f}</td>
            <td>{m['sortino_ratio']:.2f}</td>
            <td>{m['win_rate']:.1f}%</td>
            <td>${m['final_value']:,.0f}</td>
        </tr>"""

    # Determine best strategy
    best_sharpe = max(results.items(), key=lambda x: x[1]['metrics']['sharpe_ratio'])
    best_return = max(results.items(), key=lambda x: x[1]['metrics']['annualized_return'])
    best_dd = max(results.items(), key=lambda x: x[1]['metrics']['max_drawdown'])  # least negative
    best_calmar = max(results.items(), key=lambda x: x[1]['metrics']['calmar_ratio'])

    # Trade counts
    trade_summary = ""
    for name, data in results.items():
        if name == 'Buy & Hold':
            continue
        n_trades = len(data['trades'])
        total_cost = sum(t['transaction_cost'] for t in data['trades'])
        trade_summary += f"<li><strong>{name}</strong>: {n_trades}次调仓，总交易成本 ${total_cost:,.2f}</li>"

    start_date = dates[0].strftime('%Y-%m-%d')
    end_date = dates[-1].strftime('%Y-%m-%d')

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>标普500动态仓位管理策略回测报告</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f7fa; color: #333; line-height: 1.6; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #0f3460 0%, #16213e 50%, #1a1a2e 100%); color: white; padding: 40px; border-radius: 12px; margin-bottom: 30px; }}
        .header h1 {{ font-size: 2em; margin-bottom: 10px; }}
        .header p {{ font-size: 1.1em; opacity: 0.9; }}
        .summary-box {{ background: white; border-radius: 12px; padding: 30px; margin-bottom: 30px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }}
        .summary-box h2 {{ color: #1a1a2e; border-bottom: 3px solid #0f3460; padding-bottom: 10px; margin-bottom: 20px; }}
        .chart-container {{ background: white; border-radius: 12px; padding: 20px; margin-bottom: 30px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 0.85em; }}
        th {{ background: #1a1a2e; color: white; padding: 12px 8px; text-align: center; }}
        td {{ padding: 10px 8px; text-align: center; border-bottom: 1px solid #eee; }}
        tr:hover {{ background: #f8f9fa; }}
        .conclusion {{ background: #e8f5e9; border-left: 5px solid #4caf50; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .conclusion h3 {{ color: #2e7d32; margin-bottom: 10px; }}
        .warning {{ background: #fff3e0; border-left: 5px solid #ff9800; padding: 15px; border-radius: 8px; margin: 15px 0; }}
        .key-finding {{ background: #e3f2fd; border-left: 5px solid #2196f3; padding: 15px; border-radius: 8px; margin: 15px 0; }}
        .strategy-desc {{ background: #f3e5f5; border-left: 5px solid #9c27b0; padding: 15px; border-radius: 8px; margin: 15px 0; }}
        ul {{ padding-left: 20px; }}
        li {{ margin: 8px 0; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .stat-card {{ background: linear-gradient(135deg, #f8f9fa, #e9ecef); padding: 15px; border-radius: 10px; text-align: center; border-left: 4px solid #0f3460; }}
        .stat-card .number {{ font-size: 1.5em; font-weight: bold; color: #0f3460; }}
        .stat-card .label {{ font-size: 0.85em; color: #666; }}
    </style>
</head>
<body>
<div class="container">

    <div class="header">
        <h1>标普500动态仓位管理策略回测报告</h1>
        <p>回测区间：{start_date} 至 {end_date} | 初始资金：$100,000 | 交易成本：0.1%</p>
        <p>报告生成日期：{datetime.now().strftime('%Y-%m-%d')}</p>
    </div>

    <!-- ===== 研究结论 ===== -->
    <div class="summary-box">
        <h2>研究结论</h2>

        <div class="key-finding">
            <h3>各策略收益与风险对比</h3>
            <ul>
                <li><strong>最高年化收益</strong>：{best_return[0]}（{best_return[1]['metrics']['annualized_return']:.1f}%）</li>
                <li><strong>最佳风险调整收益（夏普比率）</strong>：{best_sharpe[0]}（{best_sharpe[1]['metrics']['sharpe_ratio']:.2f}）</li>
                <li><strong>最小最大回撤</strong>：{best_dd[0]}（{best_dd[1]['metrics']['max_drawdown']:.1f}%）</li>
                <li><strong>最佳卡尔马比率</strong>：{best_calmar[0]}（{best_calmar[1]['metrics']['calmar_ratio']:.2f}）</li>
            </ul>
        </div>

        <div class="conclusion">
            <h3>最优仓位调整策略推荐</h3>
            <p>综合考虑收益率、风险控制和实际可操作性，<strong>{best_sharpe[0]}</strong>
            在风险调整后收益方面表现最佳（夏普比率 {best_sharpe[1]['metrics']['sharpe_ratio']:.2f}），
            而<strong>{best_dd[0]}</strong>在控制回撤方面最为出色（最大回撤 {best_dd[1]['metrics']['max_drawdown']:.1f}%）。</p>
            <p>投资者应根据自身风险偏好选择合适的策略：</p>
            <ul>
                <li>保守型投资者可优先考虑回撤控制较好的策略</li>
                <li>进取型投资者可选择收益最大化的策略</li>
                <li>平衡型投资者推荐夏普比率最高的策略</li>
            </ul>
        </div>

        <div class="conclusion">
            <h3>动态管理 vs Buy and Hold 对比</h3>
            <ul>
                <li><strong>优势</strong>：动态仓位管理策略在市场回调期间能有效降低回撤，提供更好的风险调整后收益和更平滑的收益曲线。</li>
                <li><strong>劣势</strong>：交易成本侵蚀部分收益；择时判断可能不完美，在快速反弹中可能踏空；需要更多的管理精力和纪律性。</li>
                <li><strong>核心权衡</strong>：减少下跌时的暴露通常也意味着减少上涨时的参与度。没有策略能完美预测市场转折点。</li>
            </ul>
        </div>

        <div class="warning">
            <h3>不同市场环境下的策略适用性</h3>
            <ul>
                <li><strong>牛市环境</strong>：Buy and Hold通常表现最好，动态策略可能因频繁减仓而落后。</li>
                <li><strong>熊市/回调</strong>：动态减仓策略（如策略A）能显著降低损失。</li>
                <li><strong>震荡市</strong>：波动率调整策略（如策略C）可能更具优势。</li>
                <li><strong>V型反弹</strong>：渐进加仓策略（如策略B）能更好地捕捉反弹。</li>
            </ul>
        </div>

        <div class="warning">
            <h3>实战建议与风险提示</h3>
            <ul>
                <li>回测结果基于历史数据，不代表未来表现。过去的回报不保证未来收益。</li>
                <li>实际交易中存在滑点、流动性风险等额外成本。</li>
                <li>建议根据个人风险承受能力和投资期限调整策略参数。</li>
                <li>任何策略都无法完全消除市场风险，做好最坏情况的心理准备。</li>
                <li>交易成本（含滑点）可能高于本回测中假设的0.1%。</li>
            </ul>
        </div>
    </div>

    <!-- ===== Strategy Descriptions ===== -->
    <div class="summary-box">
        <h2>策略说明</h2>

        <div class="strategy-desc">
            <h3>基准策略 - Buy & Hold</h3>
            <p>始终持有100%标普500仓位，不做任何调整。作为所有策略的对比基准。</p>
        </div>

        <div class="strategy-desc">
            <h3>策略A - 固定减仓法</h3>
            <p>根据从峰值的回调幅度，分级减少仓位：回调5%减至80%，10%减至60%，15%减至40%，20%+减至20%。价格恢复后回到100%。</p>
        </div>

        <div class="strategy-desc">
            <h3>策略B - 渐进加仓法</h3>
            <p>回调开始后立即减至50%，在确认触底后根据反弹幅度逐步加仓恢复到100%。</p>
        </div>

        <div class="strategy-desc">
            <h3>策略C - 动态对冲法（移动止损 + 波动率调整）</h3>
            <p>结合200日均线趋势判断和波动率调整仓位。价格在200MA下方时仓位减半；根据目标波动率(15%)动态调整持仓比例。</p>
        </div>

        <div class="strategy-desc">
            <h3>策略D - 多因子量化策略</h3>
            <p>综合四个因子决策：252日动量、14日RSI、均线趋势（50MA/200MA）、短期波动率。多因子评分映射为0.2-1.0的仓位范围。</p>
        </div>
    </div>

    <!-- ===== Performance Table ===== -->
    <div class="summary-box">
        <h2>策略表现对比表</h2>
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
                    <th>胜率</th>
                    <th>最终价值</th>
                </tr>
            </thead>
            <tbody>
                {perf_rows}
            </tbody>
        </table>
        </div>

        <h3 style="margin-top:20px;">交易统计</h3>
        <ul>
            {trade_summary}
        </ul>
    </div>

    <!-- ===== Charts ===== -->
    <div class="chart-container">{charts['cumulative']}</div>
    <div class="chart-container">{charts['drawdown']}</div>
    <div class="chart-container">{charts['position']}</div>
    <div class="chart-container">{charts['rolling']}</div>
    <div class="chart-container">{charts['scatter']}</div>
    <div class="chart-container">{charts['annual']}</div>
    <div class="chart-container">{charts['radar']}</div>
    <div class="chart-container">{charts['trades']}</div>
    <div class="chart-container">{charts['pnl']}</div>

    <div class="summary-box" style="text-align:center; color:#999; font-size:0.9em;">
        <p>本报告仅供学术研究和教育目的使用，不构成投资建议。</p>
        <p>回测区间：{start_date} 至 {end_date} | 初始资金：$100,000</p>
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
    print("Task 2: Dynamic Position Management Strategy Backtesting")
    print("=" * 60)

    # Load data
    print("\n[1/6] Loading data...")
    df_full = load_data('SP500.csv')
    df = get_backtest_period(df_full)
    print(f"  Backtest period: {df['Date'].iloc[0].strftime('%Y-%m-%d')} to {df['Date'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"  Trading days: {len(df)}")

    prices = df['Close'].values
    dates = df['Date'].values
    dates_ts = pd.to_datetime(dates)

    # Run strategies
    print("\n[2/6] Running strategies...")

    results = {}

    print("  Running Buy & Hold...")
    portfolio_bh, trades_bh, positions_bh = strategy_buy_and_hold(prices, dates_ts)
    results['Buy & Hold'] = {
        'portfolio': portfolio_bh,
        'trades': trades_bh,
        'positions': positions_bh,
        'metrics': compute_performance_metrics(portfolio_bh, dates_ts)
    }

    print("  Running Strategy A (Fixed Reduction)...")
    portfolio_a, trades_a, positions_a = strategy_a_fixed_reduction(prices, dates_ts)
    results['Strategy A'] = {
        'portfolio': portfolio_a,
        'trades': trades_a,
        'positions': positions_a,
        'metrics': compute_performance_metrics(portfolio_a, dates_ts)
    }

    print("  Running Strategy B (Gradual Rebalance)...")
    portfolio_b, trades_b, positions_b = strategy_b_gradual_rebalance(prices, dates_ts)
    results['Strategy B'] = {
        'portfolio': portfolio_b,
        'trades': trades_b,
        'positions': positions_b,
        'metrics': compute_performance_metrics(portfolio_b, dates_ts)
    }

    print("  Running Strategy C (Dynamic Hedging)...")
    portfolio_c, trades_c, positions_c = strategy_c_moving_stop(prices, dates_ts)
    results['Strategy C'] = {
        'portfolio': portfolio_c,
        'trades': trades_c,
        'positions': positions_c,
        'metrics': compute_performance_metrics(portfolio_c, dates_ts)
    }

    print("  Running Strategy D (Multi-Factor)...")
    portfolio_d, trades_d, positions_d = strategy_d_momentum_mean_reversion(prices, dates_ts)
    results['Strategy D'] = {
        'portfolio': portfolio_d,
        'trades': trades_d,
        'positions': positions_d,
        'metrics': compute_performance_metrics(portfolio_d, dates_ts)
    }

    # Print summary
    print("\n[3/6] Performance Summary:")
    print(f"  {'Strategy':<15} {'Total Return':>12} {'Annual Return':>14} {'Max DD':>10} {'Sharpe':>8} {'Calmar':>8}")
    print("  " + "-" * 70)
    for name, data in results.items():
        m = data['metrics']
        print(f"  {name:<15} {m['total_return']:>11.1f}% {m['annualized_return']:>13.1f}% {m['max_drawdown']:>9.1f}% {m['sharpe_ratio']:>7.2f} {m['calmar_ratio']:>7.2f}")

    # Save performance CSV
    print("\n[4/6] Saving performance data...")
    os.makedirs('outputs', exist_ok=True)
    perf_data = []
    for name, data in results.items():
        m = data['metrics']
        m['strategy'] = name
        perf_data.append(m)
    perf_df = pd.DataFrame(perf_data)
    perf_df.to_csv('outputs/task2_performance.csv', index=False, encoding='utf-8-sig')
    print("  Saved to outputs/task2_performance.csv")

    # Save trades CSV
    print("\n[5/6] Saving trade records...")
    all_trades = []
    for name, data in results.items():
        all_trades.extend(data['trades'])
    trades_df = pd.DataFrame(all_trades)
    if not trades_df.empty:
        trades_df.to_csv('outputs/task2_trades.csv', index=False, encoding='utf-8-sig')
    print(f"  Saved {len(all_trades)} trades to outputs/task2_trades.csv")

    # Generate HTML report
    print("\n[6/6] Generating HTML report...")
    html = generate_html_report(dates_ts, prices, results)
    with open('outputs/task2_backtest.html', 'w', encoding='utf-8') as f:
        f.write(html)
    print("  Saved to outputs/task2_backtest.html")

    print("\n" + "=" * 60)
    print("Task 2 Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
