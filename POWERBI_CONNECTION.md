# Connecting Power BI Desktop to S&P 500 Data

This project exposes an **OData v4 feed** that Power BI Desktop connects to natively,
giving you live access to all four datasets without copying CSV files manually.

---

## 1. Start the server

```bash
# Install dependencies (one-time)
pip install -r requirements_powerbi.txt

# Start the OData server
python powerbi_server.py
```

The server starts at **http://localhost:5000**.

---

## 2. Connect Power BI Desktop

### Option A — OData Feed (recommended, gets all tables at once)

1. Open **Power BI Desktop**
2. **Home → Get Data → OData Feed**
3. Enter URL: `http://localhost:5000/odata`
4. Click **OK** → **Connect**
5. In the **Navigator** panel, tick all four tables:
   - `SP500Prices`
   - `Drawdowns`
   - `Performance`
   - `Trades`
6. Click **Load** (or **Transform Data** to preview first)

### Option B — Web connector (individual table)

1. **Home → Get Data → Web**
2. Enter one of the endpoint URLs:

| Table | URL |
|-------|-----|
| SP500 daily prices (1995–2025) | `http://localhost:5000/odata/SP500Prices` |
| Historical drawdown events | `http://localhost:5000/odata/Drawdowns` |
| Strategy performance metrics | `http://localhost:5000/odata/Performance` |
| Individual trade records | `http://localhost:5000/odata/Trades` |

3. In **Access Web content**, choose **Anonymous** → **Connect**
4. Power Query opens — select **List → value** to expand the table

---

## 3. Available datasets

### SP500Prices
Daily S&P 500 OHLCV data from 1995-01-01 to present.

| Column | Type | Description |
|--------|------|-------------|
| Date | Text | Trading date (YYYY-MM-DD) |
| Close | Decimal | Closing price |
| High | Decimal | Daily high |
| Low | Decimal | Daily low |
| Open | Decimal | Opening price |
| Volume | Decimal | Trading volume |

### Drawdowns
12 major drawdown events (>10% decline) identified from 1995–2025.

| Column | Type | Description |
|--------|------|-------------|
| peak_date | Text | Date market peaked |
| trough_date | Text | Date market bottomed |
| recovery_date | Text | Date market fully recovered |
| drawdown_pct | Decimal | Max drawdown percentage |
| duration_to_trough | Integer | Calendar days peak → trough |
| duration_to_recovery | Integer | Calendar days peak → recovery |
| severity | Text | Classification (轻度/中度/严重/熊市/股灾) |
| label | Text | Event name |
| pain_index | Decimal | Magnitude × duration composite |

### Performance
Backtest results comparing 5 portfolio strategies (2009–present).

| Column | Type | Description |
|--------|------|-------------|
| strategy | Text | Strategy name |
| total_return | Decimal | Cumulative return (%) |
| annualized_return | Decimal | CAGR (%) |
| sharpe_ratio | Decimal | Risk-adjusted return |
| max_drawdown | Decimal | Worst peak-to-trough decline |
| final_value | Decimal | Final portfolio value ($100k start) |

### Trades
All position changes per strategy during the backtest.

| Column | Type | Description |
|--------|------|-------------|
| date | Text | Trade date |
| strategy | Text | Strategy name |
| action | Text | REDUCE or INCREASE |
| old_position | Decimal | Position before trade |
| new_position | Decimal | Position after trade |
| price | Decimal | S&P 500 price at trade |
| portfolio_value | Decimal | Portfolio value at trade |

---

## 4. Suggested Power BI visuals

| Visual | Datasets | Fields |
|--------|----------|--------|
| Line chart — Price history | SP500Prices | Date (X), Close (Y) |
| Shaded area — Drawdown periods | SP500Prices + Drawdowns | Date, Close; peak/trough dates |
| Bar chart — Drawdown severity | Drawdowns | label (X), drawdown_pct (Y) |
| Table — Strategy comparison | Performance | All columns |
| Scatter — Risk vs Return | Performance | annual_volatility (X), annualized_return (Y), strategy (Legend) |
| Line chart — Trades overlay | Trades | date, portfolio_value, strategy |

---

## 5. Query parameters

The server supports basic OData query options:

| Parameter | Example | Effect |
|-----------|---------|--------|
| `$top` | `/odata/SP500Prices?$top=100` | Return first 100 rows |
| `$filter` (SP500Prices) | `/odata/SP500Prices?$filter=Date ge '2020-01-01'` | Filter by date |
| `$filter` (Trades) | `/odata/Trades?$filter=strategy eq 'Fixed 50%'` | Filter by strategy |

---

## 6. Troubleshooting

**Power BI can't reach the server**
- Ensure `python powerbi_server.py` is running (check the console)
- Check firewall: allow inbound TCP on port 5000
- Try `http://127.0.0.1:5000/health` in a browser — you should see `{"status":"ok"}`

**OData Feed shows no tables**
- Use the **Web** connector instead and point to `http://localhost:5000/odata/SP500Prices`
- In Power Query, navigate: **Record → value** to get the list

**Date columns show as Text**
- In Power Query: select the Date column → **Transform → Data Type → Date**

**Chinese characters appear garbled**
- Power BI handles UTF-8 natively; ensure your system locale supports it
