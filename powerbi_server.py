"""
Power BI Desktop OData Feed Server for S&P 500 Data

Run this script to start a local OData server that Power BI Desktop can connect to.

Usage:
    python powerbi_server.py

Then in Power BI Desktop:
    Get Data → OData Feed → http://localhost:5000/odata
    or connect to individual endpoints via Web connector.
"""

import os
import json
import pandas as pd
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Data loaders ──────────────────────────────────────────────────────────────

def load_sp500():
    df = pd.read_csv(os.path.join(BASE_DIR, "SP500.csv"))
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    # Filter to 1995-2025 to match analysis scope
    df = df[df["Date"] >= "1995-01-01"].copy()
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    return df


def load_drawdowns():
    path = os.path.join(BASE_DIR, "outputs", "task1_drawdowns.csv")
    df = pd.read_csv(path)
    for col in ["peak_date", "trough_date", "recovery_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")
    # Round floats for readability
    float_cols = df.select_dtypes(include="float").columns
    df[float_cols] = df[float_cols].round(4)
    return df


def load_performance():
    path = os.path.join(BASE_DIR, "outputs", "task2_performance.csv")
    df = pd.read_csv(path).dropna(subset=["strategy"])
    float_cols = df.select_dtypes(include="float").columns
    df[float_cols] = df[float_cols].round(4)
    return df


def load_trades():
    path = os.path.join(BASE_DIR, "outputs", "task2_trades.csv")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    float_cols = df.select_dtypes(include="float").columns
    df[float_cols] = df[float_cols].round(4)
    return df


# ── OData helpers ──────────────────────────────────────────────────────────────

def df_to_odata(df, entity_name, base_url):
    """Convert a DataFrame to an OData JSON response (OData v4 format)."""
    records = df.where(pd.notna(df), None).to_dict(orient="records")
    return {
        "@odata.context": f"{base_url}/$metadata#{entity_name}",
        "value": records,
    }


def odata_response(data):
    return Response(
        json.dumps(data, ensure_ascii=False, default=str),
        mimetype="application/json;odata.metadata=minimal",
        headers={"OData-Version": "4.0"},
    )


# ── Service document & metadata ────────────────────────────────────────────────

@app.route("/odata", methods=["GET"])
def odata_root():
    base = request.url_root.rstrip("/") + "/odata"
    service_doc = {
        "@odata.context": f"{base}/$metadata",
        "value": [
            {"name": "SP500Prices", "kind": "EntitySet", "url": "SP500Prices"},
            {"name": "Drawdowns",   "kind": "EntitySet", "url": "Drawdowns"},
            {"name": "Performance", "kind": "EntitySet", "url": "Performance"},
            {"name": "Trades",      "kind": "EntitySet", "url": "Trades"},
        ],
    }
    return odata_response(service_doc)


@app.route("/odata/$metadata", methods=["GET"])
def odata_metadata():
    """Minimal EDMX metadata so Power BI can introspect entity types."""
    edmx = """<?xml version="1.0" encoding="utf-8"?>
<edmx:Edmx Version="4.0" xmlns:edmx="http://docs.oasis-open.org/odata/ns/edmx">
  <edmx:DataServices>
    <Schema Namespace="SP500" xmlns="http://docs.oasis-open.org/odata/ns/edm">

      <EntityType Name="SP500Price">
        <Key><PropertyRef Name="Date"/></Key>
        <Property Name="Date"   Type="Edm.String" Nullable="false"/>
        <Property Name="Close"  Type="Edm.Double"/>
        <Property Name="High"   Type="Edm.Double"/>
        <Property Name="Low"    Type="Edm.Double"/>
        <Property Name="Open"   Type="Edm.Double"/>
        <Property Name="Volume" Type="Edm.Double"/>
      </EntityType>

      <EntityType Name="Drawdown">
        <Key><PropertyRef Name="peak_date"/></Key>
        <Property Name="peak_date"           Type="Edm.String" Nullable="false"/>
        <Property Name="trough_date"          Type="Edm.String"/>
        <Property Name="recovery_date"        Type="Edm.String"/>
        <Property Name="peak_price"           Type="Edm.Double"/>
        <Property Name="trough_price"         Type="Edm.Double"/>
        <Property Name="recovery_price"       Type="Edm.Double"/>
        <Property Name="drawdown_pct"         Type="Edm.Double"/>
        <Property Name="duration_to_trough"   Type="Edm.Int32"/>
        <Property Name="duration_to_recovery" Type="Edm.Int32"/>
        <Property Name="recovery_from_trough" Type="Edm.Int32"/>
        <Property Name="recovered"            Type="Edm.Boolean"/>
        <Property Name="label"                Type="Edm.String"/>
        <Property Name="category"             Type="Edm.String"/>
        <Property Name="severity"             Type="Edm.String"/>
        <Property Name="decline_speed_annual" Type="Edm.Double"/>
        <Property Name="recovery_speed_annual"Type="Edm.Double"/>
        <Property Name="pain_index"           Type="Edm.Double"/>
      </EntityType>

      <EntityType Name="Performance">
        <Key><PropertyRef Name="strategy"/></Key>
        <Property Name="strategy"           Type="Edm.String" Nullable="false"/>
        <Property Name="total_return"       Type="Edm.Double"/>
        <Property Name="annualized_return"  Type="Edm.Double"/>
        <Property Name="annual_volatility"  Type="Edm.Double"/>
        <Property Name="max_drawdown"       Type="Edm.Double"/>
        <Property Name="sharpe_ratio"       Type="Edm.Double"/>
        <Property Name="calmar_ratio"       Type="Edm.Double"/>
        <Property Name="sortino_ratio"      Type="Edm.Double"/>
        <Property Name="win_rate"           Type="Edm.Double"/>
        <Property Name="final_value"        Type="Edm.Double"/>
      </EntityType>

      <EntityType Name="Trade">
        <Key><PropertyRef Name="date"/><PropertyRef Name="strategy"/><PropertyRef Name="action"/></Key>
        <Property Name="date"            Type="Edm.String" Nullable="false"/>
        <Property Name="action"          Type="Edm.String"/>
        <Property Name="old_position"    Type="Edm.Double"/>
        <Property Name="new_position"    Type="Edm.Double"/>
        <Property Name="price"           Type="Edm.Double"/>
        <Property Name="reason"          Type="Edm.String"/>
        <Property Name="portfolio_value" Type="Edm.Double"/>
        <Property Name="strategy"        Type="Edm.String" Nullable="false"/>
      </EntityType>

      <EntityContainer Name="SP500Container">
        <EntitySet Name="SP500Prices" EntityType="SP500.SP500Price"/>
        <EntitySet Name="Drawdowns"   EntityType="SP500.Drawdown"/>
        <EntitySet Name="Performance" EntityType="SP500.Performance"/>
        <EntitySet Name="Trades"      EntityType="SP500.Trade"/>
      </EntityContainer>

    </Schema>
  </edmx:DataServices>
</edmx:Edmx>"""
    return Response(edmx, mimetype="application/xml")


# ── Entity set endpoints ───────────────────────────────────────────────────────

@app.route("/odata/SP500Prices", methods=["GET"])
def sp500_prices():
    base = request.url_root.rstrip("/") + "/odata"
    df = load_sp500()

    # Support basic $filter on Date (e.g. ?$filter=Date ge '2020-01-01')
    filter_param = request.args.get("$filter", "")
    if "Date ge" in filter_param:
        try:
            date_str = filter_param.split("'")[1]
            df = df[df["Date"] >= date_str]
        except (IndexError, ValueError):
            pass

    # Support $top
    top = request.args.get("$top")
    if top:
        df = df.head(int(top))

    return odata_response(df_to_odata(df, "SP500Prices", base))


@app.route("/odata/Drawdowns", methods=["GET"])
def drawdowns():
    base = request.url_root.rstrip("/") + "/odata"
    df = load_drawdowns()
    return odata_response(df_to_odata(df, "Drawdowns", base))


@app.route("/odata/Performance", methods=["GET"])
def performance():
    base = request.url_root.rstrip("/") + "/odata"
    df = load_performance()
    return odata_response(df_to_odata(df, "Performance", base))


@app.route("/odata/Trades", methods=["GET"])
def trades():
    base = request.url_root.rstrip("/") + "/odata"
    df = load_trades()

    # Support $filter on strategy
    filter_param = request.args.get("$filter", "")
    if "strategy eq" in filter_param:
        try:
            strat = filter_param.split("'")[1]
            df = df[df["strategy"] == strat]
        except (IndexError, ValueError):
            pass

    return odata_response(df_to_odata(df, "Trades", base))


# ── Health check ───────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat()})


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║          S&P 500 Power BI OData Feed Server                  ║
╠══════════════════════════════════════════════════════════════╣
║  OData root  :  http://localhost:{port}/odata                   ║
║  Metadata    :  http://localhost:{port}/odata/$metadata         ║
║                                                              ║
║  Endpoints                                                   ║
║    SP500 Prices  :  /odata/SP500Prices                       ║
║    Drawdowns     :  /odata/Drawdowns                         ║
║    Performance   :  /odata/Performance                       ║
║    Trades        :  /odata/Trades                            ║
║                                                              ║
║  In Power BI Desktop:                                        ║
║    Get Data → OData Feed                                     ║
║    URL: http://localhost:{port}/odata                           ║
╚══════════════════════════════════════════════════════════════╝
""")
    app.run(host="0.0.0.0", port=port, debug=False)
