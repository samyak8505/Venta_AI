import sys
import asyncio
import base64
from io import BytesIO
from urllib.parse import quote_plus

# --- FastAPI & CORS ---
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# --- Plotting (headless) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# --- Data/DB ---
import numpy as np
import pandas as pd
from prophet import Prophet
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
load_dotenv()
# Windows asyncio policy (avoid Proactor loop issues with some libs)
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# -----------------------------
# App init (define ONCE)
# -----------------------------
app = FastAPI(title="Forecasting & RFM API")

# CORS (define ONCE; tighten origins later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://calm-charm-production-2d12.up.railway.app/",
        "https://ventaai-production.up.railway.app",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*",   # testing only; replace with your real origin in prod
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# 1. Database Connection
# -----------------------------
"""user = "alfawise_test_AI"
password = "WED@#$df$%as"
host = "103.180.163.212"
port = 20312
database = "alfawise_test_AI"

"""

user     = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
host     = os.getenv("DB_HOST")
port     = int(os.getenv("DB_PORT", "1433"))
database = os.getenv("DB_NAME")

encoded_password = quote_plus(password)
db_url = f"mssql+pyodbc://{user}:{encoded_password}@{host}:{port}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
engine = create_engine(db_url)
print("DB connected")

def convert_types(obj):
    if isinstance(obj, np.generic):   # e.g. np.int64, np.float32
        return obj.item()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: convert_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_types(v) for v in obj]
    return obj


def forecast_product(order_items_df, product_id, months_ahead):
    prod_sales = (
        order_items_df[order_items_df["ProductId"] == product_id]
        .dropna(subset=["PaidDateUtc"])
        .set_index("PaidDateUtc")["Quantity"]
        .resample("ME").sum()
        .reset_index()
        .rename(columns={"PaidDateUtc": "ds", "Quantity": "y"})
    )

    today              = pd.Timestamp.today().normalize()
    last_complete      = (today.replace(day=1) - pd.Timedelta(days=1))
    current_month_data = prod_sales[prod_sales["ds"] > last_complete]
    prod_sales         = prod_sales[prod_sales["ds"] <= last_complete].reset_index(drop=True)

    if len(prod_sales) < 6:
        return None, None

    # ── OUTLIER IMPUTATION (rolling median, same logic as sales) ─────────────
    monthly_clean = prod_sales.copy()
    rolling_med   = monthly_clean['y'].rolling(window=3, center=True, min_periods=1).median()
    low_mask      = monthly_clean['y'] < (rolling_med * 0.20)
    high_mask     = monthly_clean['y'] > (rolling_med * 3.0)
    outlier_mask  = low_mask | high_mask
    outliers      = monthly_clean[outlier_mask].copy()

    monthly_clean.loc[outlier_mask, 'y'] = np.nan
    monthly_clean['y'] = (
        monthly_clean['y']
        .interpolate(method='linear')
        .bfill().ffill()
    )
    monthly_clean['y'] = monthly_clean['y'].clip(lower=1)  # never below 1 unit

    # ── ROBUST CAP / FLOOR ───────────────────────────────────────────────────
    cap_value = monthly_clean['y'].quantile(0.90) * 1.5
    min_floor = max(1.0, monthly_clean['y'].quantile(0.10) * 0.5)

    # ── PROPHET FIT ──────────────────────────────────────────────────────────
    m = Prophet(
        growth='linear',
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='additive',
        changepoint_prior_scale=0.01,
        seasonality_prior_scale=2.0,
        interval_width=0.80,
        n_changepoints=min(10, len(monthly_clean) // 2)  # safe for short series
    )
    m.add_seasonality(name='yearly', period=365.25, fourier_order=3)
    m.fit(monthly_clean[['ds', 'y']])

    # ── FORECAST ─────────────────────────────────────────────────────────────
    future        = m.make_future_dataframe(periods=months_ahead, freq='ME')
    forecast      = m.predict(future)
    hist_end      = monthly_clean['ds'].max()
    forecast_mask = forecast['ds'] > hist_end

    for col in ['yhat', 'yhat_lower', 'yhat_upper']:
        forecast.loc[forecast_mask, col] = forecast.loc[forecast_mask, col].clip(lower=min_floor)
    forecast.loc[forecast_mask, 'yhat']       = forecast.loc[forecast_mask, 'yhat'].clip(upper=cap_value)
    forecast.loc[forecast_mask, 'yhat_upper'] = forecast.loc[forecast_mask, 'yhat_upper'].clip(upper=cap_value * 1.1)

    # ── PLOT ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))

    # Uncertainty band — forecast period only
    ax.fill_between(
        forecast.loc[forecast_mask, 'ds'],
        forecast.loc[forecast_mask, 'yhat_lower'],
        forecast.loc[forecast_mask, 'yhat_upper'],
        color='#1f77b4', alpha=0.12, label='Uncertainty Band'
    )

    # Historical line (clean)
    ax.plot(
        monthly_clean['ds'], monthly_clean['y'],
        color='#1f77b4', linewidth=2, marker='o', markersize=4,
        label='Historical', zorder=3
    )

    # Outlier markers
    if not outliers.empty:
        ax.scatter(
            outliers['ds'], outliers['y'],
            color='#aaaaaa', marker='x', s=60, linewidths=2,
            label='Excluded Outlier', zorder=4
        )

    # Current (partial) month
    if not current_month_data.empty:
        ax.scatter(
            current_month_data['ds'], current_month_data['y'],
            color='green', marker='D', s=50, zorder=5,
            label='Current Month (partial)'
        )

    # Forecast line — connected from last historical point
    last_ds = monthly_clean['ds'].iloc[-1]
    last_y  = monthly_clean['y'].iloc[-1]
    conn_ds = pd.concat([pd.Series([last_ds]), forecast.loc[forecast_mask, 'ds']], ignore_index=True)
    conn_y  = pd.concat([pd.Series([last_y]),  forecast.loc[forecast_mask, 'yhat']], ignore_index=True)
    ax.plot(conn_ds, conn_y, color='#ff7f0e', linestyle='--', linewidth=2, label='Forecast', zorder=3)

    # Floor line
    ax.axhline(y=min_floor, color='red', linestyle=':', alpha=0.4, linewidth=1,
               label=f'Min Floor ({min_floor:,.0f})')

    y_max = max(monthly_clean['y'].max(), forecast.loc[forecast_mask, 'yhat_upper'].max())
    ax.set_ylim(0, y_max * 1.20)
    ax.set_title(f"Product {product_id} — Demand Forecast", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Month", fontsize=11, labelpad=10)
    ax.set_ylabel("Demand (Units)", fontsize=11, labelpad=10)
    ax.legend(fontsize=9, frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.close(fig)

    return forecast, fig
    

def sales_forecast_limited(sales_df, forecast_months):
    monthly = (
        sales_df
        .assign(OrderDate=pd.to_datetime(sales_df['OrderDate'], errors='coerce'))
        .dropna(subset=['OrderDate'])
        .set_index('OrderDate')['OrderTotal']
        .resample('ME')
        .sum()
        .reset_index()
    )
    monthly.columns = ['ds', 'y']
    monthly = monthly.dropna(subset=['y'])
    monthly = monthly[monthly['y'] > 0].reset_index(drop=True)

    # Drop incomplete current month
    today = pd.Timestamp.today().normalize()
    if monthly['ds'].max().to_period('M') == today.to_period('M'):
        monthly = monthly[monthly['ds'] < monthly['ds'].max()]

    # ── OUTLIER DETECTION: only flag near-zero DB glitches ───────────────────
    # Strategy: only remove points below a hard lower threshold.
    # Upper spikes are kept — they may be real sales peaks.
    # Lower threshold = 20% of rolling 3-month median (catches DB zeros only)
    monthly_clean = monthly.copy()

    # Rolling median with min_periods=1 so first rows aren't NaN
    rolling_med = monthly_clean['y'].rolling(window=3, center=True, min_periods=1).median()

    # A point is a low outlier ONLY if it's < 20% of its local rolling median
    # Catch BOTH low (DB zeros) and high (DB spikes) outliers
    low_outlier_mask  = monthly_clean['y'] < (rolling_med * 0.20)
    high_outlier_mask = monthly_clean['y'] > (rolling_med * 3.0)
    outlier_mask      = low_outlier_mask | high_outlier_mask
    outliers          = monthly_clean[outlier_mask].copy()

    monthly_clean.loc[outlier_mask, 'y'] = np.nan
    monthly_clean['y'] = (
        monthly_clean['y']
        .interpolate(method='linear')
        .bfill()
        .ffill()
    )

    # ── PROPHET FIT ───────────────────────────────────────────────────────────
    cap_value = monthly_clean['y'].quantile(0.90) * 1.5
    min_floor = monthly_clean['y'].quantile(0.10) * 0.5

    m = Prophet(
        growth='linear',
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='additive',
        changepoint_prior_scale=0.01,
        seasonality_prior_scale=2.0,
        interval_width=0.80,
        n_changepoints=10
    )
    m.add_seasonality(name='yearly', period=365.25, fourier_order=3)
    m.fit(monthly_clean[['ds', 'y']])

    # ── FORECAST ─────────────────────────────────────────────────────────────
    future        = m.make_future_dataframe(periods=forecast_months, freq='ME')
    forecast      = m.predict(future)
    hist_end      = monthly_clean['ds'].max()
    forecast_mask = forecast['ds'] > hist_end

    forecast.loc[forecast_mask, 'yhat']       = forecast.loc[forecast_mask, 'yhat'].clip(lower=min_floor, upper=cap_value)
    forecast.loc[forecast_mask, 'yhat_lower'] = forecast.loc[forecast_mask, 'yhat_lower'].clip(lower=min_floor)
    forecast.loc[forecast_mask, 'yhat_upper'] = forecast.loc[forecast_mask, 'yhat_upper'].clip(upper=cap_value * 1.1)

    # ── PLOT ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.fill_between(
        forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
        color='#1f77b4', alpha=0.12, label='Uncertainty Band'
    )
    ax.plot(
        monthly_clean['ds'], monthly_clean['y'],
        label='Historical', color='#1f77b4',
        marker='o', linewidth=2, markersize=5, zorder=3
    )
    
    # Bridge point: last historical value connects to first forecast point
    last_hist_ds = monthly_clean['ds'].iloc[-1]
    last_hist_y  = monthly_clean['y'].iloc[-1]

    forecast_ds  = forecast.loc[forecast_mask, 'ds']
    forecast_y   = forecast.loc[forecast_mask, 'yhat']

    connected_ds = pd.concat([pd.Series([last_hist_ds]), forecast_ds], ignore_index=True)
    connected_y  = pd.concat([pd.Series([last_hist_y]),  forecast_y],  ignore_index=True)

    ax.plot(
        connected_ds, connected_y,
        label='Forecast', color='#ff7f0e',
        linestyle='--', linewidth=2.2, zorder=3
    )
    

    y_max = max(monthly_clean['y'].max(), forecast.loc[forecast_mask, 'yhat_upper'].max())
    ax.set_ylim(0, y_max * 1.20)
    ax.set_title("Sales Forecasting", fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Month", fontsize=12, labelpad=12)
    ax.set_ylabel("Total Sales (in Millions)", fontsize=12, labelpad=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.2f}M"))
    ax.legend(loc='upper left', fontsize=10, frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.close(fig)

    # ── TABLE OUTPUT ─────────────────────────────────────────────────────────
    # Use ORIGINAL values in the table (not imputed) — more honest to the user
    hist_df            = monthly[['ds', 'y']].rename(columns={'y': 'Total_Sales'})
    hist_df['Source']  = 'Historical'

    future_df           = forecast.loc[forecast_mask, ['ds', 'yhat']].rename(columns={'yhat': 'Total_Sales'})
    future_df['Source'] = 'Forecast'

    final_df                = pd.concat([hist_df, future_df], ignore_index=True)
    final_df['Month']       = final_df['ds'].dt.strftime('%Y-%m')
    final_df['Total_Sales'] = final_df['Total_Sales'].round(2)
    final_df                = final_df[['Month', 'Total_Sales', 'Source']]

    return final_df, fig
    

def forecast_inventory_demand2(order_items_df, batch_df, product_id, months_ahead):
    # ── CURRENT STOCK ─────────────────────────────────────────────────────────
    stock_summary = (
        batch_df.groupby("ProductId")["InhandQuantity"]
        .sum().reset_index()
        .rename(columns={"InhandQuantity": "CurrentStock"})
    )
    if product_id not in stock_summary["ProductId"].values:
        raise ValueError(f"Product {product_id} not found in batch_df.")

    current_stock = stock_summary.loc[
        stock_summary["ProductId"] == product_id, "CurrentStock"
    ].values[0]

    # ── SALES HISTORY ─────────────────────────────────────────────────────────
    today         = pd.Timestamp.today().normalize()
    last_complete = today.replace(day=1) - pd.Timedelta(days=1)

    prod_sales = (
        order_items_df[order_items_df["ProductId"] == product_id]
        .dropna(subset=["PaidDateUtc"])
        .set_index("PaidDateUtc")["Quantity"]
        .resample("ME").sum()
        .reset_index()
        .rename(columns={"PaidDateUtc": "ds", "Quantity": "y"})
    )
    train_sales = prod_sales[prod_sales["ds"] <= last_complete].reset_index(drop=True)

    if len(train_sales) < 6:
        return None, None

    # ── OUTLIER IMPUTATION ────────────────────────────────────────────────────
    monthly_clean = train_sales.copy()
    rolling_med   = monthly_clean['y'].rolling(window=3, center=True, min_periods=1).median()
    outlier_mask  = (monthly_clean['y'] < rolling_med * 0.20) | (monthly_clean['y'] > rolling_med * 3.0)
    outliers      = monthly_clean[outlier_mask].copy()

    monthly_clean.loc[outlier_mask, 'y'] = np.nan
    monthly_clean['y'] = monthly_clean['y'].interpolate(method='linear').bfill().ffill().clip(lower=1)

    # ── CAP / FLOOR ───────────────────────────────────────────────────────────
    cap_value = monthly_clean['y'].quantile(0.90) * 1.5
    min_floor = max(1.0, monthly_clean['y'].quantile(0.10) * 0.5)

    # ── PROPHET FIT ───────────────────────────────────────────────────────────
    m = Prophet(
        growth='linear',
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='additive',
        changepoint_prior_scale=0.01,
        seasonality_prior_scale=2.0,
        interval_width=0.80,
        n_changepoints=min(10, len(monthly_clean) // 2)
    )
    m.add_seasonality(name='yearly', period=365.25, fourier_order=3)
    m.fit(monthly_clean[['ds', 'y']])

    # ── FORECAST ─────────────────────────────────────────────────────────────
    future        = m.make_future_dataframe(periods=months_ahead, freq='ME')
    forecast      = m.predict(future)
    hist_end      = monthly_clean['ds'].max()
    forecast_mask = forecast['ds'] > hist_end

    for col in ['yhat', 'yhat_lower', 'yhat_upper']:
        forecast.loc[forecast_mask, col] = forecast.loc[forecast_mask, col].clip(lower=min_floor)
    forecast.loc[forecast_mask, 'yhat']       = forecast.loc[forecast_mask, 'yhat'].clip(upper=cap_value)
    forecast.loc[forecast_mask, 'yhat_upper'] = forecast.loc[forecast_mask, 'yhat_upper'].clip(upper=cap_value * 1.1)

    # ── STOCKOUT DETECTION (from today forward only) ──────────────────────────
    # Cumulate only the FUTURE forecast rows, not historical
    future_fc = forecast.loc[forecast_mask].copy().reset_index(drop=True)
    future_fc['cumulative_demand'] = future_fc['yhat'].cumsum()
    stockout_rows = future_fc[future_fc['cumulative_demand'] > current_stock]
    stockout_date = stockout_rows['ds'].iloc[0] if not stockout_rows.empty else None

    # ── PLOT ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))

    # Uncertainty band (forecast only)
    ax.fill_between(
        future_fc['ds'], future_fc['yhat_lower'], future_fc['yhat_upper'],
        color='#1f77b4', alpha=0.12, label='Uncertainty Band'
    )

    # Historical
    ax.plot(
        monthly_clean['ds'], monthly_clean['y'],
        color='#1f77b4', linewidth=2, marker='o', markersize=4,
        label='Historical Demand', zorder=3
    )
    if not outliers.empty:
        ax.scatter(outliers['ds'], outliers['y'],
                   color='#aaaaaa', marker='x', s=60, linewidths=2,
                   label='Excluded Outlier', zorder=4)

    # Forecast line — connected
    last_ds = monthly_clean['ds'].iloc[-1]
    last_y  = monthly_clean['y'].iloc[-1]
    conn_ds = pd.concat([pd.Series([last_ds]), future_fc['ds']], ignore_index=True)
    conn_y  = pd.concat([pd.Series([last_y]),  future_fc['yhat']], ignore_index=True)
    ax.plot(conn_ds, conn_y, color='#ff7f0e', linestyle='--', linewidth=2,
            label='Demand Forecast', zorder=3)

    # Current stock line
    ax.axhline(y=current_stock, color='red', linestyle='--', linewidth=1.5,
               label=f'Current Stock ({current_stock:,.0f} units)')

    # Stockout marker
    if stockout_date:
        ax.axvline(x=stockout_date, color='darkred', linestyle=':', linewidth=1.5,
                   label=f'Stockout ~{stockout_date.strftime("%b %Y")}')

    y_max = max(monthly_clean['y'].max(), future_fc['yhat_upper'].max(), current_stock)
    ax.set_ylim(0, y_max * 1.20)
    ax.set_title(f"Product {product_id} — Demand vs Stock Forecast", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Month", fontsize=11, labelpad=10)
    ax.set_ylabel("Demand (Units)", fontsize=11, labelpad=10)
    ax.legend(fontsize=9, frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.close(fig)

    summary = {
        "ProductId":                    product_id,
        "CurrentStock":                 int(current_stock),
        "ForecastedCumulativeDemand":   round(future_fc['cumulative_demand'].iloc[-1], 0),
        "StockoutDate":                 stockout_date.strftime("%Y-%m") if stockout_date else "No stockout in forecast window",
    }

    return fig, summary


def rfm_segmentation(sales_df, customers_df):
    """
    Perform RFM segmentation on customers and return plots as figures.
    
    Args:
        sales_df (pd.DataFrame): Must include CustomerId, OrderId, OrderDate, OrderTotala
        customers_df (pd.DataFrame): Customer details
    
    Returns:
        tuple: (rfm dataframe, list of matplotlib Figure objects)
    """
    # --- Step 1: Prep ---
    sales_df["OrderDate"] = pd.to_datetime(sales_df["OrderDate"])
    snapshot_date = sales_df["OrderDate"].max() + pd.Timedelta(days=1)  # reference point

    # --- Step 2: Aggregate RFM ---
    rfm = (
        sales_df.groupby("CustomerId")
        .agg(
            Recency=("OrderDate", lambda x: (snapshot_date - x.max()).days),
            Frequency=("OrderId", "nunique"),
            Monetary=("OrderTotal", "sum"),
        )
        .reset_index()
    )

    # --- Step 3: Score (quintiles) ---
    rfm["R_Score"] = pd.qcut(rfm["Recency"], 5, labels=[5,4,3,2,1]).astype(int)
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
    rfm["M_Score"] = pd.qcut(rfm["Monetary"], 5, labels=[1,2,3,4,5]).astype(int)

    # --- Step 4: Combine ---
    rfm["RFM_Segment"] = (
        rfm["R_Score"].astype(str) + 
        rfm["F_Score"].astype(str) + 
        rfm["M_Score"].astype(str)
    )
    rfm["RFM_Score"] = rfm[["R_Score","F_Score","M_Score"]].sum(axis=1)

    # --- Step 5: Segment Mapping ---
    def assign_segment(row):
        r, f, m = row['R_Score'], row['F_Score'], row['M_Score']

        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        elif r >= 4 and f >= 3:
            return "Loyal Customers"
        elif r >= 3 and f >= 2:
            return "Potential Loyalist"
        elif r == 5 and f <= 2:
            return "New Customers"
        elif r <= 2 and f >= 4:
            return "At Risk"
        elif r <= 2 and f <= 2 and m <= 2:
            return "Lost"
        else:
            return "Others"

    rfm["Segment"] = rfm.apply(assign_segment, axis=1)

    # --- Step 6: Merge with customer info ---
    rfm = rfm.merge(customers_df, on="CustomerId", how="left")
    rfm = rfm.sort_values("Monetary", ascending=False).reset_index(drop=True)

    figures = []

    # --- Figure 1: Customer Distribution ---
    fig1, ax1 = plt.subplots(figsize=(8,6))
    segment_counts = rfm['Segment'].value_counts()
    segment_counts.plot(kind='bar', color='skyblue', ax=ax1)
    ax1.set_title("Customer Distribution by Segment")
    ax1.set_xlabel("Segment")
    ax1.set_ylabel("Number of Customers")
    ax1.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.close(fig1)
    figures.append(fig1)

    # --- Figure 2: Revenue by Segment ---
    fig2, ax2 = plt.subplots(figsize=(8,6))
    revenue_by_segment = rfm.groupby("Segment")["Monetary"].sum().sort_values(ascending=False)
    revenue_by_segment.plot(kind="bar", color="orange", ax=ax2)
    ax2.set_title("Revenue Contribution by Segment")
    ax2.set_ylabel("Total Monetary Value")
    ax2.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.close(fig2)
    figures.append(fig2)

    # --- Figure 3: Recency vs Frequency ---
    fig3, ax3 = plt.subplots(figsize=(8,6))
    scatter = ax3.scatter(rfm["Recency"], rfm["Frequency"], 
                          c=rfm["RFM_Score"], cmap="viridis", alpha=0.7)
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label("RFM Score")
    ax3.set_title("Recency vs Frequency (Colored by RFM Score)")
    ax3.set_xlabel("Recency (days since last order)")
    ax3.set_ylabel("Frequency (# of orders)")
    plt.tight_layout()
    plt.close(fig3)
    figures.append(fig3)

    return rfm, figures

def calculate_clv(sales_df,bins=10):
    """
    Simple CLV estimation using average revenue per customer per period.
    
    Args:
        sales_df (pd.DataFrame): Must include CustomerId, OrderDate, OrderTotal
    
    Returns:
        pd.DataFrame: Customer CLV values
    """
    sales_df["OrderDate"] = pd.to_datetime(sales_df["OrderDate"])
    
    # Revenue per customer
    revenue = sales_df.groupby("CustomerId")["OrderTotal"].sum().reset_index()
    revenue.rename(columns={"OrderTotal": "TotalRevenue"}, inplace=True)

    # Avg revenue per order
    avg_order_value = sales_df.groupby("CustomerId")["OrderTotal"].mean().reset_index()
    avg_order_value.rename(columns={"OrderTotal": "AvgOrderValue"}, inplace=True)

    # Purchase frequency
    frequency = (
        sales_df.groupby("CustomerId")["OrderId"].nunique() /
        sales_df["CustomerId"].nunique()
    )
    purchase_freq = sales_df.groupby("CustomerId")["OrderId"].nunique().reset_index()
    purchase_freq.rename(columns={"OrderId": "TotalOrders"}, inplace=True)

    # Merge
    clv = revenue.merge(avg_order_value, on="CustomerId")
    clv = clv.merge(purchase_freq, on="CustomerId")

    # Approximate CLV (without churn/discount)
    clv["CLV"] = clv["AvgOrderValue"] * clv["TotalOrders"]
    clv = clv.sort_values("TotalRevenue", ascending=False).reset_index(drop=True)
    clv=pd.DataFrame(clv)

    fig, ax = plt.subplots(figsize=(10,6))
    ax.hist(clv["CLV"], bins=bins, color="skyblue", edgecolor="black")
    ax.set_title("Customer CLV Distribution")
    ax.set_xlabel("CLV Value")
    ax.set_ylabel("Number of Customers")
    plt.tight_layout()
    plt.close(fig)
    

    return clv,fig


def forecast_product_sales(order_items_df, product_id, months_ahead=3):
    """
    Forecast sales (revenue) for a single product using Prophet.
    Same robust outlier handling, cap/floor, and styling as sales_forecast_limited.
    """
    prod_sales = (
        order_items_df[order_items_df["ProductId"] == product_id]
        .dropna(subset=["PaidDateUtc"])
        .set_index("PaidDateUtc")["LineTotal"]
        .resample("ME").sum()
        .reset_index()
        .rename(columns={"PaidDateUtc": "ds", "LineTotal": "y"})
    )

    today             = pd.Timestamp.today().normalize()
    last_complete     = today.replace(day=1) - pd.Timedelta(days=1)
    current_month     = prod_sales[prod_sales["ds"] > last_complete].copy()
    prod_sales        = prod_sales[prod_sales["ds"] <= last_complete].reset_index(drop=True)
    prod_sales        = prod_sales[prod_sales["y"] > 0].reset_index(drop=True)

    if len(prod_sales) < 6:
        return None, None

    # ── OUTLIER IMPUTATION (rolling median) ──────────────────────────────────
    monthly_clean = prod_sales.copy()
    rolling_med   = monthly_clean['y'].rolling(window=3, center=True, min_periods=1).median()
    low_mask      = monthly_clean['y'] < (rolling_med * 0.20)
    high_mask     = monthly_clean['y'] > (rolling_med * 3.0)
    outlier_mask  = low_mask | high_mask
    outliers      = monthly_clean[outlier_mask].copy()

    monthly_clean.loc[outlier_mask, 'y'] = np.nan
    monthly_clean['y'] = (
        monthly_clean['y']
        .interpolate(method='linear')
        .bfill().ffill()
    )
    monthly_clean['y'] = monthly_clean['y'].clip(lower=1.0)

    # ── ROBUST CAP / FLOOR ───────────────────────────────────────────────────
    cap_value = monthly_clean['y'].quantile(0.90) * 1.5
    min_floor = max(1.0, monthly_clean['y'].quantile(0.10) * 0.5)

    # ── PROPHET FIT ──────────────────────────────────────────────────────────
    m = Prophet(
        growth='linear',
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='additive',
        changepoint_prior_scale=0.01,
        seasonality_prior_scale=2.0,
        interval_width=0.80,
        n_changepoints=min(10, len(monthly_clean) // 2)
    )
    m.add_seasonality(name='yearly', period=365.25, fourier_order=3)
    m.fit(monthly_clean[['ds', 'y']])

    # ── FORECAST ─────────────────────────────────────────────────────────────
    future        = m.make_future_dataframe(periods=months_ahead, freq='ME')
    forecast      = m.predict(future)
    hist_end      = monthly_clean['ds'].max()
    forecast_mask = forecast['ds'] > hist_end

    for col in ['yhat', 'yhat_lower', 'yhat_upper']:
        forecast.loc[forecast_mask, col] = forecast.loc[forecast_mask, col].clip(lower=min_floor)
    forecast.loc[forecast_mask, 'yhat']       = forecast.loc[forecast_mask, 'yhat'].clip(upper=cap_value)
    forecast.loc[forecast_mask, 'yhat_upper'] = forecast.loc[forecast_mask, 'yhat_upper'].clip(upper=cap_value * 1.1)

    # ── AUTO SCALE (K or M) ───────────────────────────────────────────────────
    max_val = max(monthly_clean['y'].max(), forecast.loc[forecast_mask, 'yhat'].max())
    if max_val >= 1e6:
        scale       = 1e6
        scale_label = "in Millions"
        fmt         = mticker.FuncFormatter(lambda x, _: f"${x/1e6:.2f}M")
    elif max_val >= 1e3:
        scale       = 1e3
        scale_label = "in Thousands"
        fmt         = mticker.FuncFormatter(lambda x, _: f"${x/1e3:,.0f}K")
    else:
        scale       = 1
        scale_label = ""
        fmt         = mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")

    # ── PLOT ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 7))

    # Uncertainty band — forecast only
    ax.fill_between(
        forecast.loc[forecast_mask, 'ds'],
        forecast.loc[forecast_mask, 'yhat_lower'],
        forecast.loc[forecast_mask, 'yhat_upper'],
        color='#1f77b4', alpha=0.12, label='Uncertainty Band'
    )

    # Historical line (clean)
    ax.plot(
        monthly_clean['ds'], monthly_clean['y'],
        color='#1f77b4', linewidth=2, marker='o', markersize=5,
        label='Historical', zorder=3
    )

    # Outlier markers
    if not outliers.empty:
        ax.scatter(
            outliers['ds'], outliers['y'],
            color='#aaaaaa', marker='x', s=70, linewidths=2,
            label='Excluded Outlier (DB glitch)', zorder=4
        )

    # Current (partial) month
    if not current_month.empty:
        ax.scatter(
            current_month['ds'], current_month['y'],
            color='green', marker='D', s=60, zorder=5,
            label='Current Month (partial)'
        )

    # Forecast line — connected from last historical point
    last_ds  = monthly_clean['ds'].iloc[-1]
    last_y   = monthly_clean['y'].iloc[-1]
    conn_ds  = pd.concat([pd.Series([last_ds]), forecast.loc[forecast_mask, 'ds']], ignore_index=True)
    conn_y   = pd.concat([pd.Series([last_y]),  forecast.loc[forecast_mask, 'yhat']], ignore_index=True)
    ax.plot(conn_ds, conn_y, color='#ff7f0e', linestyle='--', linewidth=2.2,
            label='Forecast', zorder=3)

    # Floor line
    ax.axhline(y=min_floor, color='red', linestyle=':', alpha=0.4, linewidth=1,
               label=f'Min Floor ({min_floor:,.0f})')

    y_max = max(monthly_clean['y'].max(), forecast.loc[forecast_mask, 'yhat_upper'].max())
    ax.set_ylim(0, y_max * 1.20)
    ax.set_title(f"Product {product_id} — Sales Forecast", fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Month", fontsize=12, labelpad=12)
    ax.set_ylabel(f"Sales Revenue ({scale_label})", fontsize=12, labelpad=12)
    ax.yaxis.set_major_formatter(fmt)
    ax.legend(loc='upper left', fontsize=10, frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.close(fig)

    # ── TABLE OUTPUT ─────────────────────────────────────────────────────────
    hist_df           = prod_sales[['ds', 'y']].rename(columns={'y': 'TotalSales'})
    hist_df['Source'] = 'Historical'

    future_df           = forecast.loc[forecast_mask, ['ds', 'yhat']].rename(columns={'yhat': 'TotalSales'})
    future_df['Source'] = 'Forecast'

    output_df               = pd.concat([hist_df, future_df], ignore_index=True)
    output_df['Month']      = output_df['ds'].dt.strftime('%Y-%m')
    output_df['TotalSales'] = output_df['TotalSales'].round(2)
    output_df               = output_df[['Month', 'TotalSales', 'Source']]

    return output_df, fig

    
def inventory_analysis(order_items_df, inventory_df, lead_time=2, service_level=0.95):
    """
    Performs inventory analysis per product, calculating safety stock and reorder points.
    - Accepts multiple stock column names (ClosingStock, InhandQuantity, etc.)
    - Uses month-end ('ME') grouping to avoid FutureWarning.
    """
    import math

    order_items_df = order_items_df.copy()
    order_items_df["PaidDateUtc"] = pd.to_datetime(order_items_df["PaidDateUtc"], errors="coerce")
    order_items_df = order_items_df.dropna(subset=["PaidDateUtc"])

    # ---- Monthly demand per product (month-end) ----
    monthly_demand = (
        order_items_df
        .groupby([pd.Grouper(key="PaidDateUtc",freq="ME"), "ProductId"])["Quantity"]
        .sum()
        .reset_index()
        .rename(columns={"PaidDateUtc": "Month", "Quantity": "TotalQuantity"})
    )

    # ---- Detect stock column in inventory_df ----
    stock_candidates = [
        "ClosingStock",
        "InhandQuantity",
        "InHandQuantity",
        "QuantityOnHand",
        "OnHand",
        "Stock",
        "In_Hand_Quantity",
    ]
    stock_col = next((c for c in stock_candidates if c in inventory_df.columns), None)
    if stock_col is None:
        # Make the error explicit and helpful
        raise KeyError(
            f"No stock column found. Expected one of {stock_candidates}, "
            f"but inventory_df columns are: {list(inventory_df.columns)}"
        )

    # Ensure ProductId exists in inventory_df
    if "ProductId" not in inventory_df.columns:
        # try a few common variants
        pid_variants = [c for c in inventory_df.columns if c.lower() in ("productid", "product_id", "productid")]
        if pid_variants:
            inventory_df = inventory_df.rename(columns={pid_variants[0]: "ProductId"})
        else:
            raise KeyError("Column 'ProductId' not found in inventory_df.")

    # Pre-aggregate current stock per product
    stock_summary = (
        inventory_df.groupby("ProductId")[stock_col]
        .sum()
        .reset_index()
        .rename(columns={stock_col: "CurrentStock"})
    )

    # Z-score for ~95% service level (you can map from service_level if you like)
    z = 1.65

    inventory_plan = []
    for prod_id, grp in monthly_demand.groupby("ProductId"):
        if len(grp) < 3:
            # skip products with very little history
            continue

        avg_demand = float(grp["TotalQuantity"].mean())
        demand_std = float(grp["TotalQuantity"].std(ddof=1)) if len(grp) > 1 else 0.0
        if math.isnan(demand_std):
            demand_std = 0.0

        safety_stock = z * demand_std * (lead_time ** 0.5)
        reorder_point = avg_demand * lead_time + safety_stock

        # current stock (0 if missing)
        cs = stock_summary.loc[stock_summary["ProductId"] == prod_id, "CurrentStock"]
        current_stock = float(cs.values[0]) if not cs.empty else 0.0

        recommendation = "OK"
        if current_stock < reorder_point:
            recommendation = "Reorder Needed"

        inventory_plan.append({
            "ProductId": int(prod_id),
            "AvgMonthlyDemand": round(avg_demand, 2),
            "DemandStdDev": round(demand_std, 2),
            "SafetyStock": round(safety_stock, 2),
            "ReorderPoint": round(reorder_point, 2),
            "CurrentStock": int(current_stock),
            "Recommendation": recommendation,
        })

    plan_df = pd.DataFrame(inventory_plan).sort_values("AvgMonthlyDemand", ascending=False).reset_index(drop=True)
    return plan_df


def load_view(view_name):
    return pd.read_sql(f"SELECT * FROM {view_name}", engine)

# -----------------------------
# 2. FastAPI Init
# -----------------------------
app = FastAPI(title="Forecasting & RFM API")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",   # if you open your index.html with Live Server
        "http://127.0.0.1:5500",  # alternate
        "http://localhost:3000",  # if using React/Vite
        "http://127.0.0.1:3000",
        "http://localhost:8000",  # backend itself (useful for swagger/docs)
        "http://127.0.0.1:8000",
        "*"                       # allow everything (for testing only)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper to convert matplotlib fig → base64
def fig_to_base64(fig):
    buf = BytesIO()
    # increase DPI and figure size for higher resolution
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# -----------------------------
# 3. Endpoints
# -----------------------------

@app.get("/sales_forecasting")
def sales_forecasting(months: int = Query(1, description="Forecast horizon in months")):
    sales_df = load_view("vw_Sales")
    data, fig = sales_forecast_limited(sales_df, forecast_months=months)
    return {
        "data": convert_types(data.to_dict(orient="records")),
        "plot": fig_to_base64(fig)
    }

@app.get("/product_forecast")
def product_forecast(product_id: int = Query(..., description="Product ID")):
    order_items_df = load_view("vw_OrderItem")
    forecast, fig = forecast_product(order_items_df, product_id, months_ahead=6)
    return {
        "forecast": convert_types(forecast.to_dict(orient="records")),
        "plot": fig_to_base64(fig)
    }

@app.get("/product_sales_forecast")
def product_sales_forecast(product_id: int = Query(..., description="Product ID"),months: int = Query(6, description="Forecast horizon in months")):
    # Load order items view
    order_items_df = load_view("vw_OrderItem")

    # Run sales forecast (using LineTotal)
    forecast, fig = forecast_product_sales(order_items_df, product_id, months_ahead=months)

    if forecast is None:
        return {"message": f"Not enough history to forecast sales for Product {product_id}"}

    return {
        "forecast": convert_types(forecast.to_dict(orient="records")),
        "plot": fig_to_base64(fig)
    }

@app.get("/inventory_demand")
def inventory_vs_demand(
    product_id: int = Query(..., description="Product ID"),
    months: int = Query(12, description="Forecast horizon in months")
):
    order_items_df = load_view("vw_OrderItem")
    batch_df = load_view("vw_BatchBalance")
    fig, summary = forecast_inventory_demand2(
        order_items_df, batch_df, product_id, months_ahead=months
    )
    if fig is None:
        return {
            "summary": convert_types(summary) if summary else None,
            "plot": None
        }
    return {
        "summary": convert_types(summary),
        "plot": fig_to_base64(fig)
    }


@app.get("/rfm")
def rfm_analysis():
    customers_df = load_view("vw_Customers")
    sales_df = load_view("vw_Sales")

    rfm_df, rfm_figs = rfm_segmentation(sales_df, customers_df)

    return {
        "rfm_table": convert_types(rfm_df.head(100).to_dict(orient="records")),
        "plots": [fig_to_base64(fig) for fig in rfm_figs]
    }

@app.get("/clv")
def clv_analysis():
    sales_df = load_view("vw_Sales")
    clv_df, clv_fig = calculate_clv(sales_df)

    return {
        "clv_table": convert_types(clv_df.head(100).to_dict(orient="records")),
        "plot": fig_to_base64(clv_fig)
    }

@app.get("/clv/customer/{customer_id}")
def clv_for_customer(customer_id: int):
    """
    Get CLV for a single customer
    """
    sales_df = load_view("vw_Sales")
    clv_df, clv_fig = calculate_clv(sales_df)

    # filter by customer_id
    customer_result = clv_df[clv_df["CustomerId"] == customer_id]

    if customer_result.empty:
        return {"error": f"No CLV data found for customer {customer_id}"}

    return {
        "clv_table": convert_types(customer_result.to_dict(orient="records"))
    }

from fastapi import HTTPException

@app.get("/inventory_analysis")
def run_inventory_analysis():
    try:
        order_items_df = load_view("vw_OrderItem")
        # This view should contain a stock column like InhandQuantity / ClosingStock
        batch_df = load_view("vw_BatchBalance")  # or your inventory view/table

        plan_df = inventory_analysis(order_items_df, batch_df, lead_time=2, service_level=0.95)
        return {"plan": convert_types(plan_df.to_dict(orient="records"))}
    except KeyError as e:
        # Return a clean JSON error so the frontend can show it
        raise HTTPException(status_code=500, detail=f"Inventory analysis failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inventory analysis failed: {e}")

@app.get("/dashboard_stats")
def get_dashboard_stats():
    try:
        with engine.connect() as conn:
            first_order = pd.read_sql(
                "SELECT dbo.fnGetLocalDate(CreatedOnUtc) AS FirstOrderDate FROM dbo.[Order] WITH (NOLOCK) WHERE id=1",
                conn
            ).iloc[0]["FirstOrderDate"]

            last_order = pd.read_sql(
                "SELECT TOP 1 dbo.fnGetLocalDate(CreatedOnUtc) AS LastOrderDate FROM [Order] WITH (NOLOCK) ORDER BY id DESC",
                conn
            ).iloc[0]["LastOrderDate"]

            countries = pd.read_sql(
                "SELECT COUNT(id) AS Country FROM dbo.Country WITH (NOLOCK) WHERE Published=1",
                conn
            ).iloc[0]["Country"]

            distributors = pd.read_sql(
                "SELECT COUNT(MPD_Id) AS NoOfDistributors FROM dbo.MemProfile_Dtls WITH (NOLOCK) INNER JOIN dbo.MemJoining_Dtls WITH (NOLOCK) ON MemJoining_Dtls.MJD_MemID = MemProfile_Dtls.MPD_MemId",
                conn
            ).iloc[0]["NoOfDistributors"]

            orders = pd.read_sql(
                "SELECT COUNT(Id) AS NoOfOrders FROM dbo.[Order] WITH (NOLOCK)",
                conn
            ).iloc[0]["NoOfOrders"]

            revenue = pd.read_sql(
                "SELECT SUM(OrderTotal) AS TotalSales FROM dbo.[Order] WITH (NOLOCK)",
                conn
            ).iloc[0]["TotalSales"]

            products = pd.read_sql(
                "SELECT COUNT(*) AS NoOfProducts FROM dbo.Product WITH (NOLOCK)",
                conn
            ).iloc[0]["NoOfProducts"]

        # Format dates
        def fmt_date(d):
            if pd.isnull(d): return "N/A"
            return pd.Timestamp(d).strftime("%d-%m-%Y")

        return {
            "period":       f"{fmt_date(first_order)} To {fmt_date(last_order)}",
            "countries":    int(countries),
            "distributors": int(distributors),
            "orders":       int(orders),
            "revenue":      f"${float(revenue):,.2f}",
            "products":     int(products)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dashboard stats failed: {e}")


#uvicorn backend:app --host 0.0.0.0 --port 8000
#streamlit run chatbot.py --server.port 8501 --server.address 0.0.0.0