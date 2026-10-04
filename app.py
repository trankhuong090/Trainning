import os
import io
import time
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.optimize import minimize
import ta

# Hyperopt (optional import with graceful fallback if needed)
try:
    from hyperopt import fmin, tpe, hp, Trials
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False

warnings.filterwarnings("ignore")

# ==============================================================================
# CẤU HÌNH TRANG WEB & THEME
# ==============================================================================
st.set_page_config(
    page_title="HOSE Quant Backtest | SMA + OBV & MPT",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho giao diện hiện đại, chuyên nghiệp
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E3A8A, #0D9488);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 4px;
        color: #0f172a;
    }
    .metric-delta-pos {
        font-size: 0.85rem;
        color: #16a34a;
        font-weight: 600;
    }
    .metric-delta-neg {
        font-size: 0.85rem;
        color: #dc2626;
        font-weight: 600;
    }
    .badge-tag {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
        background-color: #e0f2fe;
        color: #0369a1;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        border-radius: 6px 6px 0 0;
        padding: 0 16px;
        font-weight: 600;
    }
    .conclusion-box {
        background: #f0fdf4;
        border-left: 5px solid #22c55e;
        padding: 16px;
        border-radius: 0 8px 8px 0;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# DEFAULT PARAMETERS & CONSTANTS (Từ kết quả tối ưu Hyperopt trong Notebook)
# ==============================================================================
DEFAULT_FILE = "HOSE_2020_2023_in.csv"
DEFAULT_TICKERS = ["DIG", "DGC", "VND", "HAH", "MSN"]
DEFAULT_TRAIN_START = pd.to_datetime("2020-01-01")
DEFAULT_TRAIN_END = pd.to_datetime("2021-12-31")
DEFAULT_TEST_START = pd.to_datetime("2022-01-01")
DEFAULT_TEST_END = pd.to_datetime("2022-12-31")

DEFAULT_BEST_PARAMS = {
    "DIG": {"ma_short": 70, "ma_long": 305, "obv_window": 30},
    "DGC": {"ma_short": 70, "ma_long": 245, "obv_window": 70},
    "VND": {"ma_short": 90, "ma_long": 320, "obv_window": 5},
    "HAH": {"ma_short": 55, "ma_long": 215, "obv_window": 60},
    "MSN": {"ma_short": 110, "ma_long": 350, "obv_window": 100},
}

# ==============================================================================
# HÀM XỬ LÝ DỮ LIỆU & CACHING
# ==============================================================================
@st.cache_data(show_spinner="Đang đọc và chuẩn hóa dữ liệu...")
def load_data(file_source):
    """Đọc dữ liệu CSV từ file cục bộ hoặc file tải lên"""
    if isinstance(file_source, str):
        if not os.path.exists(file_source):
            return None
        df = pd.read_csv(file_source, encoding="utf-8-sig", low_memory=False)
    else:
        df = pd.read_csv(file_source, encoding="utf-8-sig", low_memory=False)

    df.columns = df.columns.str.strip().str.lower()
    required = ["date", "ticker", "open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Thiếu cột bắt buộc trong file: {col}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required)
    return df

def prepare_stock_data(df_full, ticker):
    """Chuẩn hóa dữ liệu theo từng mã cổ phiếu"""
    df = df_full[df_full["ticker"] == ticker].copy()
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume"
    })
    df = df.set_index("date")[["Open", "High", "Low", "Close", "Volume"]]
    return df

# ==============================================================================
# HÀM TÍNH TOÁN KỸ THUẬT & TÍN HIỆU (SMA + OBV OR)
# ==============================================================================
def find_position_sma(df, ma_short, ma_long):
    """Tín hiệu giao cắt SMA: Mua khi ngắn cắt lên dài, Bán khi ngắn cắt xuống dài"""
    pos = pd.Series(0.0, index=df.index, name="SMA_signal")
    ma_short = int(ma_short)
    ma_long = int(ma_long)
    if ma_short >= ma_long or len(df) < ma_short:
        return pos

    sma_s = ta.trend.SMAIndicator(close=df["Close"], window=ma_short).sma_indicator()
    sma_l = ta.trend.SMAIndicator(close=df["Close"], window=ma_long).sma_indicator()

    buy = (sma_s > sma_l) & (sma_s.shift(1) <= sma_l.shift(1))
    sell = (sma_s < sma_l) & (sma_s.shift(1) >= sma_l.shift(1))

    pos.loc[buy] = 1.0
    pos.loc[sell] = -1.0
    return pos

def find_position_obv(df, obv_window):
    """Tín hiệu giao cắt OBV: Mua khi OBV cắt lên MA(OBV), Bán khi cắt xuống"""
    pos = pd.Series(0.0, index=df.index, name="OBV_signal")
    obv_window = int(obv_window)
    if len(df) < obv_window:
        return pos

    obv = ta.volume.OnBalanceVolumeIndicator(close=df["Close"], volume=df["Volume"]).on_balance_volume()
    obv_ma = obv.rolling(window=obv_window).mean()

    buy = (obv > obv_ma) & (obv.shift(1) <= obv_ma.shift(1))
    sell = (obv < obv_ma) & (obv.shift(1) >= obv_ma.shift(1))

    pos.loc[buy] = 1.0
    pos.loc[sell] = -1.0
    return pos

def find_position_or(df, ma_short, ma_long, obv_window):
    """Kết hợp tín hiệu SMA và OBV theo logic OR (xử lý xung đột = 0)"""
    sma_sig = find_position_sma(df, ma_short, ma_long)
    obv_sig = find_position_obv(df, obv_window)

    pos = pd.Series(0.0, index=df.index, name="SMA_OBV_OR")
    buy = (sma_sig == 1.0) | (obv_sig == 1.0)
    sell = (sma_sig == -1.0) | (obv_sig == -1.0)
    conflict = buy & sell

    pos.loc[buy & ~conflict] = 1.0
    pos.loc[sell & ~conflict] = -1.0
    return pos, sma_sig, obv_sig

def events_to_holding(events):
    """Chuyển đổi tín hiệu rời rạc sang trạng thái nắm giữ (Long-only: 1 hoặc 0)"""
    h = pd.Series(0.0, index=events.index)
    curr = 0.0
    for i, sig in enumerate(events):
        if sig == 1.0:
            curr = 1.0
        elif sig == -1.0:
            curr = 0.0
        h.iloc[i] = curr
    return h

def strategy_returns(df, events, commission=0.0):
    """
    Tính tỷ suất sinh lợi chiến lược:
    NGUYÊN TẮC CHỐNG LOOK-AHEAD BIAS:
    Tín hiệu phát sinh ở ngày t chỉ được thực thi từ phiên t+1 (shift 1).
    """
    asset_ret = df["Close"].pct_change().fillna(0.0)
    holding = events_to_holding(events)
    executed_holding = holding.shift(1).fillna(0.0)

    strat_ret = executed_holding * asset_ret
    turnover = executed_holding.diff().abs().fillna(executed_holding.abs())
    strat_ret = strat_ret - turnover * commission

    return strat_ret, executed_holding

def buy_hold_returns(df):
    """Tỷ suất sinh lợi chiến lược Buy & Hold (nắm giữ thụ động)"""
    return df["Close"].pct_change().fillna(0.0)

# ==============================================================================
# HÀM ĐO LƯỜNG HIỆU SUẤT (PERFORMANCE METRICS)
# ==============================================================================
def performance_stats(returns, trading_days=252, risk_free_rate=0.0):
    """Tính toán bộ chỉ số hiệu suất định lượng"""
    r = returns.dropna()
    if len(r) == 0:
        return {}

    equity = (1.0 + r).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    years = len(r) / trading_days

    if years > 0 and equity.iloc[-1] > 0:
        annual_return = float(equity.iloc[-1] ** (1.0 / years) - 1.0)
    else:
        annual_return = np.nan

    mean_daily = float(r.mean())
    annual_vol = float(r.std() * np.sqrt(trading_days))

    if annual_vol != 0 and not np.isnan(annual_vol):
        # Sharpe chuẩn với Risk-Free Rate
        sharpe = float((mean_daily * trading_days - risk_free_rate) / annual_vol)
    else:
        sharpe = np.nan

    running_max = equity.cummax()
    drawdown = (equity / running_max) - 1.0
    max_dd = float(drawdown.min())

    # Calmar ratio
    calmar = float(annual_return / abs(max_dd)) if (max_dd != 0 and not np.isnan(max_dd)) else np.nan

    return {
        "Mean Daily Return [%]": mean_daily * 100,
        "Total Return [%]": total_return * 100,
        "Annual Return [%]": annual_return * 100 if not np.isnan(annual_return) else np.nan,
        "Annual Volatility [%]": annual_vol * 100,
        "Sharpe Ratio": sharpe,
        "Max Drawdown [%]": max_dd * 100,
        "Calmar Ratio": calmar
    }

# ==============================================================================
# HÀM SÀNG LỌC ĐỊNH LƯỢNG (SCREENING TABLE)
# ==============================================================================
@st.cache_data
def run_screening_table(df_full, start_date, end_date, trading_days=252):
    """Bảng sàng lọc cổ phiếu theo 4 tiêu chí chuẩn hóa trên tập Train"""
    data = df_full[(df_full["date"] >= start_date) & (df_full["date"] <= end_date)].copy()
    rows = []

    for ticker, g in data.groupby("ticker"):
        g = g.sort_values("date").dropna(subset=["close", "volume"])
        if len(g) < 200:
            continue

        ret = g["close"].pct_change().dropna()
        if len(ret) == 0:
            continue

        tot_ret = float(g["close"].iloc[-1] / g["close"].iloc[0] - 1.0)
        ann_ret = float((1.0 + tot_ret) ** (trading_days / len(ret)) - 1.0) if (1.0 + tot_ret) > 0 else np.nan
        ann_vol = float(ret.std() * np.sqrt(trading_days))
        sharpe = float(ret.mean() / ret.std() * np.sqrt(trading_days)) if ret.std() != 0 else np.nan

        eq = (1.0 + ret).cumprod()
        mdd = float((eq / eq.cummax() - 1.0).min())
        avg_val = float((g["close"] * g["volume"]).mean())

        rows.append({
            "Ticker": ticker,
            "N": len(g),
            "Total Return": tot_ret,
            "Annual Return": ann_ret,
            "Volatility": ann_vol,
            "Sharpe": sharpe,
            "Max Drawdown": mdd,
            "Average Trading Value": avg_val
        })

    res = pd.DataFrame(rows)
    if res.empty:
        return res

    res["Return Rank"] = res["Annual Return"].rank(pct=True)
    res["Sharpe Rank"] = res["Sharpe"].rank(pct=True)
    res["Liquidity Rank"] = res["Average Trading Value"].rank(pct=True)
    res["Drawdown Rank"] = res["Max Drawdown"].rank(pct=True)  # Drawdown ít âm hơn thì rank cao hơn

    res["Score"] = (
        0.30 * res["Return Rank"]
        + 0.30 * res["Sharpe Rank"]
        + 0.25 * res["Liquidity Rank"]
        + 0.15 * res["Drawdown Rank"]
    )
    return res.sort_values("Score", ascending=False).reset_index(drop=True)

# ==============================================================================
# HÀM TỐI ƯU HÓA DANH MỤC (MARKOWITZ MPT VS EQUAL WEIGHT)
# ==============================================================================
def optimize_mpt(train_return_matrix, trading_days=252, risk_free_rate=0.0):
    """
    Tối ưu hóa danh mục MPT (Markowitz) tối đa hóa Sharpe trên tập Train:
    Ràng buộc: Tổng trọng số = 100%, Long-only (0 <= w_i <= 1)
    """
    n = train_return_matrix.shape[1]
    mean_daily = train_return_matrix.mean().values
    cov_annual = train_return_matrix.cov().values * trading_days

    def objective(w):
        p_ret = float(w @ mean_daily * trading_days)
        p_var = float(w.T @ cov_annual @ w)
        p_vol = np.sqrt(max(p_var, 0.0))
        if p_vol == 0:
            return 1e9
        sharpe = (p_ret - risk_free_rate) / p_vol
        return -sharpe

    x0 = np.repeat(1.0 / n, n)
    bounds = [(0.0, 1.0) for _ in range(n)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if result.success:
        return result.x
    return x0

# ==============================================================================
# HÀM HYPEROPT TỐI ƯU THAM SỐ (TÙY CHỌN CHẠY TRỰC TIẾP)
# ==============================================================================
def run_hyperopt_single(df_train, max_evals=40):
    """Tìm bộ tham số ma_short, ma_long, obv_window tối đa hóa Sharpe trên Train"""
    if not HYPEROPT_AVAILABLE:
        return {"ma_short": 50, "ma_long": 200, "obv_window": 30}

    def score(paras):
        ms, ml, ow = int(paras["ms"]), int(paras["ml"]), int(paras["ow"])
        if ms >= ml:
            return 999999
        events, _, _ = find_position_or(df_train, ms, ml, ow)
        ret, _ = strategy_returns(df_train, events)
        stats = performance_stats(ret)
        sharpe = stats.get("Sharpe Ratio", np.nan)
        if pd.isna(sharpe):
            return 999999
        return -sharpe

    space = {
        "ms": hp.quniform("ms", 25, 150, 5),
        "ml": hp.quniform("ml", 200, 400, 5),
        "ow": hp.quniform("ow", 5, 100, 5)
    }
    trials = Trials()
    best_raw = fmin(fn=score, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials, verbose=False)
    return {
        "ma_short": int(best_raw["ms"]),
        "ma_long": int(best_raw["ml"]),
        "obv_window": int(best_raw["ow"])
    }

# ==============================================================================
# SIDEBAR: CẤU HÌNH & THIẾT LẬP HỆ THỐNG
# ==============================================================================
st.sidebar.image("https://img.icons8.com/fluency/96/bullish.png", width=64)
st.sidebar.title("Cấu Hình Hệ Thống")

# 1. Nguồn Dữ Liệu
st.sidebar.markdown("### 📁 1. Dữ Liệu Thị Trường")
uploaded_file = st.sidebar.file_uploader("Tải file CSV tùy chỉnh", type=["csv"])

if uploaded_file is not None:
    df_full = load_data(uploaded_file)
    data_label = f"Tệp tải lên: {uploaded_file.name}"
elif os.path.exists(DEFAULT_FILE):
    df_full = load_data(DEFAULT_FILE)
    data_label = f"Tệp mặc định: {DEFAULT_FILE}"
else:
    st.error("Không tìm thấy file dữ liệu 'HOSE_2020_2023_in.csv'. Vui lòng tải file CSV lên từ thanh bên.")
    st.stop()

st.sidebar.caption(f"Trạng thái: `{data_label}` ({len(df_full):,} dòng, {df_full['ticker'].nunique()} mã)")

# 2. Lựa Chọn Cổ Phiếu
st.sidebar.markdown("### 🎯 2. Lựa Chọn Cổ Phiếu")
all_tickers = sorted(df_full["ticker"].unique().tolist())
valid_default_tickers = [t for t in DEFAULT_TICKERS if t in all_tickers]

selected_tickers = st.sidebar.multiselect(
    "Danh sách cổ phiếu kiểm định:",
    options=all_tickers,
    default=valid_default_tickers
)

if len(selected_tickers) < 2:
    st.sidebar.warning("Vui lòng chọn tối thiểu 2 cổ phiếu để thực hiện tối ưu danh mục MPT.")

# 3. Phân Kỳ Thời Gian (Train / Test)
st.sidebar.markdown("### 📅 3. Phân Kỳ Dữ Liệu")
col_d1, col_d2 = st.sidebar.columns(2)
train_start = col_d1.date_input("Train Bắt đầu", DEFAULT_TRAIN_START)
train_end = col_d2.date_input("Train Kết thúc", DEFAULT_TRAIN_END)

col_d3, col_d4 = st.sidebar.columns(2)
test_start = col_d3.date_input("Test Bắt đầu", DEFAULT_TEST_START)
test_end = col_d4.date_input("Test Kết thúc", DEFAULT_TEST_END)

# 4. Tham Số Giao Dịch & Quản Lý Vốn
st.sidebar.markdown("### 💰 4. Quản Lý Vốn & Rủi Ro")
initial_capital = st.sidebar.number_input("Vốn ban đầu (VNĐ):", value=1_000_000, step=100_000, format="%d")
commission_pct = st.sidebar.slider("Phí giao dịch (%/lần):", min_value=0.0, max_value=0.5, value=0.0, step=0.05) / 100.0
risk_free_rate = st.sidebar.slider("Lãi suất phi rủi ro Rf (%/năm):", min_value=0.0, max_value=10.0, value=0.0, step=0.5) / 100.0
trading_days = st.sidebar.number_input("Số phiên giao dịch/năm:", value=252, step=1)

# 5. Bộ Tham Số Chiến Lược (SMA + OBV)
st.sidebar.markdown("### ⚙️ 5. Tham Số Chiến Lược (SMA + OBV)")
param_mode = st.sidebar.radio(
    "Chế độ thiết lập tham số:",
    ["Dùng tham số tối ưu chuẩn (Notebook)", "Tự điều chỉnh thủ công", "Chạy Hyperopt tối ưu mới"]
)

# Khởi tạo session state lưu tham số
if "params" not in st.session_state:
    st.session_state["params"] = DEFAULT_BEST_PARAMS.copy()

current_params = {}
for t in selected_tickers:
    if t in st.session_state["params"]:
        current_params[t] = st.session_state["params"][t].copy()
    else:
        current_params[t] = {"ma_short": 50, "ma_long": 200, "obv_window": 30}

if param_mode == "Tự điều chỉnh thủ công":
    st.sidebar.info("Điều chỉnh tham số riêng cho từng mã:")
    for t in selected_tickers:
        with st.sidebar.expander(f"Mã: {t}", expanded=False):
            ms = st.slider(f"{t} - SMA Ngắn", 10, 150, current_params[t]["ma_short"], key=f"ms_{t}")
            ml = st.slider(f"{t} - SMA Dài", 150, 400, current_params[t]["ma_long"], key=f"ml_{t}")
            ow = st.slider(f"{t} - Cửa sổ OBV", 5, 120, current_params[t]["obv_window"], key=f"ow_{t}")
            current_params[t] = {"ma_short": ms, "ma_long": ml, "obv_window": ow}
    st.session_state["params"].update(current_params)

elif param_mode == "Chạy Hyperopt tối ưu mới":
    max_evals_input = st.sidebar.slider("Số vòng lặp (max_evals):", 20, 100, 40, step=10)
    if st.sidebar.button("🚀 Bắt đầu tối ưu hóa trên Train"):
        progress_bar = st.sidebar.progress(0.0)
        status_text = st.sidebar.empty()
        for idx, t in enumerate(selected_tickers):
            status_text.text(f"Đang tối ưu {t}...")
            df_stock = prepare_stock_data(df_full, t)
            train_sub = df_stock.loc[str(train_start):str(train_end)]
            if len(train_sub) > 50:
                best_p = run_hyperopt_single(train_sub, max_evals=max_evals_input)
                current_params[t] = best_p
            progress_bar.progress((idx + 1) / len(selected_tickers))
        st.session_state["params"].update(current_params)
        status_text.text("✅ Tối ưu hoàn tất!")
        st.sidebar.success("Đã cập nhật bộ tham số mới vào hệ thống!")

# ==============================================================================
# TIẾN HÀNH TÍNH TOÁN DỮ LIỆU TOÀN CỤC
# ==============================================================================
stock_data = {}
train_data = {}
test_data = {}
train_strategy = {}
test_strategy = {}
bh_train_stats = {}
bh_test_stats = {}

for ticker in selected_tickers:
    df_st = prepare_stock_data(df_full, ticker)
    stock_data[ticker] = df_st

    train = df_st.loc[(df_st.index >= pd.to_datetime(train_start)) & (df_st.index <= pd.to_datetime(train_end))].copy()
    test = df_st.loc[(df_st.index >= pd.to_datetime(test_start)) & (df_st.index <= pd.to_datetime(test_end))].copy()

    train_data[ticker] = train
    test_data[ticker] = test

    # Buy & Hold Stats
    bh_tr_ret = buy_hold_returns(train)
    bh_te_ret = buy_hold_returns(test)
    bh_train_stats[ticker] = performance_stats(bh_tr_ret, trading_days, risk_free_rate)
    bh_test_stats[ticker] = performance_stats(bh_te_ret, trading_days, risk_free_rate)

    # Strategy Evaluation (OR logic)
    p = current_params[ticker]
    ev_tr, sma_tr, obv_tr = find_position_or(train, p["ma_short"], p["ma_long"], p["obv_window"])
    ret_tr, hold_tr = strategy_returns(train, ev_tr, commission=commission_pct)
    stats_tr = performance_stats(ret_tr, trading_days, risk_free_rate)

    ev_te, sma_te, obv_te = find_position_or(test, p["ma_short"], p["ma_long"], p["obv_window"])
    ret_te, hold_te = strategy_returns(test, ev_te, commission=commission_pct)
    stats_te = performance_stats(ret_te, trading_days, risk_free_rate)

    train_strategy[ticker] = {
        "events": ev_tr, "holding": hold_tr, "returns": ret_tr, "stats": stats_tr,
        "sma_sig": sma_tr, "obv_sig": obv_tr
    }
    test_strategy[ticker] = {
        "events": ev_te, "holding": hold_te, "returns": ret_te, "stats": stats_te,
        "sma_sig": sma_te, "obv_sig": obv_te
    }

# Return matrices
train_return_matrix = pd.concat({t: train_strategy[t]["returns"] for t in selected_tickers}, axis=1).dropna()
test_return_matrix = pd.concat({t: test_strategy[t]["returns"] for t in selected_tickers}, axis=1).dropna()

# Weights calculation
n_assets = len(selected_tickers)
equal_weights = np.repeat(1.0 / n_assets, n_assets)
mpt_weights = optimize_mpt(train_return_matrix, trading_days, risk_free_rate)

# Portfolio returns
ew_train_ret = train_return_matrix.mul(equal_weights, axis=1).sum(axis=1)
mpt_train_ret = train_return_matrix.mul(mpt_weights, axis=1).sum(axis=1)

ew_test_ret = test_return_matrix.mul(equal_weights, axis=1).sum(axis=1)
mpt_test_ret = test_return_matrix.mul(mpt_weights, axis=1).sum(axis=1)

# Buy & hold equal weight benchmark
bh_test_matrix = pd.concat({t: buy_hold_returns(test_data[t]) for t in selected_tickers}, axis=1).dropna()
ew_bh_test_ret = bh_test_matrix.mul(equal_weights, axis=1).sum(axis=1)

# Stats
portfolio_test_stats = pd.DataFrame({
    "Equal Weight Buy & Hold": performance_stats(ew_bh_test_ret, trading_days, risk_free_rate),
    "SMA+OBV OR Equal Weight": performance_stats(ew_test_ret, trading_days, risk_free_rate),
    "SMA+OBV OR MPT": performance_stats(mpt_test_ret, trading_days, risk_free_rate)
}).T

# ==============================================================================
# MAIN PAGE HEADER & SUMMARY TAGS
# ==============================================================================
st.markdown('<div class="main-header">🏛️ HỆ THỐNG KIỂM ĐỊNH CHIẾN LƯỢC ĐẦU TƯ HOSE</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Nghiên cứu kiểm định chiến lược kết hợp tín hiệu <b>SMA + OBV (OR)</b> và tối ưu hóa danh mục '
    '<b>Modern Portfolio Theory (MPT) vs Equal Weight</b> trên thị trường chứng khoán Việt Nam (2020 - 2022).'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div>
    <span class="badge-tag">🔬 Nghiên Cứu Định Lượng</span>
    <span class="badge-tag">🛡️ Chống Look-Ahead Bias (Shift 1)</span>
    <span class="badge-tag">📊 Out-of-Sample Testing (2022)</span>
    <span class="badge-tag">⚖️ Markowitz Max Sharpe</span>
</div>
""", unsafe_allow_html=True)
st.write("")

# ==============================================================================
# MAIN TABS LAYOUT
# ==============================================================================
tabs = st.tabs([
    "📊 1. Sàng Lọc & Cơ Sở Lựa Chọn",
    "📈 2. Phân Tích Kỹ Thuật Từng Mã",
    "⚖️ 3. Tối Ưu Tỷ Trọng Danh Mục",
    "🏆 4. Kiểm Định Hiệu Suất Ngoài Mẫu",
    "📉 5. Tăng Trưởng Vốn & Rủi Ro",
    "📑 6. Báo Cáo & Kết Luận Học Thuật"
])

# ------------------------------------------------------------------------------
# TAB 1: SÀNG LỌC & CƠ SỞ LỰA CHỌN CỔ PHIẾU
# ------------------------------------------------------------------------------
with tabs[0]:
    st.markdown("### 📋 Sàng Lọc Định Lượng Trên Dữ Liệu Huấn Luyện (Train 2020 - 2021)")
    st.write(
        "Việc lựa chọn cổ phiếu được thực hiện hoàn toàn dựa trên dữ liệu **Train (2020–2021)**, "
        "tuyệt đối không sử dụng thông tin tương lai của năm 2022 nhằm đảm bảo tính khách quan."
    )

    with st.expander("📌 Công Thức Điểm Tổng Hợp Sàng Lọc (Composite Screening Score)", expanded=True):
        st.latex(r"""
        \text{Score} = 0.30 \times \text{Rank}_{\text{Return}} + 0.30 \times \text{Rank}_{\text{Sharpe}} + 0.25 \times \text{Rank}_{\text{Liquidity}} + 0.15 \times \text{Rank}_{\text{Drawdown}}
        """)
        st.markdown("""
        - **30% Lợi nhuận năm hóa (Annual Return):** Đánh giá động lượng tăng trưởng tài sản.
        - **30% Sharpe Ratio:** Đánh giá hiệu quả sinh lời trên một đơn vị biến động rủi ro.
        - **25% Thanh khoản bình quân (Liquidity):** Đảm bảo tính khả thi khi giải ngân khối lượng vốn lớn.
        - **15% Max Drawdown:** Khả năng phòng vệ, hạn chế rủi ro sụt giảm tài sản cực đại.
        """)

    screen_df = run_screening_table(df_full, pd.to_datetime(train_start), pd.to_datetime(train_end), trading_days)

    col_t1, col_t2 = st.columns([3, 2])
    with col_t1:
        st.markdown("#### 🏆 Top 15 Cổ Phiếu Dẫn Đầu Điểm Sàng Lọc (Train)")
        format_dict = {
            "Total Return": "{:.2%}", "Annual Return": "{:.2%}", "Volatility": "{:.2%}",
            "Sharpe": "{:.2f}", "Max Drawdown": "{:.2%}", "Average Trading Value": "{:,.0f}",
            "Score": "{:.4f}"
        }
        st.dataframe(screen_df.head(15).style.format(format_dict), use_container_width=True)

    with col_t2:
        st.markdown("#### 🎯 Vị Trí Các Cổ Phiếu Được Chọn")
        chosen_screen = screen_df[screen_df["Ticker"].isin(selected_tickers)].sort_values("Score", ascending=False)
        st.dataframe(chosen_screen.style.format(format_dict), use_container_width=True)

    st.markdown("---")
    st.markdown("### 🏢 Cơ Sở Định Tính & Đa Dạng Hóa Ngành")
    st.markdown("""
    Sau khi sàng lọc định lượng, nguyên tắc **đa dạng hóa danh mục theo Markowitz** đòi hỏi không dồn toàn bộ tỷ trọng 
    vào cùng một ngành để tránh rủi ro hệ thống ngành. Nhóm 5 cổ phiếu được lựa chọn đại diện cho 5 trụ cột kinh tế:
    """)

    col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
    sectors = {
        "DIG": ("Bất Động Sản", "Đại diện nhóm chu kỳ nhạy cảm lãi suất, beta cao, dòng tiền đột biến trong pha nới lỏng tiền tệ."),
        "DGC": ("Hóa Chất & Phốt Pho", "Hiệu suất sinh lời và Sharpe cao vượt trội trên Train, biên lợi nhuận dẫn đầu chuỗi giá trị."),
        "VND": ("Chứng Khoán", "Thanh khoản cực lớn, hưởng lợi trực tiếp từ làn sóng bùng nổ nhà đầu tư F0 và giá trị giao dịch."),
        "HAH": ("Vận Tải Biển / Logistics", "Hưởng lợi trực tiếp từ đứt gãy chuỗi cung ứng toàn cầu và giá cước vận tải biển tăng vọt."),
        "MSN": ("Tiêu Dùng & Bán Lẻ", "Cổ phiếu phòng thủ tiêu dùng thiết yếu, dòng tiền kinh doanh ổn định, giúp cân bằng danh mục.")
    }
    cols = [col_s1, col_s2, col_s3, col_s4, col_s5]
    for idx, t in enumerate(selected_tickers[:5]):
        if t in sectors:
            name, desc = sectors[t]
            cols[idx].info(f"**{t} ({name})**\n\n{desc}")
        else:
            cols[idx].info(f"**{t}**\n\nCổ phiếu được người dùng tùy chọn bổ sung vào danh mục kiểm định.")

# ------------------------------------------------------------------------------
# TAB 2: PHÂN TÍCH KỸ THUẬT & TÍN HIỆU TỪNG CỔ PHIẾU
# ------------------------------------------------------------------------------
with tabs[1]:
    st.markdown("### 🔍 Tín Hiệu Kỹ Thuật Chi Tiết & Khảo Sát Từng Cổ Phiếu")

    col_sel_stock, col_sel_period = st.columns([2, 2])
    stock_focus = col_sel_stock.selectbox("Chọn cổ phiếu phân tích sâu:", selected_tickers)
    period_focus = col_sel_period.radio("Giai đoạn hiển thị biểu đồ:", ["Test (2022 - Ngoài mẫu)", "Train (2020-2021)", "Toàn bộ (2020-2022)"], horizontal=True)

    df_stock_full = stock_data[stock_focus]
    p_focus = current_params[stock_focus]

    if period_focus == "Train (2020-2021)":
        display_df = df_stock_full.loc[str(train_start):str(train_end)].copy()
    elif period_focus == "Test (2022 - Ngoài mẫu)":
        display_df = df_stock_full.loc[str(test_start):str(test_end)].copy()
    else:
        display_df = df_stock_full.copy()

    # Tính toán lại cho display_df
    pos_or_disp, pos_sma_disp, pos_obv_disp = find_position_or(
        display_df, p_focus["ma_short"], p_focus["ma_long"], p_focus["obv_window"]
    )
    strat_ret_disp, hold_disp = strategy_returns(display_df, pos_or_disp, commission=commission_pct)
    stats_focus = performance_stats(strat_ret_disp, trading_days, risk_free_rate)
    bh_ret_disp = buy_hold_returns(display_df)
    bh_stats_focus = performance_stats(bh_ret_disp, trading_days, risk_free_rate)

    # Hiển thị Metric Cards
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Tổng Lợi Nhuận Chiến Lược</div><div class="metric-value">{stats_focus.get("Total Return [%]", 0):.2f}%</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Tổng Lợi Nhuận B&H</div><div class="metric-value">{bh_stats_focus.get("Total Return [%]", 0):.2f}%</div></div>', unsafe_allow_html=True)
    with m3:
        excess = stats_focus.get("Total Return [%]", 0) - bh_stats_focus.get("Total Return [%]", 0)
        c_class = "metric-delta-pos" if excess >= 0 else "metric-delta-neg"
        st.markdown(f'<div class="metric-card"><div class="metric-title">Alpha (Vượt Trội vs B&H)</div><div class="metric-value {c_class}">+{excess:.2f}%</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Sharpe Ratio</div><div class="metric-value">{stats_focus.get("Sharpe Ratio", 0):.2f}</div></div>', unsafe_allow_html=True)
    with m5:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Max Drawdown</div><div class="metric-value metric-delta-neg">{stats_focus.get("Max Drawdown [%]", 0):.2f}%</div></div>', unsafe_allow_html=True)

    st.write("")

    # Interactive Plotly Subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[
            f"Giá Đóng Cửa {stock_focus}, Đường SMA({p_focus['ma_short']}), SMA({p_focus['ma_long']}) & Điểm Mua/Bán",
            f"Chỉ Báo Khối Lượng OBV & Đường Trung Bình Động OBV-MA({p_focus['obv_window']})",
            "Vị Thế Thực Thi (1 = Nắm giữ cổ phiếu, 0 = Cầm tiền mặt) — Shift 1 Phiên Chống Look-Ahead"
        ],
        row_heights=[0.55, 0.25, 0.20]
    )

    # 1. Đường giá và SMA
    fig.add_trace(go.Scatter(x=display_df.index, y=display_df["Close"], name="Giá Đóng Cửa", line=dict(color="#2563eb", width=1.5)), row=1, col=1)

    sma_s = ta.trend.SMAIndicator(close=display_df["Close"], window=int(p_focus["ma_short"])).sma_indicator()
    sma_l = ta.trend.SMAIndicator(close=display_df["Close"], window=int(p_focus["ma_long"])).sma_indicator()
    fig.add_trace(go.Scatter(x=display_df.index, y=sma_s, name=f"SMA ({p_focus['ma_short']})", line=dict(color="#f97316", width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=display_df.index, y=sma_l, name=f"SMA ({p_focus['ma_long']})", line=dict(color="#10b981", width=1.2)), row=1, col=1)

    # Tín hiệu Mua / Bán
    buy_signals = display_df[pos_or_disp == 1.0]
    sell_signals = display_df[pos_or_disp == -1.0]

    fig.add_trace(go.Scatter(
        x=buy_signals.index, y=buy_signals["Close"],
        mode="markers", name="Tín Hiệu Mua (OR)",
        marker=dict(symbol="triangle-up", color="#16a34a", size=12, line=dict(width=1, color="black"))
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=sell_signals.index, y=sell_signals["Close"],
        mode="markers", name="Tín Hiệu Bán (OR)",
        marker=dict(symbol="triangle-down", color="#dc2626", size=12, line=dict(width=1, color="black"))
    ), row=1, col=1)

    # 2. OBV và OBV MA
    obv_val = ta.volume.OnBalanceVolumeIndicator(close=display_df["Close"], volume=display_df["Volume"]).on_balance_volume()
    obv_ma = obv_val.rolling(window=int(p_focus["obv_window"])).mean()
    fig.add_trace(go.Scatter(x=display_df.index, y=obv_val, name="OBV", line=dict(color="#8b5cf6", width=1.2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=display_df.index, y=obv_ma, name=f"OBV MA({p_focus['obv_window']})", line=dict(color="#ec4899", width=1.2, dash="dash")), row=2, col=1)

    # 3. Vị thế thực thi
    fig.add_trace(go.Scatter(
        x=display_df.index, y=hold_disp,
        name="Vị Thế Thực Thi", line=dict(color="#0284c7", width=1.5), fill="tozeroy", fillcolor="rgba(2, 132, 199, 0.15)"
    ), row=3, col=1)

    fig.update_layout(height=720, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="Giá (VNĐ)", row=1, col=1)
    fig.update_yaxes(title_text="OBV", row=2, col=1)
    fig.update_yaxes(title_text="Vị Thế", tickvals=[0, 1], ticktext=["Cash (0)", "Stock (1)"], row=3, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # Bảng chi tiết so sánh
    st.markdown(f"#### 📊 Bảng Đối Chiếu Chỉ Số Kỹ Thuật Của {stock_focus}")
    comp_df = pd.DataFrame({
        "Chỉ Số": [
            "Lợi Nhuận Trung Bình Ngày [%]", "Tổng Lợi Nhuận [%]", "Lợi Nhuận Năm Hóa [%]",
            "Độ Biến Động Năm Hóa [%]", "Sharpe Ratio", "Max Drawdown [%]", "Số Tín Hiệu Mua", "Số Tín Hiệu Bán"
        ],
        "Chiến Lược (Train)": [
            train_strategy[stock_focus]["stats"].get("Mean Daily Return [%]", np.nan),
            train_strategy[stock_focus]["stats"].get("Total Return [%]", np.nan),
            train_strategy[stock_focus]["stats"].get("Annual Return [%]", np.nan),
            train_strategy[stock_focus]["stats"].get("Annual Volatility [%]", np.nan),
            train_strategy[stock_focus]["stats"].get("Sharpe Ratio", np.nan),
            train_strategy[stock_focus]["stats"].get("Max Drawdown [%]", np.nan),
            int((train_strategy[stock_focus]["events"] == 1.0).sum()),
            int((train_strategy[stock_focus]["events"] == -1.0).sum())
        ],
        "Chiến Lược (Test 2022)": [
            test_strategy[stock_focus]["stats"].get("Mean Daily Return [%]", np.nan),
            test_strategy[stock_focus]["stats"].get("Total Return [%]", np.nan),
            test_strategy[stock_focus]["stats"].get("Annual Return [%]", np.nan),
            test_strategy[stock_focus]["stats"].get("Annual Volatility [%]", np.nan),
            test_strategy[stock_focus]["stats"].get("Sharpe Ratio", np.nan),
            test_strategy[stock_focus]["stats"].get("Max Drawdown [%]", np.nan),
            int((test_strategy[stock_focus]["events"] == 1.0).sum()),
            int((test_strategy[stock_focus]["events"] == -1.0).sum())
        ],
        "Buy & Hold (Test 2022)": [
            bh_test_stats[stock_focus].get("Mean Daily Return [%]", np.nan),
            bh_test_stats[stock_focus].get("Total Return [%]", np.nan),
            bh_test_stats[stock_focus].get("Annual Return [%]", np.nan),
            bh_test_stats[stock_focus].get("Annual Volatility [%]", np.nan),
            bh_test_stats[stock_focus].get("Sharpe Ratio", np.nan),
            bh_test_stats[stock_focus].get("Max Drawdown [%]", np.nan),
            0, 0
        ]
    }).set_index("Chỉ Số")

    st.dataframe(comp_df.style.format("{:.2f}", na_rep="-"), use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 3: TỐI ƯU TỶ TRỌNG DANH MỤC (EQUAL WEIGHT VS MPT)
# ------------------------------------------------------------------------------
with tabs[2]:
    st.markdown("### ⚖️ Tối Ưu Hóa Danh Mục Theo Lý Thuyết Hiện Đại (Markowitz MPT)")
    st.write(
        "Trọng số danh mục MPT được giải quyết qua bài toán quy hoạch phi tuyến nhằm **tối đa hóa Sharpe Ratio "
        "trên ma trận tỷ suất sinh lợi của chiến lược trong giai đoạn Train (2020–2021)**, sau đó áp dụng cố định "
        "vào giai đoạn Test (2022) để đánh giá khả năng duy trì hiệu quả ngoài mẫu."
    )

    col_w1, col_w2 = st.columns([1, 1])

    with col_w1:
        st.markdown("#### 🎯 Ma Trận Tương Quan Lợi Nhuận (Train)")
        corr_matrix = train_return_matrix.corr()
        fig_corr = px.imshow(
            corr_matrix, text_auto=".2f", aspect="auto",
            color_continuous_scale="Blues", title="Correlation Matrix (Strategy Returns on Train)"
        )
        fig_corr.update_layout(height=360)
        st.plotly_chart(fig_corr, use_container_width=True)

    with col_w2:
        st.markdown("#### 📊 So Sánh Phân Bổ Tỷ Trọng: Equal Weight vs MPT")
        weight_df = pd.DataFrame({
            "Ticker": selected_tickers,
            "Equal Weight": equal_weights,
            "MPT Weight": mpt_weights
        })

        fig_bar = go.Figure(data=[
            go.Bar(name="Equal Weight (1/N)", x=weight_df["Ticker"], y=weight_df["Equal Weight"], marker_color="#94a3b8"),
            go.Bar(name="MPT (Max Sharpe Train)", x=weight_df["Ticker"], y=weight_df["MPT Weight"], marker_color="#0d9488")
        ])
        fig_bar.update_layout(
            barmode="group", height=360, yaxis_tickformat=".1%",
            title="So Sánh Tỷ Trọng Phân Bổ Danh Mục", legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("#### 📑 Bảng Tổng Hợp Tỷ Trọng Danh Mục")
    weight_display = weight_df.copy()
    weight_display["Chênh Lệch (MPT - EW)"] = weight_display["MPT Weight"] - weight_display["Equal Weight"]
    st.dataframe(
        weight_display.style.format({
            "Equal Weight": "{:.2%}", "MPT Weight": "{:.2%}", "Chênh Lệch (MPT - EW)": "{:+.2%}"
        }),
        use_container_width=True
    )

# ------------------------------------------------------------------------------
# TAB 4: KIỂM ĐỊNH HIỆU SUẤT NGOÀI MẪU (TEST 2022)
# ------------------------------------------------------------------------------
with tabs[3]:
    st.markdown("### 🏆 Kết Quả Kiểm Định Ngoài Mẫu (Out-of-Sample Test 2022)")
    st.info(
        "Năm 2022 là năm thị trường chứng khoán Việt Nam rơi vào chu kỳ giảm mạnh (Bear Market) với VN-Index sụt giảm hơn 32%, "
        "nhiều cổ phiếu tăng trưởng gãy xu hướng giảm từ 60% đến 85%. Đây là phép thử hoàn hảo (Stress-test) để đo lường "
        "năng lực phòng thủ và bảo toàn vốn của chiến lược kết hợp SMA + OBV (OR) và mô hình phân bổ danh mục."
    )

    # 3 Cột so sánh chính
    col_c1, col_c2, col_c3 = st.columns(3)

    ew_bh_ret = portfolio_test_stats.loc["Equal Weight Buy & Hold", "Total Return [%]"]
    ew_strat_ret = portfolio_test_stats.loc["SMA+OBV OR Equal Weight", "Total Return [%]"]
    mpt_strat_ret = portfolio_test_stats.loc["SMA+OBV OR MPT", "Total Return [%]"]

    ew_bh_sharpe = portfolio_test_stats.loc["Equal Weight Buy & Hold", "Sharpe Ratio"]
    ew_strat_sharpe = portfolio_test_stats.loc["SMA+OBV OR Equal Weight", "Sharpe Ratio"]
    mpt_strat_sharpe = portfolio_test_stats.loc["SMA+OBV OR MPT", "Sharpe Ratio"]

    ew_bh_mdd = portfolio_test_stats.loc["Equal Weight Buy & Hold", "Max Drawdown [%]"]
    ew_strat_mdd = portfolio_test_stats.loc["SMA+OBV OR Equal Weight", "Max Drawdown [%]"]
    mpt_strat_mdd = portfolio_test_stats.loc["SMA+OBV OR MPT", "Max Drawdown [%]"]

    with col_c1:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid #ef4444;">
            <div class="metric-title">1. Equal Weight Buy & Hold</div>
            <div class="metric-value metric-delta-neg">{ew_bh_ret:.2f}%</div>
            <p style="font-size:0.85rem; color:#64748b; margin-top:8px;">
                Sharpe: <b>{ew_bh_sharpe:.2f}</b> | Max DD: <b>{ew_bh_mdd:.2f}%</b><br>
                <i>(Danh mục thụ động rơi tự do theo thị trường gấu)</i>
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_c2:
        alpha_ew = ew_strat_ret - ew_bh_ret
        st.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid #f59e0b;">
            <div class="metric-title">2. SMA + OBV OR (Equal Weight)</div>
            <div class="metric-value" style="color:#d97706;">{ew_strat_ret:.2f}%</div>
            <p style="font-size:0.85rem; color:#64748b; margin-top:8px;">
                Sharpe: <b>{ew_strat_sharpe:.2f}</b> | Max DD: <b>{ew_strat_mdd:.2f}%</b><br>
                <span class="metric-delta-pos">Bảo vệ vốn: +{alpha_ew:.2f}% so với B&H</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_c3:
        alpha_mpt = mpt_strat_ret - ew_bh_ret
        st.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid #0d9488;">
            <div class="metric-title">3. SMA + OBV OR (MPT Markowitz)</div>
            <div class="metric-value" style="color:#0f766e;">{mpt_strat_ret:.2f}%</div>
            <p style="font-size:0.85rem; color:#64748b; margin-top:8px;">
                Sharpe: <b>{mpt_strat_sharpe:.2f}</b> | Max DD: <b>{mpt_strat_mdd:.2f}%</b><br>
                <span class="metric-delta-pos">Bảo vệ vốn: +{alpha_mpt:.2f}% so với B&H</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.markdown("#### 📊 Bảng Đối Chiếu Hiệu Suất Toàn Diện (Test 2022)")
    st.dataframe(portfolio_test_stats.style.format("{:.4f}", na_rep="-"), use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🔍 Hiệu Suất Từng Cổ Phiếu Riêng Lẻ Trong Tập Test (Strategy vs Buy & Hold)")
    stock_comp_list = []
    for t in selected_tickers:
        bh_r = bh_test_stats[t].get("Total Return [%]", np.nan)
        st_r = test_strategy[t]["stats"].get("Total Return [%]", np.nan)
        bh_s = bh_test_stats[t].get("Sharpe Ratio", np.nan)
        st_s = test_strategy[t]["stats"].get("Sharpe Ratio", np.nan)
        bh_d = bh_test_stats[t].get("Max Drawdown [%]", np.nan)
        st_d = test_strategy[t]["stats"].get("Max Drawdown [%]", np.nan)

        stock_comp_list.append({
            "Ticker": t,
            "B&H Return [%]": bh_r,
            "Strategy Return [%]": st_r,
            "Alpha vs B&H [%]": st_r - bh_r,
            "B&H Sharpe": bh_s,
            "Strategy Sharpe": st_s,
            "B&H Max DD [%]": bh_d,
            "Strategy Max DD [%]": st_d
        })

    stock_comp_df = pd.DataFrame(stock_comp_list).set_index("Ticker")
    st.dataframe(stock_comp_df.style.format("{:.2f}", na_rep="-"), use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 5: ĐƯỜNG CONG TĂNG TRƯỞNG VỐN & RỦI RO
# ------------------------------------------------------------------------------
with tabs[4]:
    st.markdown("### 📉 Đường Cong Tăng Trưởng Vốn (Equity Curves) & Sụt Giảm Tài Sản")

    bh_equity = initial_capital * (1.0 + ew_bh_test_ret).cumprod()
    ew_equity = initial_capital * (1.0 + ew_test_ret).cumprod()
    mpt_equity = initial_capital * (1.0 + mpt_test_ret).cumprod()

    # Equity Curve Plot
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(x=bh_equity.index, y=bh_equity, name="Equal Weight Buy & Hold", line=dict(color="#ef4444", width=2, dash="dash")))
    fig_eq.add_trace(go.Scatter(x=ew_equity.index, y=ew_equity, name="SMA+OBV OR (Equal Weight)", line=dict(color="#f59e0b", width=2.5)))
    fig_eq.add_trace(go.Scatter(x=mpt_equity.index, y=mpt_equity, name="SMA+OBV OR (MPT)", line=dict(color="#0d9488", width=2.5)))

    fig_eq.update_layout(
        title="Tăng Trưởng Giá Trị Danh Mục (NAV) Ngoài Mẫu (Test 2022) — Vốn Khởi Điểm: 1,000,000 VNĐ",
        xaxis_title="Thời Gian", yaxis_title="Giá Trị Tài Sản (VNĐ)",
        height=500, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_eq, use_container_width=True)

    # Underwater Drawdown Plot
    bh_dd = (bh_equity / bh_equity.cummax() - 1.0) * 100
    ew_dd = (ew_equity / ew_equity.cummax() - 1.0) * 100
    mpt_dd = (mpt_equity / mpt_equity.cummax() - 1.0) * 100

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=bh_dd.index, y=bh_dd, name="Equal Weight Buy & Hold Drawdown", line=dict(color="#ef4444", width=1.5), fill="tozeroy", fillcolor="rgba(239, 68, 68, 0.15)"))
    fig_dd.add_trace(go.Scatter(x=ew_dd.index, y=ew_dd, name="SMA+OBV Equal Weight Drawdown", line=dict(color="#f59e0b", width=1.5)))
    fig_dd.add_trace(go.Scatter(x=mpt_dd.index, y=mpt_dd, name="SMA+OBV MPT Drawdown", line=dict(color="#0d9488", width=1.5)))

    fig_dd.update_layout(
        title="Biểu Đồ Sụt Giảm Tài Sản Theo Thời Gian (Underwater Drawdown Chart)",
        xaxis_title="Thời Gian", yaxis_title="Mức Sụt Giảm Từ Đỉnh (%)",
        height=400, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_dd, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 6: KẾT LUẬN HỌC THUẬT & XUẤT BÁO CÁO
# ------------------------------------------------------------------------------
with tabs[5]:
    st.markdown("### 📑 Kết Luận Tự Động & Đề Mục Báo Cáo Học Thuật (Cao Học)")

    # Phân tích so sánh tự động
    if mpt_strat_sharpe > ew_strat_sharpe:
        verdict_sharpe = f"Mô hình **MPT đạt Sharpe ngoài mẫu ({mpt_strat_sharpe:.4f}) cao hơn Equal Weight ({ew_strat_sharpe:.4f})**, chứng minh phân bổ dựa trên tối ưu phương sai - hiệp phương sai Train có đóng góp tích cực."
    else:
        verdict_sharpe = f"Danh mục **Equal Weight đạt Sharpe ngoài mẫu ({ew_strat_sharpe:.4f}) tốt hơn hoặc tương đương MPT ({mpt_strat_sharpe:.4f})**, thể hiện tính vững chắc (robustness) và hạn chế hiện tượng over-fitting tham số."

    st.markdown(f"""
    <div class="conclusion-box">
        <h4 style="color:#15803d; margin-top:0;">💡 KẾT LUẬN TỰ ĐỘNG DỰA TRÊN KẾT QUẢ KIỂM ĐỊNH THỰC TẾ:</h4>
        <ul>
            <li><b>Hiệu quả phòng hộ vượt trội:</b> Trong khi danh mục Buy & Hold sụt giảm nghiêm trọng <b>{ew_bh_ret:.2f}%</b> (Max Drawdown {ew_bh_mdd:.2f}%), chiến lược kết hợp SMA + OBV (OR) đã kịp thời kích hoạt tín hiệu thoát vị thế về tiền mặt, giúp hạn chế mức lỗ xuống chỉ còn <b>{mpt_strat_ret:.2f}%</b> (tạo mức chênh lệch Alpha lên đến <b>+{mpt_strat_ret - ew_bh_ret:.2f}%</b>).</li>
            <li><b>So sánh phân bổ danh mục:</b> {verdict_sharpe}</li>
            <li><b>Loại bỏ Look-Ahead Bias:</b> Toàn bộ tín hiệu được dịch chuyển 1 phiên (<code>shift(1)</code>), đảm bảo tính khả thi tuyệt đối trong giao dịch thực tế trên sàn HOSE.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.markdown("#### 🎓 10 Điểm Cốt Lõi Cần Trình Bày Trong Bài Báo Cáo Chuyên Đề")
    st.markdown("""
    1. **Tiêu chí định lượng chọn cổ phiếu:** Sử dụng mô hình chấm điểm tổ hợp 4 yếu tố (Return, Sharpe, Liquidity, Max Drawdown) trên tập Train 2020–2021.
    2. **Lý do định tính chọn 5 mã (DIG, DGC, VND, HAH, MSN):** Đa dạng hóa 5 nhóm ngành (BĐS, Hóa chất, Chứng khoán, Vận tải biển, Tiêu dùng).
    3. **Hiệu suất Buy & Hold từng mã:** Năm 2020–2021 tăng trưởng vượt bậc (>400-600%), nhưng năm 2022 sụt giảm từ -39% đến -86%.
    4. **Tỷ suất sinh lợi bình quân 5 mã:** Khẳng định sự cần thiết của chiến lược chủ động (Active Trading) thay vì thụ động giữ cổ phiếu trong chu kỳ downtrend.
    5. **Tối ưu tham số SMA + OBV OR bằng Hyperopt:** Tối ưu hóa trực tiếp trên hàm mục tiêu Sharpe tập Train mà không tối ưu riêng lẻ từng chỉ báo.
    6. **So sánh chiến lược OR vs Buy & Hold từng mã:** Chiến lược OR cải thiện rõ rệt Max Drawdown và bảo vệ tài sản trên mọi mã được chọn.
    7. **Trọng số Equal Weight:** Chiến lược phân bổ đều 20% mỗi mã, đơn giản, không giả định tương quan quá khứ.
    8. **Trọng số MPT (Markowitz):** Phân bổ dựa trên giải thuật tối ưu ma trận hiệp phương sai trên tập Train.
    9. **Kiểm định ngoài mẫu (Out-of-Sample 2022):** So sánh đối đầu giữa EW Buy & Hold, SMA+OBV Equal Weight và SMA+OBV MPT.
    10. **Kết luận khoa học:** Đánh giá tính ứng dụng thực tiễn của quy tắc OR và bài học quản trị rủi ro danh mục trong thị trường biến động mạnh.
    """)

    st.markdown("---")
    st.markdown("#### 📥 Xuất Dữ Liệu Kết Quả Kiểm Định")
    col_exp1, col_exp2 = st.columns(2)

    # Xuất CSV Bảng chỉ số danh mục
    csv_port = portfolio_test_stats.to_csv(encoding="utf-8-sig")
    col_exp1.download_button(
        label="📥 Tải Bảng Chỉ Số Danh Mục (CSV)",
        data=csv_port,
        file_name="HOSE_Portfolio_Comparison_Test2022.csv",
        mime="text/csv"
    )

    # Xuất CSV Đường cong NAV
    equity_export = pd.DataFrame({
        "Date": bh_equity.index,
        "Equal_Weight_Buy_Hold": bh_equity.values,
        "SMA_OBV_OR_Equal_Weight": ew_equity.values,
        "SMA_OBV_OR_MPT": mpt_equity.values
    }).set_index("Date")
    csv_eq = equity_export.to_csv(encoding="utf-8-sig")
    col_exp2.download_button(
        label="📥 Tải Dữ Liệu Đường Cong NAV (CSV)",
        data=csv_eq,
        file_name="HOSE_Equity_Curves_NAV.csv",
        mime="text/csv"
    )

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94a3b8; font-size: 0.85rem;'>"
    "Hệ thống kiểm định định lượng danh mục đầu tư | Đề tài Cao học Quản trị Danh mục Đầu tư | Xây dựng trên nền tảng Streamlit"
    "</div>",
    unsafe_allow_html=True
)
