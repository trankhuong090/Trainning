"""
=============================================================================
ỨNG DỤNG KIỂM ĐỊNH CHIẾN LƯỢC ĐẦU TƯ: KẾT HỢP CHỈ BÁO SMA VÀ OBV
Nhóm nghiên cứu: Phân tích Định lượng & Quản lý Danh mục Đầu tư (Nhóm 5 - ACB)
Dành cho: Kiểm định chiến lược đầu tư chứng khoán (Backtesting & Train/Test Split)
Triển khai: Tương thích 100% với Streamlit Cloud (Python 3.10+, NumPy 1.x & 2.x, Pandas 2.x+)
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import io

# -----------------------------------------------------------------------------
# 1. CẤU HÌNH TRANG WEB (PAGE CONFIG & STYLING)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Kiểm Định Chiến Lược SMA + OBV | Backtest Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho giao diện hiện đại, chuyên nghiệp
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1565C0, #00897B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 4px solid #1565C0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        margin-bottom: 10px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. HÀM TÍNH TOÁN KỸ THUẬT & TÍN HIỆU (TECHNICAL INDICATORS & SIGNALS)
# -----------------------------------------------------------------------------
def calculate_indicators(df, ma_short, ma_long, obv_window):
    """
    Tính toán 2 đường SMA (ngắn hạn & dài hạn) và chỉ báo OBV kèm đường trung bình OBV_MA.
    """
    data = df.copy()
    close = data['Close']
    volume = data['Volume']

    # 1. Hai đường SMA (Simple Moving Average)
    data['SMA_Short'] = close.rolling(window=int(ma_short)).mean()
    data['SMA_Long'] = close.rolling(window=int(ma_long)).mean()

    # 2. On-Balance Volume (OBV)
    price_diff = close.diff()
    direction = np.where(price_diff > 0, 1.0, np.where(price_diff < 0, -1.0, 0.0))
    direction[0] = 0.0
    data['OBV'] = (volume * direction).cumsum()

    # 3. Đường trung bình động của OBV (OBV Moving Average)
    data['OBV_MA'] = data['OBV'].rolling(window=int(obv_window)).mean()

    return data


def find_position_sma(df, ma_short, ma_long):
    """
    Chiến lược SMA: Giao cắt 2 đường trung bình động.
    - Mua (Golden Cross): Đường SMA ngắn cắt lên trên đường SMA dài.
    - Bán (Death Cross): Đường SMA ngắn cắt xuống dưới đường SMA dài.
    """
    position = pd.Series(0, index=df.index, name="position", dtype=int)
    if ma_short >= ma_long:
        return position

    ma_s = df['Close'].rolling(window=int(ma_short)).mean()
    ma_l = df['Close'].rolling(window=int(ma_long)).mean()

    buy_signal = (ma_s > ma_l) & (ma_s.shift(1) <= ma_l.shift(1))
    sell_signal = (ma_s < ma_l) & (ma_s.shift(1) >= ma_l.shift(1))

    position.loc[buy_signal] = 1
    position.loc[sell_signal] = -1
    return position


def find_position_obv(df, obv_window):
    """
    Chiến lược OBV: Giao cắt giữa OBV và đường trung bình động của chính nó.
    - Mua: OBV cắt lên trên OBV_MA.
    - Bán: OBV cắt xuống dưới OBV_MA.
    """
    position = pd.Series(0, index=df.index, name="position", dtype=int)
    close = df['Close']
    volume = df['Volume']

    price_diff = close.diff()
    direction = np.where(price_diff > 0, 1.0, np.where(price_diff < 0, -1.0, 0.0))
    direction[0] = 0.0
    obv = (volume * direction).cumsum()
    obv_ma = obv.rolling(window=int(obv_window)).mean()

    buy_signal = (obv > obv_ma) & (obv.shift(1) <= obv_ma.shift(1))
    sell_signal = (obv < obv_ma) & (obv.shift(1) >= obv_ma.shift(1))

    position.loc[buy_signal] = 1
    position.loc[sell_signal] = -1
    return position


def find_position_combined(df, ma_short, ma_long, obv_window, mode="AND"):
    """
    Chiến lược KẾT HỢP SMA + OBV:
    - Quy tắc AND (Đồng thời): Mua khi cả hai cùng cho tín hiệu Mua, Bán khi cả hai cùng cho tín hiệu Bán.
    - Quy tắc State-Confirmed (Xu hướng + Dòng tiền):
      Mua khi SMA_Short > SMA_Long VÀ OBV cắt lên OBV_MA.
      Bán khi SMA_Short cắt xuống SMA_Long HOẶC OBV cắt xuống OBV_MA.
    """
    sma_pos = find_position_sma(df, ma_short, ma_long)
    obv_pos = find_position_obv(df, obv_window)

    position = pd.Series(0, index=df.index, name="position", dtype=int)

    if mode == "AND":
        # Đúng theo Cell 10 của Notebook Nhóm 5
        buy_signal = (sma_pos == 1) & (obv_pos == 1)
        sell_signal = (sma_pos == -1) & (obv_pos == -1)
        position.loc[buy_signal] = 1
        position.loc[sell_signal] = -1
    elif mode == "CONFIRMED":
        # Chế độ xu hướng kết hợp dòng tiền
        ma_s = df['Close'].rolling(window=int(ma_short)).mean()
        ma_l = df['Close'].rolling(window=int(ma_long)).mean()
        price_diff = df['Close'].diff()
        direction = np.where(price_diff > 0, 1.0, np.where(price_diff < 0, -1.0, 0.0))
        direction[0] = 0.0
        obv = (df['Volume'] * direction).cumsum()
        obv_ma = obv.rolling(window=int(obv_window)).mean()

        buy_signal = (ma_s > ma_l) & (obv > obv_ma) & ((obv.shift(1) <= obv_ma.shift(1)) | (ma_s.shift(1) <= ma_l.shift(1)))
        sell_signal = (ma_s < ma_l) | (obv < obv_ma)
        position.loc[buy_signal] = 1
        position.loc[sell_signal] = -1

    return position


# -----------------------------------------------------------------------------
# 3. ĐỘNG CƠ MÔ PHỎNG BACKTEST THUẦN TÚY (FAST EVENT-DRIVEN SIMULATOR)
# Mô phỏng đúng quy tắc GeneralStrategy trong backtesting.py của notebook
# -----------------------------------------------------------------------------
def run_backtest_simulation(df, position_series, initial_cash=1_000_000, commission=0.0):
    """
    Thực hiện backtest mô phỏng danh mục:
    - Khi signal == 1 và chưa có vị thế: Mua toàn bộ vốn (buy on close).
    - Khi signal == -1 và đang có vị thế: Bán toàn bộ cổ phiếu đóng vị thế.
    """
    closes = df['Close'].values
    dates = df.index
    signals = position_series.values
    n = len(df)

    cash = float(initial_cash)
    shares = 0
    in_position = False
    buy_price = 0.0
    entry_date = None
    entry_idx = 0
    total_cost = 0.0

    equity_history = np.zeros(n, dtype=float)
    pos_history = np.zeros(n, dtype=int)
    cash_history = np.zeros(n, dtype=float)
    trades = []

    for i in range(n):
        curr_price = float(closes[i])
        curr_date = dates[i]
        sig = signals[i]

        # Kiểm tra đóng vị thế nếu đang có cổ phiếu
        if in_position and sig == -1:
            revenue = shares * curr_price
            fee_exit = revenue * commission
            net_revenue = revenue - fee_exit

            cash += net_revenue
            pnl_pct = (net_revenue - total_cost) / total_cost * 100.0 if total_cost > 0 else 0.0
            pnl_cash = net_revenue - total_cost

            trades.append({
                'STT': len(trades) + 1,
                'Ngày Mua': entry_date,
                'Giá Mua': round(buy_price, 1),
                'Ngày Bán': curr_date,
                'Giá Bán': round(curr_price, 1),
                'Số CP': int(shares),
                'Lợi Nhuận (%)': round(pnl_pct, 2),
                'Lợi Nhuận (VNĐ)': round(pnl_cash, 0),
                'Thời Gian Giữ (phiên)': int(i - entry_idx),
                'Lý Do Bán': 'Tín Hiệu Bán (Signal = -1)'
            })

            in_position = False
            shares = 0
            buy_price = 0.0

        # Kiểm tra mở vị thế nếu chưa có cổ phiếu
        elif not in_position and sig == 1:
            fee_factor = 1.0 + commission
            affordable_shares = int(cash / (curr_price * fee_factor))

            if affordable_shares > 0:
                cost_shares = affordable_shares * curr_price
                fee_entry = cost_shares * commission
                total_cost = cost_shares + fee_entry

                cash -= total_cost
                shares = affordable_shares
                buy_price = curr_price
                entry_date = curr_date
                entry_idx = i
                in_position = True

        # Ghi nhận giá trị danh mục cuối phiên
        current_val = cash + (shares * curr_price if in_position else 0.0)
        equity_history[i] = current_val
        pos_history[i] = 1 if in_position else 0
        cash_history[i] = cash

    # Xử lý lệnh còn mở ở phiên cuối cùng (Mark to market)
    if in_position:
        final_price = float(closes[-1])
        revenue = shares * final_price
        fee_exit = revenue * commission
        net_revenue = revenue - fee_exit
        pnl_pct = (net_revenue - total_cost) / total_cost * 100.0 if total_cost > 0 else 0.0
        pnl_cash = net_revenue - total_cost

        trades.append({
            'STT': len(trades) + 1,
            'Ngày Mua': entry_date,
            'Giá Mua': round(buy_price, 1),
            'Ngày Bán': dates[-1],
            'Giá Bán': round(final_price, 1),
            'Số CP': int(shares),
            'Lợi Nhuận (%)': round(pnl_pct, 2),
            'Lợi Nhuận (VNĐ)': round(pnl_cash, 0),
            'Thời Gian Giữ (phiên)': int(n - 1 - entry_idx),
            'Lý Do Bán': 'Kết Thúc Giai Đoạn (Đóng vị thế)'
        })

    equity_df = pd.DataFrame({
        'Date': dates,
        'Equity': equity_history,
        'Position': pos_history,
        'Cash': cash_history
    }).set_index('Date')

    trades_df = pd.DataFrame(trades)

    # Tính toán các chỉ số thống kê hiệu suất
    metrics = compute_metrics(equity_df, trades_df, initial_cash, closes)

    return equity_df, trades_df, metrics


def compute_metrics(equity_df, trades_df, initial_cash, closes):
    """
    Tính toán các chỉ số tài chính định lượng chuẩn xác (Return %, Sharpe, Max Drawdown, Win Rate,...).
    """
    eq = equity_df['Equity']
    final_eq = float(eq.iloc[-1])
    total_return = float((final_eq - initial_cash) / initial_cash * 100.0)

    n_days = len(equity_df)
    years = max(n_days / 252.0, 0.05)
    cagr = float(((final_eq / initial_cash) ** (1.0 / years) - 1.0) * 100.0)

    # Drawdown
    cummax = eq.cummax()
    dd = (eq - cummax) / cummax
    max_dd = float(dd.min() * 100.0)

    # Sharpe Ratio (annualized, Rf = 0)
    returns = eq.pct_change().dropna()
    std_ret = float(returns.std())
    sharpe = float((returns.mean() / std_ret) * np.sqrt(252)) if std_ret > 0 else 0.0

    # Sortino Ratio
    neg_ret = returns[returns < 0]
    std_neg = float(neg_ret.std()) if len(neg_ret) > 1 else 0.0
    sortino = float((returns.mean() / std_neg) * np.sqrt(252)) if std_neg > 0 else 0.0

    # Trade stats
    n_trades = len(trades_df)
    if n_trades > 0:
        win_trades = trades_df[trades_df['Lợi Nhuận (%)'] > 0]
        loss_trades = trades_df[trades_df['Lợi Nhuận (%)'] <= 0]

        win_rate = float(len(win_trades) / n_trades * 100.0)
        avg_profit = float(trades_df['Lợi Nhuận (%)'].mean())

        total_gain = float(win_trades['Lợi Nhuận (VNĐ)'].sum()) if len(win_trades) > 0 else 0.0
        total_loss = float(abs(loss_trades['Lợi Nhuận (VNĐ)'].sum())) if len(loss_trades) > 0 else 0.0
        profit_factor = float(total_gain / total_loss) if total_loss > 0 else (np.nan if total_gain > 0 else 0.0)
        avg_hold = float(trades_df['Thời Gian Giữ (phiên)'].mean())
    else:
        win_rate = 0.0
        avg_profit = 0.0
        profit_factor = np.nan
        avg_hold = 0.0

    # Benchmark: Buy & Hold
    bh_series = pd.Series(closes, dtype=float)
    bh_return = float((closes[-1] - closes[0]) / closes[0] * 100.0)
    bh_cagr = float(((closes[-1] / closes[0]) ** (1.0 / years) - 1.0) * 100.0)
    bh_cummax = bh_series.cummax()
    bh_mdd = float(((bh_series - bh_cummax) / bh_cummax).min() * 100.0)

    bh_daily_ret = bh_series.pct_change().dropna()
    bh_std = float(bh_daily_ret.std())
    bh_sharpe = float((bh_daily_ret.mean() / bh_std) * np.sqrt(252)) if bh_std > 0 else 0.0

    return {
        'Vốn Cuối Kỳ (VNĐ)': final_eq,
        'Tổng Lợi Nhuận (%)': total_return,
        'Lợi Nhuận Năm CAGR (%)': cagr,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown (%)': max_dd,
        'Số Giao Dịch': n_trades,
        'Tỷ Lệ Thắng (%)': win_rate,
        'Profit Factor': profit_factor,
        'LN Trung Bình/Lệnh (%)': avg_profit,
        'Số Phiên Giữ TB': avg_hold,
        'Buy & Hold Return (%)': bh_return,
        'Buy & Hold CAGR (%)': bh_cagr,
        'Buy & Hold Sharpe': bh_sharpe,
        'Buy & Hold Max DD (%)': bh_mdd
    }


# -----------------------------------------------------------------------------
# 4. HÀM TỐI ƯU HÓA THAM SỐ (FAST GRID SEARCH ENGINE)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def optimize_sma(df, short_range, long_range, initial_cash=1_000_000, commission=0.0):
    """
    Quét không gian tham số SMA (ma_short, ma_long) để tối đa hóa Lợi Nhuận hoặc Sharpe.
    """
    results = []
    best_ret = -np.inf
    best_params = {}

    for s in short_range:
        for l in long_range:
            if s >= l:
                continue
            pos = find_position_sma(df, s, l)
            _, tr_df, met = run_backtest_simulation(df, pos, initial_cash, commission)

            ret = met['Tổng Lợi Nhuận (%)']
            sharpe = met['Sharpe Ratio']
            n_tr = met['Số Giao Dịch']

            results.append({
                'ma_short': int(s),
                'ma_long': int(l),
                'Return [%]': float(ret),
                'Sharpe Ratio': float(sharpe),
                'Max Drawdown [%]': float(met['Max Drawdown (%)']),
                'Trades': int(n_tr)
            })

            if ret > best_ret:
                best_ret = ret
                best_params = {'ma_short': int(s), 'ma_long': int(l)}

    return best_params, pd.DataFrame(results)


@st.cache_data(show_spinner=False)
def optimize_obv(df, window_range, initial_cash=1_000_000, commission=0.0):
    """
    Quét không gian tham số OBV (obv_window) để tối đa hóa Lợi Nhuận.
    """
    results = []
    best_ret = -np.inf
    best_params = {}

    for w in window_range:
        pos = find_position_obv(df, w)
        _, tr_df, met = run_backtest_simulation(df, pos, initial_cash, commission)

        ret = met['Tổng Lợi Nhuận (%)']
        sharpe = met['Sharpe Ratio']
        n_tr = met['Số Giao Dịch']

        results.append({
            'obv_window': int(w),
            'Return [%]': float(ret),
            'Sharpe Ratio': float(sharpe),
            'Max Drawdown [%]': float(met['Max Drawdown (%)']),
            'Trades': int(n_tr)
        })

        if ret > best_ret:
            best_ret = ret
            best_params = {'obv_window': int(w)}

    return best_params, pd.DataFrame(results)


# -----------------------------------------------------------------------------
# 5. HÀM NẠP DỮ LIỆU CHUẨN XÁC (DATA LOADING & PARSING)
# -----------------------------------------------------------------------------
@st.cache_data
def load_stock_data(uploaded_file=None):
    """
    Nạp dữ liệu từ file tải lên hoặc file ACB.csv mặc định.
    Chuẩn hóa theo đúng Cell 4 của Notebook Nhóm 5.
    """
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
        else:
            df = pd.read_csv("ACB.csv", encoding="utf-8-sig")
    except Exception as e:
        return None, f"Không thể đọc file: {str(e)}"

    df.columns = df.columns.str.strip().str.lower()

    # Nhận diện cột Date
    date_col = None
    for col in df.columns:
        if col in ['date', 'ngày', 'time', 'ngay', 'datetime']:
            date_col = col
            break

    if date_col is None:
        return None, "Dữ liệu thiếu cột ngày giao dịch (Date)!"

    df['date'] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date')

    # Nhận diện các cột Open, High, Low, Close, Volume
    rename_map = {}
    for col in df.columns:
        c = col.lower()
        if c in ['open', 'mo_cua']: rename_map[col] = 'Open'
        elif c in ['high', 'cao_nhat']: rename_map[col] = 'High'
        elif c in ['low', 'thap_nhat']: rename_map[col] = 'Low'
        elif c in ['close', 'dong_cua', 'adj_close', 'price']: rename_map[col] = 'Close'
        elif c in ['volume', 'khoi_luong', 'vol']: rename_map[col] = 'Volume'

    df = df.rename(columns=rename_map)

    # Đảm bảo có đủ Close và Volume
    if 'Close' not in df.columns or 'Volume' not in df.columns:
        return None, "Dữ liệu bắt buộc phải có cột Close (Giá đóng cửa) và Volume (Khối lượng)!"

    if 'Open' not in df.columns: df['Open'] = df['Close']
    if 'High' not in df.columns: df['High'] = df['Close']
    if 'Low' not in df.columns: df['Low'] = df['Close']

    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
    df = df.drop_duplicates(subset='date', keep='last')
    df = df.set_index('date').sort_index()

    return df, None


# -----------------------------------------------------------------------------
# 6. GIAO DIỆN CHÍNH (STREAMLIT APP & SIDEBAR)
# -----------------------------------------------------------------------------
def main():
    st.markdown('<div class="main-title">📊 HỆ THỐNG KIỂM ĐỊNH CHIẾN LƯỢC: SMA + OBV</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Nghiên cứu kiểm định định lượng, phân tách Train/Test và tối ưu hóa tham số (Theo mô hình Nhóm 5 - ACB)</div>', unsafe_allow_html=True)

    # ----------------- SIDEBAR: CẤU HÌNH & THAM SỐ -----------------
    with st.sidebar:
        st.header("⚙️ Thiết Lập Dữ Liệu & Tham Số")

        # 1. Nguồn dữ liệu
        data_source = st.radio(
            "📂 Nguồn dữ liệu:",
            ["Dữ liệu mẫu ACB.csv", "Tải lên file CSV cá nhân"]
        )

        uploaded_file = None
        if data_source == "Tải lên file CSV cá nhân":
            uploaded_file = st.file_uploader("Chọn file CSV:", type=['csv'])

        df, err = load_stock_data(uploaded_file)
        if err or df is None:
            st.error(f"Lỗi nạp dữ liệu: {err}")
            st.info("Vui lòng tải lên file CSV có chứa các cột: Date, Close, Volume.")
            return

        st.success(f"Đã nạp: {len(df):,} phiên ({df.index.min().strftime('%d/%m/%Y')} - {df.index.max().strftime('%d/%m/%Y')})")

        # 2. Phân chia Train / Test Split
        st.subheader("📅 Phân Chia Tập Dữ Liệu (Train / Test)")
        split_method = st.radio("Cách phân chia:", ["Theo tỷ lệ phần trăm (80/20 như Notebook)", "Theo năm tùy chỉnh"])

        if split_method == "Theo tỷ lệ phần trăm (80/20 như Notebook)":
            train_ratio = st.slider("Tỷ lệ tập Train (%):", min_value=50, max_value=90, value=80, step=5) / 100.0
            split_idx = int(len(df) * train_ratio)
            train_df = df.iloc[:split_idx].copy()
            test_df = df.iloc[split_idx:].copy()
        else:
            min_y = int(df.index.min().year)
            max_y = int(df.index.max().year)
            col_y1, col_y2 = st.columns(2)
            with col_y1:
                t_end_y = st.number_input("Train Đến Hết Năm:", min_value=min_y, max_value=max_y-1, value=min(2021, max_y-1))
            with col_y2:
                te_start_y = st.number_input("Test Từ Năm:", min_value=min_y+1, max_value=max_y, value=min(2022, max_y))

            train_df = df.loc[df.index.year <= t_end_y].copy()
            test_df = df.loc[df.index.year >= te_start_y].copy()

        st.caption(f"🔹 **Train:** {len(train_df):,} phiên ({train_df.index.min().strftime('%d/%m/%Y')} → {train_df.index.max().strftime('%d/%m/%Y')})")
        st.caption(f"🔹 **Test:** {len(test_df):,} phiên ({test_df.index.min().strftime('%d/%m/%Y')} → {test_df.index.max().strftime('%d/%m/%Y')})")

        # 3. Tham số chiến lược SMA & OBV
        st.subheader("🎯 Tham Số Chiến Lược")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            ma_short_val = st.number_input("SMA Ngắn (phiên):", min_value=5, max_value=250, value=150, step=5,
                                           help="Giá trị tối ưu của Nhóm 5 trên Train là 150.")
        with col_s2:
            ma_long_val = st.number_input("SMA Dài (phiên):", min_value=20, max_value=500, value=210, step=5,
                                          help="Giá trị tối ưu của Nhóm 5 trên Train là 210.")

        obv_win_val = st.number_input("Chu kỳ OBV MA (phiên):", min_value=2, max_value=100, value=5, step=1,
                                      help="Giá trị tối ưu của Nhóm 5 trên Train là 5.")

        combine_mode = st.selectbox(
            "Quy tắc kết hợp:",
            ["AND (Đồng thời cả 2)", "CONFIRMED (Xu hướng + Dòng tiền)"],
            index=0,
            help="AND: Cả SMA và OBV cùng phát lệnh mua/bán đồng thời (theo đúng Cell 10 notebook)."
        )
        mode_code = "AND" if "AND" in combine_mode else "CONFIRMED"

        # 4. Quản trị rủi ro & Chi phí
        with st.expander("💰 Vốn Ban Đầu & Phí Giao Dịch", expanded=False):
            init_cash = st.number_input("Vốn ban đầu (VNĐ)", min_value=1_000_000, max_value=10_000_000_000, value=1_000_000, step=1_000_000)
            comm_pct = st.number_input("Phí giao dịch (%)", min_value=0.0, max_value=1.0, value=0.0, step=0.05) / 100.0

        # Cấu hình nhanh
        st.markdown("---")
        st.markdown("💡 **Cấu Hình Nhanh Theo Bài Báo Cáo:**")
        col_btn1, col_btn2 = st.columns(2)
        if col_btn1.button("📌 Kinh Điển (50, 200, 20)"):
            st.session_state['quick_short'] = 50
            st.session_state['quick_long'] = 200
            st.session_state['quick_obv'] = 20
            st.rerun()

        if col_btn2.button("⭐ Tối Ưu Nhóm 5 (150, 210, 5)"):
            st.session_state['quick_short'] = 150
            st.session_state['quick_long'] = 210
            st.session_state['quick_obv'] = 5
            st.rerun()

        if 'quick_short' in st.session_state: ma_short_val = st.session_state['quick_short']
        if 'quick_long' in st.session_state: ma_long_val = st.session_state['quick_long']
        if 'quick_obv' in st.session_state: obv_win_val = st.session_state['quick_obv']

    if len(train_df) == 0 or len(test_df) == 0:
        st.error("Dữ liệu phân chia Train/Test không hợp lệ! Vui lòng chọn lại bộ lọc.")
        return

    # ----------------- CHẠY CẢ 3 CHIẾN LƯỢC TRÊN CẢ 2 TẬP DỮ LIỆU -----------------
    results = {'Train': {}, 'Test': {}}
    for phase_name, p_df in [('Train', train_df), ('Test', test_df)]:
        # SMA
        pos_sma = find_position_sma(p_df, ma_short_val, ma_long_val)
        eq_sma, tr_sma, met_sma = run_backtest_simulation(p_df, pos_sma, init_cash, comm_pct)

        # OBV
        pos_obv = find_position_obv(p_df, obv_win_val)
        eq_obv, tr_obv, met_obv = run_backtest_simulation(p_df, pos_obv, init_cash, comm_pct)

        # Combined
        pos_comb = find_position_combined(p_df, ma_short_val, ma_long_val, obv_win_val, mode=mode_code)
        eq_comb, tr_comb, met_comb = run_backtest_simulation(p_df, pos_comb, init_cash, comm_pct)

        # Indicators Data
        ind_data = calculate_indicators(p_df, ma_short_val, ma_long_val, obv_win_val)

        results[phase_name] = {
            'SMA': {'equity': eq_sma, 'trades': tr_sma, 'metrics': met_sma, 'position': pos_sma},
            'OBV': {'equity': eq_obv, 'trades': tr_obv, 'metrics': met_obv, 'position': pos_obv},
            'Combined': {'equity': eq_comb, 'trades': tr_comb, 'metrics': met_comb, 'position': pos_comb},
            'indicators': ind_data
        }

    # ----------------- TẠO TABS CHÍNH -----------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Bảng So Sánh Hiệu Suất (Train vs Test)",
        "📈 Biểu Đồ & Tín Hiệu Kỹ Thuật",
        "📑 Nhật Ký Lệnh (Trade Log)",
        "⚡ Tự Động Tối Ưu Hóa (Train)",
        "📖 Cơ Sở Lý Thuyết & Hướng Dẫn"
    ])

    # =========================================================================
    # TAB 1: BẢNG SO SÁNH HIỆU SUẤT (CELL 18 NOTEBOOK)
    # =========================================================================
    with tab1:
        st.subheader("🎯 Bảng So Sánh Hiệu Suất Các Chiến Lược (Train vs Test)")
        st.markdown("*So sánh kết quả theo đúng cấu trúc Cell 18 của Notebook: SMA vs OBV vs SMA+OBV và Benchmark Buy & Hold.*")

        table_rows = []
        strat_display = {'SMA': 'SMA (Giao Cắt MA)', 'OBV': 'OBV (Giao Cắt OBV MA)', 'Combined': 'SMA + OBV Kết Hợp'}

        for phase in ['Train', 'Test']:
            for s_key in ['SMA', 'OBV', 'Combined']:
                m = results[phase][s_key]['metrics']
                table_rows.append({
                    'Chiến Lược': strat_display[s_key],
                    'Tập Dữ Liệu': phase,
                    'Return [%]': float(m['Tổng Lợi Nhuận (%)']),
                    'Sharpe Ratio': float(m['Sharpe Ratio']),
                    'Max Drawdown [%]': float(m['Max Drawdown (%)']),
                    'Trades': int(m['Số Giao Dịch']),
                    'Win Rate [%]': float(m['Tỷ Lệ Thắng (%)']),
                    'CAGR [%]': float(m['Lợi Nhuận Năm CAGR (%)']),
                    'Profit Factor': float(m['Profit Factor']) if (np.isfinite(m['Profit Factor']) and not np.isnan(m['Profit Factor'])) else np.nan
                })

            # Buy & Hold
            m_sample = results[phase]['Combined']['metrics']
            table_rows.append({
                'Chiến Lược': 'Buy & Hold (Mua & Giữ)',
                'Tập Dữ Liệu': phase,
                'Return [%]': float(m_sample['Buy & Hold Return (%)']),
                'Sharpe Ratio': float(m_sample['Buy & Hold Sharpe']),
                'Max Drawdown [%]': float(m_sample['Buy & Hold Max DD (%)']),
                'Trades': 1,
                'Win Rate [%]': 100.0 if m_sample['Buy & Hold Return (%)'] > 0 else 0.0,
                'CAGR [%]': float(m_sample['Buy & Hold CAGR (%)']),
                'Profit Factor': np.nan
            })

        comparison_df = pd.DataFrame(table_rows)

        # Hiển thị bảng bằng column_config: An toàn tuyệt đối, không gặp lỗi UFuncNoLoopError
        st.dataframe(
            comparison_df,
            column_config={
                "Chiến Lược": st.column_config.TextColumn("Chiến Lược", width="medium"),
                "Tập Dữ Liệu": st.column_config.TextColumn("Tập Dữ Liệu", width="small"),
                "Return [%]": st.column_config.NumberColumn("Return [%]", format="%.2f%%"),
                "Sharpe Ratio": st.column_config.NumberColumn("Sharpe Ratio", format="%.2f"),
                "Max Drawdown [%]": st.column_config.NumberColumn("Max Drawdown [%]", format="%.2f%%"),
                "Trades": st.column_config.NumberColumn("Trades", format="%d"),
                "Win Rate [%]": st.column_config.NumberColumn("Win Rate [%]", format="%.1f%%"),
                "CAGR [%]": st.column_config.NumberColumn("CAGR [%]", format="%.2f%%"),
                "Profit Factor": st.column_config.NumberColumn("Profit Factor", format="%.2f"),
            },
            use_container_width=True,
            hide_index=True
        )

        # Metric cards tóm tắt
        st.markdown("### 🏆 Hiệu Suất Chiến Lược Kết Hợp (SMA + OBV)")
        c1, c2, c3, c4 = st.columns(4)
        c_train_m = results['Train']['Combined']['metrics']
        c_test_m = results['Test']['Combined']['metrics']

        c1.metric("Return [%] (Train / Test)",
                  f"{c_train_m['Tổng Lợi Nhuận (%)']:.2f}%",
                  delta=f"Test: {c_test_m['Tổng Lợi Nhuận (%)']:.2f}%")

        c2.metric("Sharpe Ratio (Train / Test)",
                  f"{c_train_m['Sharpe Ratio']:.2f}",
                  delta=f"Test: {c_test_m['Sharpe Ratio']:.2f}")

        c3.metric("Max Drawdown (Train / Test)",
                  f"{c_train_m['Max Drawdown (%)']:.2f}%",
                  delta=f"Test: {c_test_m['Max Drawdown (%)']:.2f}%",
                  delta_color="inverse")

        c4.metric("Số Lệnh Đã Thực Hiện",
                  f"{c_train_m['Số Giao Dịch']} lệnh (Train)",
                  delta=f"{c_test_m['Số Giao Dịch']} lệnh (Test)")

        # Biểu đồ so sánh Bar Chart giữa Train và Test
        st.markdown("### 📊 Trực Quan Hóa So Sánh Chỉ Số (Train vs Test)")
        chart_df = comparison_df[comparison_df['Chiến Lược'] != 'Buy & Hold (Mua & Giữ)'].copy()

        fig_bars = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Tổng Lợi Nhuận (Return %)', 'Tỷ Lệ Sharpe (Sharpe Ratio)', 'Mức Sụt Giảm Tối Đa (Max Drawdown %)'),
            horizontal_spacing=0.08
        )

        colors = {'Train': '#1565C0', 'Test': '#FF8F00'}
        for phase in ['Train', 'Test']:
            subset = chart_df[chart_df['Tập Dữ Liệu'] == phase]
            fig_bars.add_trace(
                go.Bar(name=f'{phase} - Return', x=subset['Chiến Lược'], y=subset['Return [%]'],
                       marker_color=colors[phase], text=subset['Return [%]'].apply(lambda x: f"{x:.1f}%"), textposition='outside',
                       showlegend=(phase == 'Train')),
                row=1, col=1
            )
            fig_bars.add_trace(
                go.Bar(name=f'{phase} - Sharpe', x=subset['Chiến Lược'], y=subset['Sharpe Ratio'],
                       marker_color=colors[phase], text=subset['Sharpe Ratio'].apply(lambda x: f"{x:.2f}"), textposition='outside',
                       showlegend=False),
                row=1, col=2
            )
            fig_bars.add_trace(
                go.Bar(name=f'{phase} - MaxDD', x=subset['Chiến Lược'], y=subset['Max Drawdown [%]'],
                       marker_color=colors[phase], text=subset['Max Drawdown [%]'].apply(lambda x: f"{x:.1f}%"), textposition='outside',
                       showlegend=False),
                row=1, col=3
            )

        fig_bars.update_layout(
            barmode='group', height=450, template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
            margin=dict(l=20, r=20, t=60, b=20)
        )
        st.plotly_chart(fig_bars, use_container_width=True)

        # Biểu đồ đường cong vốn (Equity Curve)
        st.markdown("### 📈 Đường Cong Vốn (Equity Curve) So Sánh với Buy & Hold")
        view_phase = st.radio("Chọn giai đoạn hiển thị đường cong vốn:", ["Tập Test (Out-of-Sample)", "Tập Train (In-Sample)"], horizontal=True)
        active_phase = 'Test' if "Test" in view_phase else 'Train'

        fig_eq = go.Figure()
        palette = {'SMA': '#AB47BC', 'OBV': '#26A69A', 'Combined': '#1565C0'}

        for s_key in ['SMA', 'OBV', 'Combined']:
            eq_series = results[active_phase][s_key]['equity']['Equity']
            fig_eq.add_trace(go.Scatter(
                x=eq_series.index, y=eq_series.values,
                mode='lines', name=strat_display[s_key],
                line=dict(width=2.5 if s_key == 'Combined' else 1.5, color=palette[s_key])
            ))

        # Benchmark Buy & Hold
        active_df = test_df if active_phase == 'Test' else train_df
        bh_curve = init_cash * (active_df['Close'] / active_df['Close'].iloc[0])
        fig_eq.add_trace(go.Scatter(
            x=active_df.index, y=bh_curve.values,
            mode='lines', name='Buy & Hold Benchmark',
            line=dict(dash='dash', color='#757575', width=1.5)
        ))

        fig_eq.update_layout(
            title=f"Tăng Trưởng Danh Mục (Vốn Khởi Điểm: {init_cash:,.0f} VNĐ) - Giai Đoạn {active_phase}",
            xaxis_title="Thời Gian", yaxis_title="Giá Trị Danh Mục (VNĐ)",
            template='plotly_white', height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_eq, use_container_width=True)


    # =========================================================================
    # TAB 2: BIỂU ĐỒ NẾN & TÍN HIỆU KỸ THUẬT (SMA + OBV)
    # =========================================================================
    with tab2:
        st.subheader("🔍 Phân Tích Biểu Đồ Kỹ Thuật & Tín Hiệu Giao Dịch")

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            sel_strat = st.selectbox("Chọn Chiến Lược Xem Tín Hiệu:", ["Combined", "SMA", "OBV"],
                                     format_func=lambda x: strat_display[x])
        with col_c2:
            sel_phase = st.selectbox("Chọn Giai Đoạn Phân Tích:", ["Test", "Train"])

        active_res = results[sel_phase][sel_strat]
        ind_data = results[sel_phase]['indicators']
        tr_data = active_res['trades']

        fig_tech = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            row_heights=[0.65, 0.35],
            subplot_titles=(f"Giá & 2 Đường SMA ({ma_short_val}, {ma_long_val})", f"Chỉ Báo OBV & Đường OBV MA ({obv_win_val})")
        )

        # 1. Giá và 2 đường SMA
        fig_tech.add_trace(
            go.Scatter(x=ind_data.index, y=ind_data['Close'], mode='lines', name='Close Price', line=dict(color='#37474F', width=1.5)),
            row=1, col=1
        )
        fig_tech.add_trace(
            go.Scatter(x=ind_data.index, y=ind_data['SMA_Short'], mode='lines', name=f'SMA Short ({ma_short_val})', line=dict(color='#FF5722', width=2)),
            row=1, col=1
        )
        fig_tech.add_trace(
            go.Scatter(x=ind_data.index, y=ind_data['SMA_Long'], mode='lines', name=f'SMA Long ({ma_long_val})', line=dict(color='#1E88E5', width=2)),
            row=1, col=1
        )

        # Đánh dấu các điểm Mua / Bán từ Trade Log
        if len(tr_data) > 0:
            fig_tech.add_trace(
                go.Scatter(x=tr_data['Ngày Mua'], y=tr_data['Giá Mua'], mode='markers', name='Điểm MUA',
                           marker=dict(symbol='triangle-up', size=11, color='#2E7D32', line=dict(width=1, color='white'))),
                row=1, col=1
            )
            fig_tech.add_trace(
                go.Scatter(x=tr_data['Ngày Bán'], y=tr_data['Giá Bán'], mode='markers', name='Điểm BÁN',
                           marker=dict(symbol='triangle-down', size=11, color='#C62828', line=dict(width=1, color='white'))),
                row=1, col=1
            )

        # 2. Chỉ báo OBV và OBV MA
        fig_tech.add_trace(
            go.Scatter(x=ind_data.index, y=ind_data['OBV'], mode='lines', name='OBV', line=dict(color='#00897B', width=1.5)),
            row=2, col=1
        )
        fig_tech.add_trace(
            go.Scatter(x=ind_data.index, y=ind_data['OBV_MA'], mode='lines', name=f'OBV MA ({obv_win_val})', line=dict(color='#E53935', width=1.5, dash='dot')),
            row=2, col=1
        )

        fig_tech.update_layout(
            height=650, template='plotly_white',
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_tech, use_container_width=True)


    # =========================================================================
    # TAB 3: NHẬT KÝ GIAO DỊCH (TRADE LOG)
    # =========================================================================
    with tab3:
        st.subheader("📑 Chi Tiết Các Lệnh Giao Dịch Đã Thực Hiện")

        col_l1, col_l2 = st.columns(2)
        with col_l1:
            log_strat = st.selectbox("Xem Lệnh Của Chiến Lược:", ["Combined", "SMA", "OBV"],
                                     key="log_strat_view", format_func=lambda x: strat_display[x])
        with col_l2:
            log_phase = st.selectbox("Giai Đoạn:", ["Test", "Train"], key="log_phase_view")

        trades_table = results[log_phase][log_strat]['trades'].copy()

        if len(trades_table) == 0:
            st.info("Chiến lược không phát sinh giao dịch nào trong giai đoạn này với bộ tham số hiện tại.")
        else:
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Tổng Số Lệnh", f"{len(trades_table)}")
            win_count = len(trades_table[trades_table['Lợi Nhuận (%)'] > 0])
            col_m2.metric("Số Lệnh Thắng", f"{win_count} ({win_count/len(trades_table)*100:.1f}%)")
            col_m3.metric("Lợi Nhuận TB / Lệnh", f"{trades_table['Lợi Nhuận (%)'].mean():.2f}%")
            col_m4.metric("Thời Gian Giữ TB", f"{trades_table['Thời Gian Giữ (phiên)'].mean():.1f} phiên")

            display_trades = trades_table.copy()
            display_trades['Kết Quả'] = np.where(display_trades['Lợi Nhuận (%)'] > 0, "🟢 Thắng", "🔴 Thua")
            display_trades['Ngày Mua'] = pd.to_datetime(display_trades['Ngày Mua']).dt.strftime('%d/%m/%Y')
            display_trades['Ngày Bán'] = pd.to_datetime(display_trades['Ngày Bán']).dt.strftime('%d/%m/%Y')

            cols_order = ['STT', 'Kết Quả', 'Ngày Mua', 'Giá Mua', 'Ngày Bán', 'Giá Bán',
                          'Số CP', 'Lợi Nhuận (%)', 'Lợi Nhuận (VNĐ)', 'Thời Gian Giữ (phiên)', 'Lý Do Bán']
            display_trades = display_trades[[c for c in cols_order if c in display_trades.columns]]

            st.dataframe(
                display_trades,
                column_config={
                    "STT": st.column_config.NumberColumn("STT", width="small", format="%d"),
                    "Kết Quả": st.column_config.TextColumn("Kết Quả", width="small"),
                    "Ngày Mua": st.column_config.TextColumn("Ngày Mua"),
                    "Giá Mua": st.column_config.NumberColumn("Giá Mua", format="%.1f"),
                    "Ngày Bán": st.column_config.TextColumn("Ngày Bán"),
                    "Giá Bán": st.column_config.NumberColumn("Giá Bán", format="%.1f"),
                    "Số CP": st.column_config.NumberColumn("Số CP", format="%d"),
                    "Lợi Nhuận (%)": st.column_config.NumberColumn("Lợi Nhuận (%)", format="%.2f%%"),
                    "Lợi Nhuận (VNĐ)": st.column_config.NumberColumn("Lợi Nhuận (VNĐ)", format="%d VNĐ"),
                    "Thời Gian Giữ (phiên)": st.column_config.NumberColumn("Số Phiên Giữ", format="%d"),
                    "Lý Do Bán": st.column_config.TextColumn("Lý Do Bán", width="medium")
                },
                use_container_width=True,
                hide_index=True
            )

            # Xuất file CSV
            csv_buffer = io.StringIO()
            trades_table.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 Tải Về Nhật Ký Giao Dịch (CSV)",
                data=csv_buffer.getvalue(),
                file_name=f"Trade_Log_SMA_OBV_{log_strat}_{log_phase}.csv",
                mime="text/csv"
            )


    # =========================================================================
    # TAB 4: TỰ ĐỘNG TỐI ƯU HÓA (CHỈ TRÊN TẬP TRAIN)
    # =========================================================================
    with tab4:
        st.subheader("⚡ Tự Động Tối Ưu Hóa Tham Số (Chỉ Trên Tập Train)")
        st.markdown(r"""
        *Theo nguyên tắc khoa học trong Notebook Nhóm 5:*
        1. **Chỉ dùng tập Train** để tối ưu hóa tham số (tránh Data Snooping).
        2. Sau khi tìm được bộ tham số tốt nhất, mang nguyên bộ tham số sang **Tập Test** để kiểm định mù độc lập.
        """)

        opt_target = st.radio("Chọn chiến lược muốn tối ưu hóa:", ["Tối ưu SMA (ma_short, ma_long)", "Tối ưu OBV (obv_window)"], horizontal=True)

        if "SMA" in opt_target:
            col_o1, col_o2 = st.columns(2)
            with col_o1:
                s_min = st.number_input("SMA Ngắn (Từ):", min_value=10, max_value=100, value=25)
                s_max = st.number_input("SMA Ngắn (Đến):", min_value=50, max_value=250, value=150)
                s_step = st.number_input("Bước nhảy Ngắn:", min_value=5, max_value=50, value=25)
            with col_o2:
                l_min = st.number_input("SMA Dài (Từ):", min_value=100, max_value=300, value=200)
                l_max = st.number_input("SMA Dài (Đến):", min_value=200, max_value=500, value=350)
                l_step = st.number_input("Bước nhảy Dài:", min_value=5, max_value=50, value=25)

            if st.button("🚀 Quét Không Gian Tham Số SMA (Train)", type="primary"):
                s_range = range(int(s_min), int(s_max) + 1, int(s_step))
                l_range = range(int(l_min), int(l_max) + 1, int(l_step))

                with st.spinner("Đang chạy mô phỏng tối ưu hóa SMA trên Train..."):
                    best_sma_res, df_sma_res = optimize_sma(train_df, s_range, l_range, init_cash, comm_pct)

                if len(df_sma_res) > 0:
                    st.success(f"🎉 Đã tìm ra tham số SMA tối ưu: Short = {best_sma_res['ma_short']}, Long = {best_sma_res['ma_long']}")
                    st.markdown("#### 🔝 Top 10 Bộ Tham Số SMA Tốt Nhất:")
                    top_sma = df_sma_res.sort_values(by='Return [%]', ascending=False).head(10)
                    st.dataframe(
                        top_sma,
                        column_config={
                            "ma_short": st.column_config.NumberColumn("SMA Ngắn", format="%d"),
                            "ma_long": st.column_config.NumberColumn("SMA Dài", format="%d"),
                            "Return [%]": st.column_config.NumberColumn("Return [%]", format="%.2f%%"),
                            "Sharpe Ratio": st.column_config.NumberColumn("Sharpe", format="%.2f"),
                            "Max Drawdown [%]": st.column_config.NumberColumn("Max DD [%]", format="%.2f%%"),
                            "Trades": st.column_config.NumberColumn("Số Lệnh", format="%d")
                        },
                        use_container_width=True, hide_index=True
                    )
                    if st.button("👉 Áp Dụng Bộ Tham Số SMA Này Vào Hệ Thống"):
                        st.session_state['quick_short'] = best_sma_res['ma_short']
                        st.session_state['quick_long'] = best_sma_res['ma_long']
                        st.rerun()
        else:
            col_w1, col_w2 = st.columns(2)
            with col_w1:
                w_min = st.number_input("OBV Window (Từ):", min_value=2, max_value=20, value=5)
                w_max = st.number_input("OBV Window (Đến):", min_value=20, max_value=100, value=50)
            with col_w2:
                w_step = st.number_input("Bước nhảy Window:", min_value=1, max_value=10, value=5)

            if st.button("🚀 Quét Không Gian Tham Số OBV (Train)", type="primary"):
                w_range = range(int(w_min), int(w_max) + 1, int(w_step))

                with st.spinner("Đang chạy mô phỏng tối ưu hóa OBV trên Train..."):
                    best_obv_res, df_obv_res = optimize_obv(train_df, w_range, init_cash, comm_pct)

                if len(df_obv_res) > 0:
                    st.success(f"🎉 Đã tìm ra tham số OBV tối ưu: obv_window = {best_obv_res['obv_window']}")
                    st.markdown("#### 🔝 Top 10 Tham Số OBV Tốt Nhất:")
                    top_obv = df_obv_res.sort_values(by='Return [%]', ascending=False).head(10)
                    st.dataframe(
                        top_obv,
                        column_config={
                            "obv_window": st.column_config.NumberColumn("OBV Window", format="%d"),
                            "Return [%]": st.column_config.NumberColumn("Return [%]", format="%.2f%%"),
                            "Sharpe Ratio": st.column_config.NumberColumn("Sharpe", format="%.2f"),
                            "Max Drawdown [%]": st.column_config.NumberColumn("Max DD [%]", format="%.2f%%"),
                            "Trades": st.column_config.NumberColumn("Số Lệnh", format="%d")
                        },
                        use_container_width=True, hide_index=True
                    )
                    if st.button("👉 Áp Dụng Tham Số OBV Này Vào Hệ Thống"):
                        st.session_state['quick_obv'] = best_obv_res['obv_window']
                        st.rerun()


    # =========================================================================
    # TAB 5: LÝ THUYẾT & KẾT LUẬN
    # =========================================================================
    with tab5:
        st.subheader("📚 Cơ Sở Khoa Học Của Chiến Lược Kết Hợp SMA & OBV")
        st.markdown(r"""
        ### 1. Ý Nghĩa Kỹ Thuật Của Từng Chỉ Báo
        * **Chiến lược Giao Cắt Hai Đường Trung Bình Động (SMA Crossover):**
          Sử dụng 2 đường trung bình động: SMA ngắn hạn ($N_{short}$) và SMA dài hạn ($N_{long}$).
          $$SMA_t = \frac{1}{N} \sum_{i=0}^{N-1} Close_{t-i}$$
          * **Tín hiệu Mua (Golden Cross):** $SMA_{short}$ cắt lên trên $SMA_{long}$ $\rightarrow$ Báo hiệu xu hướng tăng trung dài hạn bắt đầu.
          * **Tín hiệu Bán (Death Cross):** $SMA_{short}$ cắt xuống dưới $SMA_{long}$ $\rightarrow$ Báo hiệu xu hướng giảm bắt đầu, thoát vị thế.

        * **Chiến lược Khối Lượng Cân Bằng (OBV Crossover với OBV MA):**
          $$OBV_t = OBV_{t-1} + \begin{cases} Volume_t & \text{nếu } Close_t > Close_{t-1} \\ 0 & \text{nếu } Close_t = Close_{t-1} \\ -Volume_t & \text{nếu } Close_t < Close_{t-1} \end{cases}$$
          $$OBV\_MA_t = \frac{1}{k} \sum_{i=0}^{k-1} OBV_{t-i}$$
          * **Tín hiệu Mua:** $OBV$ cắt lên trên đường trung bình động của nó $OBV\_MA$ $\rightarrow$ Áp lực tích lũy mua chủ động gia tăng mạnh mẽ.
          * **Tín hiệu Bán:** $OBV$ cắt xuống dưới $OBV\_MA$ $\rightarrow$ Áp lực xả hàng chiếm ưu thế.

        ---

        ### 2. Nguyên Lý Kết Hợp Quy Tắc AND
        * **Quy tắc mua:** $BUY = (SMA\_BUY) \land (OBV\_BUY)$
        * **Quy tắc bán:** $SELL = (SMA\_SELL) \land (OBV\_SELL)$
        * **Ưu điểm:** Lọc bớt các tín hiệu nhiễu, hạn chế tối đa số lần giao dịch sai lầm.
        * **Nhược điểm cần lưu ý:** Vì điều kiện đồng thời quá khắt khe, số lượng lệnh phát sinh có thể rất ít (như thấy trên tập Test có thể bằng 0 lệnh nếu không có sự trùng pha chính xác trong cùng một phiên). Do đó, nhà đầu tư có thể xem xét chế độ *CONFIRMED* (kết hợp trạng thái xu hướng) để tăng cơ hội vào lệnh.
        """)


if __name__ == '__main__':
    main()
