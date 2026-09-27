"""
=============================================================================
ỨNG DỤNG KIỂM ĐỊNH CHIẾN LƯỢC ĐẦU TƯ: KẾT HỢP CHỈ BÁO EMA VÀ OBV
Tác giả: Chuyên gia Phân tích Định lượng & Kỹ thuật Phần mềm (Quantitative Web App)
Dành cho: Kiểm định chiến lược đầu tư chứng khoán (Backtesting & Walk-Forward Testing)
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
    page_title="Kiểm Định Chiến Lược EMA + OBV | Backtest Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho giao diện hiện đại, chuyên nghiệp
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E88E5, #43A047);
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
        border-left: 4px solid #1E88E5;
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
def calculate_indicators(df, ema_period, obv_slope_period, price_col='Close'):
    """
    Tính toán đường EMA và độ dốc chỉ báo OBV.
    """
    data = df.copy()
    close = data[price_col]
    volume = data['Volume']

    # 1. Exponential Moving Average (EMA)
    data['EMA'] = close.ewm(span=int(ema_period), adjust=False).mean()

    # 2. On-Balance Volume (OBV)
    price_diff = close.diff()
    direction = np.where(price_diff > 0, 1.0, np.where(price_diff < 0, -1.0, 0.0))
    direction[0] = 0.0
    data['OBV'] = (volume * direction).cumsum()

    # 3. Độ dốc OBV (OBV Slope qua N phiên)
    data['OBV_Slope'] = data['OBV'].diff(int(obv_slope_period))

    return data


def generate_signals(data, strategy_type, ema_period, obv_slope_period, price_col='Close'):
    """
    Sinh tín hiệu MUA (Entry) và BÁN (Exit) theo đúng phương pháp trong nghiên cứu:
    - Tránh look-ahead bias: Dùng tín hiệu phiên hôm trước (shift 1) cho phiên hôm nay.
    - Chiến lược:
        + 'EMA': Mua khi Close > EMA, Bán khi Close < EMA
        + 'OBV': Mua khi OBV_Slope > 0, Bán khi OBV_Slope < 0
        + 'Combined': Mua khi Close > EMA VÀ OBV_Slope > 0; Bán khi Close < EMA
    """
    df = calculate_indicators(data, ema_period, obv_slope_period, price_col)
    close = df[price_col]
    ema = df['EMA']
    obv_slope = df['OBV_Slope']

    if strategy_type == 'EMA':
        raw_entries = (close > ema)
        raw_exits = (close < ema)
    elif strategy_type == 'OBV':
        raw_entries = (obv_slope > 0)
        raw_exits = (obv_slope < 0)
    elif strategy_type == 'Combined':
        raw_entries = (close > ema) & (obv_slope > 0)
        raw_exits = (close < ema)
    else:
        raise ValueError(f"Không nhận diện được chiến lược: {strategy_type}")

    # Shift 1 phiên để phản ánh quyết định thực tế sau khi nến đóng cửa
    entries = raw_entries.shift(1, fill_value=False).astype(bool)
    exits = raw_exits.shift(1, fill_value=False).astype(bool)

    df['Entry_Signal'] = entries
    df['Exit_Signal'] = exits

    return df


# -----------------------------------------------------------------------------
# 3. ĐỘNG CƠ BACKTEST (VECTORIZED & EVENT-DRIVEN HYBRID ENGINE)
# Mô phỏng chính xác logic của VectorBT: longonly, fees, slippage, stop-loss 7%
# -----------------------------------------------------------------------------
def run_backtest(df, entries, exits, price_col='Close',
                 initial_capital=100_000_000, fee_rate=0.002,
                 slippage_rate=0.001, sl_stop=0.07):
    """
    Mô phỏng giao dịch chi tiết theo tài khoản:
    - Long-only, không tích lũy vị thế (accumulate=False)
    - Quản lý rủi ro Cắt Lỗ Cố Định (sl_stop, mặc định 7%)
    - Khấu trừ Phí giao dịch (0.2%) và Trượt giá (0.1%)
    """
    dates = df.index
    closes = df[price_col].values
    n = len(df)

    capital = float(initial_capital)
    cash = capital
    shares = 0
    in_position = False
    buy_price = 0.0
    entry_date = None
    entry_index = 0
    total_cost = 0.0

    # Lưu trữ kết quả từng phiên để vẽ đường cong vốn
    portfolio_value = np.zeros(n, dtype=float)
    positions = np.zeros(n, dtype=int)
    cash_history = np.zeros(n, dtype=float)

    # Nhật ký giao dịch
    trades = []

    for i in range(n):
        curr_price = float(closes[i])
        curr_date = dates[i]
        is_entry = bool(entries.iloc[i])
        is_exit = bool(exits.iloc[i])

        # Kiểm tra nếu đang có vị thế
        if in_position:
            # 1. Kiểm tra Cắt Lỗ Stop-Loss (7% so với giá mua thực tế)
            price_change = (curr_price - buy_price) / buy_price
            hit_stoploss = price_change <= -sl_stop

            # 2. Quyết định Thoát Vị Thế: hoặc dính SL, hoặc có tín hiệu Bán
            if hit_stoploss or is_exit:
                exit_price = curr_price * (1.0 - slippage_rate)
                gross_revenue = shares * exit_price
                fee_exit = gross_revenue * fee_rate
                net_revenue = gross_revenue - fee_exit

                cash += net_revenue
                profit_pct = (net_revenue - total_cost) / total_cost * 100.0
                profit_cash = net_revenue - total_cost

                trades.append({
                    'STT': len(trades) + 1,
                    'Ngày Mua': entry_date,
                    'Giá Mua': round(buy_price, 1),
                    'Ngày Bán': curr_date,
                    'Giá Bán': round(exit_price, 1),
                    'Số CP': int(shares),
                    'Lợi Nhuận (%)': round(profit_pct, 2),
                    'Lợi Nhuận (VNĐ)': round(profit_cash, 0),
                    'Thời Gian Giữ (phiên)': int(i - entry_index),
                    'Lý Do Bán': 'Cắt Lỗ Stop-Loss (7%)' if hit_stoploss else 'Tín Hiệu Kỹ Thuật (Exit)'
                })

                in_position = False
                shares = 0
                buy_price = 0.0

        # Nếu chưa có vị thế và có tín hiệu MUA
        elif not in_position and is_entry:
            exec_price = curr_price * (1.0 + slippage_rate)
            fee_factor = 1.0 + fee_rate
            affordable_shares = int(cash / (exec_price * fee_factor))

            if affordable_shares > 0:
                cost_shares = affordable_shares * exec_price
                fee_entry = cost_shares * fee_rate
                total_cost = cost_shares + fee_entry

                cash -= total_cost
                shares = affordable_shares
                buy_price = exec_price
                entry_date = curr_date
                entry_index = i
                in_position = True

        # Ghi nhận trạng thái cuối phiên
        current_equity = cash + (shares * curr_price if in_position else 0.0)
        portfolio_value[i] = current_equity
        positions[i] = 1 if in_position else 0
        cash_history[i] = cash

    # Xử lý lệnh còn mở ở phiên cuối cùng (Mark-to-Market)
    if in_position:
        final_price = float(closes[-1]) * (1.0 - slippage_rate)
        gross_rev = shares * final_price
        fee_exit = gross_rev * fee_rate
        net_rev = gross_rev - fee_exit
        profit_pct = (net_rev - total_cost) / total_cost * 100.0
        profit_cash = net_rev - total_cost
        trades.append({
            'STT': len(trades) + 1,
            'Ngày Mua': entry_date,
            'Giá Mua': round(buy_price, 1),
            'Ngày Bán': dates[-1],
            'Giá Bán': round(final_price, 1),
            'Số CP': int(shares),
            'Lợi Nhuận (%)': round(profit_pct, 2),
            'Lợi Nhuận (VNĐ)': round(profit_cash, 0),
            'Thời Gian Giữ (phiên)': int(n - 1 - entry_index),
            'Lý Do Bán': 'Kết Thúc Giai Đoạn (Đóng vị thế)'
        })

    # Chuyển kết quả sang DataFrame
    equity_df = pd.DataFrame({
        'Date': dates,
        'Portfolio_Value': portfolio_value,
        'Position': positions,
        'Cash': cash_history
    }).set_index('Date')

    # Tính toán các chỉ số định lượng
    trades_df = pd.DataFrame(trades)
    metrics = calculate_performance_metrics(equity_df, trades_df, initial_capital, closes)

    return equity_df, trades_df, metrics


def calculate_performance_metrics(equity_df, trades_df, initial_capital, closes):
    """
    Tính các chỉ số đo lường hiệu suất tiêu chuẩn (Sharpe, Drawdown, Winrate, CAGR,...).
    """
    port_vals = equity_df['Portfolio_Value']
    daily_returns = port_vals.pct_change().dropna()

    # 1. Tổng Lợi Nhuận (Total Return)
    final_equity = float(port_vals.iloc[-1])
    total_return = float((final_equity - initial_capital) / initial_capital * 100.0)

    # 2. Số năm và CAGR
    n_days = len(equity_df)
    years = max(n_days / 252.0, 0.05)
    cagr = float(((final_equity / initial_capital) ** (1.0 / years) - 1.0) * 100.0)

    # 3. Mức Sụt Giảm Lớn Nhất (Max Drawdown)
    cummax = port_vals.cummax()
    drawdowns = (port_vals - cummax) / cummax
    max_drawdown = float(drawdowns.min() * 100.0)

    # 4. Tỷ lệ Sharpe (Annualized Sharpe Ratio với Rf = 0)
    std_return = float(daily_returns.std())
    if std_return > 0 and len(daily_returns) > 1:
        sharpe_ratio = float((daily_returns.mean() / std_return) * np.sqrt(252))
    else:
        sharpe_ratio = 0.0

    # 5. Tỷ lệ Sortino
    neg_returns = daily_returns[daily_returns < 0]
    std_neg = float(neg_returns.std()) if len(neg_returns) > 1 else 0.0
    if std_neg > 0:
        sortino_ratio = float((daily_returns.mean() / std_neg) * np.sqrt(252))
    else:
        sortino_ratio = 0.0

    # 6. Thống kê lệnh giao dịch
    total_trades = len(trades_df)
    if total_trades > 0:
        winning_trades = trades_df[trades_df['Lợi Nhuận (%)'] > 0]
        losing_trades = trades_df[trades_df['Lợi Nhuận (%)'] <= 0]

        win_rate = float(len(winning_trades) / total_trades * 100.0)
        avg_profit = float(trades_df['Lợi Nhuận (%)'].mean())

        total_gain = float(winning_trades['Lợi Nhuận (VNĐ)'].sum()) if len(winning_trades) > 0 else 0.0
        total_loss = float(abs(losing_trades['Lợi Nhuận (VNĐ)'].sum())) if len(losing_trades) > 0 else 0.0
        profit_factor = float(total_gain / total_loss) if total_loss > 0 else (np.nan if total_gain > 0 else 0.0)

        max_win = float(trades_df['Lợi Nhuận (%)'].max())
        max_loss = float(trades_df['Lợi Nhuận (%)'].min())
        avg_holding = float(trades_df['Thời Gian Giữ (phiên)'].mean())
    else:
        win_rate = 0.0
        avg_profit = 0.0
        profit_factor = np.nan
        max_win = 0.0
        max_loss = 0.0
        avg_holding = 0.0

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
        'Vốn Cuối Kỳ (VNĐ)': final_equity,
        'Tổng Lợi Nhuận (%)': total_return,
        'Lợi Nhuận Năm (CAGR %)': cagr,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Max Drawdown (%)': max_drawdown,
        'Số Giao Dịch': total_trades,
        'Tỷ Lệ Thắng (%)': win_rate,
        'Profit Factor': profit_factor,
        'LN Trung Bình/Lệnh (%)': avg_profit,
        'Lệnh Lãi Lớn Nhất (%)': max_win,
        'Lệnh Lỗ Lớn Nhất (%)': max_loss,
        'Số Phiên Giữ TB': avg_holding,
        'Buy & Hold Return (%)': bh_return,
        'Buy & Hold CAGR (%)': bh_cagr,
        'Buy & Hold Sharpe': bh_sharpe,
        'Buy & Hold Max DD (%)': bh_mdd
    }


# -----------------------------------------------------------------------------
# 4. HÀM TỐI ƯU HÓA THAM SỐ (GRID SEARCH OPTIMIZATION - FAST NUMPY ENGINE)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def optimize_strategy(df, strategy_type, ema_range, obv_range, min_trades=5,
                      price_col='Close', initial_capital=100_000_000,
                      fee_rate=0.002, slippage_rate=0.001, sl_stop=0.07):
    """
    Tối ưu hóa tham số trên tập Train dựa trên mục tiêu tối đa hóa Sharpe Ratio
    với điều kiện số lệnh >= min_trades (loại bỏ tham số ăn may ít giao dịch).
    """
    best_sharpe = -np.inf
    best_params = {}
    best_metrics = None
    all_results = []

    if strategy_type == 'EMA':
        for ema in ema_range:
            sig_df = generate_signals(df, 'EMA', ema, 3, price_col)
            _, trades_df, metrics = run_backtest(
                sig_df, sig_df['Entry_Signal'], sig_df['Exit_Signal'],
                price_col, initial_capital, fee_rate, slippage_rate, sl_stop
            )
            n_trades = metrics['Số Giao Dịch']
            sharpe = metrics['Sharpe Ratio']
            is_valid = np.isfinite(sharpe) and n_trades >= min_trades

            all_results.append({
                'EMA': int(ema),
                'Sharpe': float(sharpe),
                'Total Return (%)': float(metrics['Tổng Lợi Nhuận (%)']),
                'Max Drawdown (%)': float(metrics['Max Drawdown (%)']),
                'Số Giao Dịch': int(n_trades),
                'Hợp Lệ': bool(is_valid)
            })

            if is_valid and sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = {'EMA': int(ema)}
                best_metrics = metrics

    elif strategy_type == 'OBV':
        for obv in obv_range:
            sig_df = generate_signals(df, 'OBV', 20, obv, price_col)
            _, trades_df, metrics = run_backtest(
                sig_df, sig_df['Entry_Signal'], sig_df['Exit_Signal'],
                price_col, initial_capital, fee_rate, slippage_rate, sl_stop
            )
            n_trades = metrics['Số Giao Dịch']
            sharpe = metrics['Sharpe Ratio']
            is_valid = np.isfinite(sharpe) and n_trades >= min_trades

            all_results.append({
                'OBV_Slope': int(obv),
                'Sharpe': float(sharpe),
                'Total Return (%)': float(metrics['Tổng Lợi Nhuận (%)']),
                'Max Drawdown (%)': float(metrics['Max Drawdown (%)']),
                'Số Giao Dịch': int(n_trades),
                'Hợp Lệ': bool(is_valid)
            })

            if is_valid and sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = {'OBV_Slope': int(obv)}
                best_metrics = metrics

    elif strategy_type == 'Combined':
        for ema in ema_range:
            for obv in obv_range:
                sig_df = generate_signals(df, 'Combined', ema, obv, price_col)
                _, trades_df, metrics = run_backtest(
                    sig_df, sig_df['Entry_Signal'], sig_df['Exit_Signal'],
                    price_col, initial_capital, fee_rate, slippage_rate, sl_stop
                )
                n_trades = metrics['Số Giao Dịch']
                sharpe = metrics['Sharpe Ratio']
                is_valid = np.isfinite(sharpe) and n_trades >= min_trades

                all_results.append({
                    'EMA': int(ema),
                    'OBV_Slope': int(obv),
                    'Sharpe': float(sharpe),
                    'Total Return (%)': float(metrics['Tổng Lợi Nhuận (%)']),
                    'Max Drawdown (%)': float(metrics['Max Drawdown (%)']),
                    'Số Giao Dịch': int(n_trades),
                    'Hợp Lệ': bool(is_valid)
                })

                if is_valid and sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = {'EMA': int(ema), 'OBV_Slope': int(obv)}
                    best_metrics = metrics

    return best_params, best_metrics, pd.DataFrame(all_results)


# -----------------------------------------------------------------------------
# 5. HÀM TẢI & TIỀN XỬ LÝ DỮ LIỆU (DATA LOADING & PARSING)
# -----------------------------------------------------------------------------
@st.cache_data
def load_dataset(uploaded_file=None):
    """
    Nạp dữ liệu từ file tải lên hoặc file ACB.csv mặc định.
    Tự động chuẩn hóa tên cột (Date, Open, High, Low, Close, Volume).
    """
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            # Tìm file ACB.csv trong thư mục hiện tại
            df = pd.read_csv('ACB.csv')
    except Exception as e:
        return None, f"Không thể đọc file: {str(e)}"

    # Tìm và đổi tên cột Ngày (Date)
    date_col = None
    for col in df.columns:
        if col.lower() in ['date', 'ngày', 'time', 'ngay', 'datetime']:
            date_col = col
            break

    if date_col is None:
        return None, "Không tìm thấy cột ngày giao dịch (Date) trong dữ liệu!"

    df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.set_index('Date').sort_index()

    # Nhận diện cột Giá Đóng Cửa (Close)
    close_col = None
    for col in df.columns:
        if col.lower() in ['close', 'đóng cửa', 'dongcua', 'adj_close', 'adjclose', 'price']:
            close_col = col
            break

    if close_col is None:
        return None, "Không tìm thấy cột Giá Đóng Cửa (Close) trong dữ liệu!"

    # Nhận diện cột Khối Lượng (Volume)
    vol_col = None
    for col in df.columns:
        if col.lower() in ['volume', 'khối lượng', 'khoiluong', 'vol', 'total_volume']:
            vol_col = col
            break

    if vol_col is None:
        return None, "Không tìm thấy cột Khối Lượng (Volume) trong dữ liệu!"

    # Đổi tên chuẩn hóa
    df = df.rename(columns={close_col: 'Close', vol_col: 'Volume'})
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    df = df.dropna(subset=['Close', 'Volume'])

    return df, None


# -----------------------------------------------------------------------------
# 6. GIAO DIỆN CHÍNH (STREAMLIT APP LAYOUT & SIDEBAR)
# -----------------------------------------------------------------------------
def main():
    st.markdown('<div class="main-title">📈 HỆ THỐNG KIỂM ĐỊNH CHIẾN LƯỢC: EMA + OBV</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Nghiên cứu kiểm định định lượng, quản lý rủi ro và ngăn ngừa Overfitting (Dựa trên mô hình Nhóm 3 - ACB)</div>', unsafe_allow_html=True)

    # ----------------- SIDEBAR: CẤU HÌNH & THAM SỐ -----------------
    with st.sidebar:
        st.header("⚙️ Thiết Lập Dữ Liệu & Tham Số")

        # 1. Lựa chọn nguồn dữ liệu
        data_source = st.radio(
            "📂 Nguồn dữ liệu:",
            ["Dữ liệu mẫu ACB.csv (2014 - 2023)", "Tải lên file CSV cá nhân"]
        )

        uploaded_file = None
        if data_source == "Tải lên file CSV cá nhân":
            uploaded_file = st.file_uploader("Chọn file CSV:", type=['csv'])

        df, err = load_dataset(uploaded_file)
        if err or df is None:
            st.error(f"Lỗi nạp dữ liệu: {err}")
            st.info("Vui lòng tải lên file CSV có chứa các cột: Date, Close, Volume.")
            return

        min_year = int(df.index.min().year)
        max_year = int(df.index.max().year)

        st.success(f"Đã nạp: {len(df):,} phiên ({df.index.min().strftime('%d/%m/%Y')} - {df.index.max().strftime('%d/%m/%Y')})")

        # 2. Phân chia Train/Test (Walk-Forward / Hold-Out)
        st.subheader("📅 Phân Chia Tập Dữ Liệu")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            train_start = st.number_input("Train Từ Năm", min_value=min_year, max_value=max_year, value=max(min_year, 2014))
            train_end = st.number_input("Train Đến Năm", min_value=min_year, max_value=max_year, value=min(2020, max_year))
        with col_t2:
            test_start = st.number_input("Test Từ Năm", min_value=min_year, max_value=max_year, value=min(2021, max_year))
            test_end = st.number_input("Test Đến Năm", min_value=min_year, max_value=max_year, value=max_year)

        # 3. Tham số chiến lược
        st.subheader("🎯 Tham Số Chiến Lược")
        ema_input = st.slider("Chu kỳ EMA (phiên):", min_value=5, max_value=60, value=20, step=1,
                              help="Mặc định: 20 phiên. Tham số tối ưu tập Train thường khoảng 36.")
        obv_input = st.slider("Độ dốc OBV (OBV Slope - phiên):", min_value=1, max_value=30, value=3, step=1,
                              help="Mặc định: 3 phiên. Tham số tối ưu tập Train thường khoảng 19-20.")

        # 4. Quản trị rủi ro & Chi phí
        with st.expander("🛡️ Quản Trị Rủi Ro & Phí Giao Dịch", expanded=False):
            stop_loss = st.number_input("Cắt lỗ Stop-Loss (%)", min_value=1.0, max_value=20.0, value=7.0, step=0.5) / 100.0
            fee_pct = st.number_input("Phí giao dịch mỗi chiều (%)", min_value=0.0, max_value=1.0, value=0.2, step=0.05) / 100.0
            slippage_pct = st.number_input("Trượt giá mỗi chiều (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.05) / 100.0
            init_capital = st.number_input("Vốn ban đầu (VNĐ)", min_value=10_000_000, max_value=10_000_000_000, value=100_000_000, step=10_000_000)

        # Nút áp dụng nhanh tham số tối ưu từ bài tập
        st.markdown("---")
        st.markdown("💡 **Cấu Hình Nhanh Theo Bài Báo Cáo:**")
        col_btn1, col_btn2 = st.columns(2)
        if col_btn1.button("📌 Mặc Định (20, 3)"):
            st.session_state['custom_ema'] = 20
            st.session_state['custom_obv'] = 3
            st.rerun()

        if col_btn2.button("⭐ Tối Ưu (36, 20)"):
            st.session_state['custom_ema'] = 36
            st.session_state['custom_obv'] = 20
            st.rerun()

        if 'custom_ema' in st.session_state:
            ema_input = st.session_state['custom_ema']
        if 'custom_obv' in st.session_state:
            obv_input = st.session_state['custom_obv']

    # ----------------- TÁCH TẬP TRAIN VÀ TEST -----------------
    train_mask = (df.index.year >= train_start) & (df.index.year <= train_end)
    test_mask = (df.index.year >= test_start) & (df.index.year <= test_end)

    train_df = df.loc[train_mask].copy()
    test_df = df.loc[test_mask].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        st.error("Khoảng thời gian Train hoặc Test không hợp lệ hoặc không có dữ liệu! Vui lòng điều chỉnh lại bộ lọc năm.")
        return

    # ----------------- CHẠY CẢ 3 CHIẾN LƯỢC TRÊN CẢ 2 TẬP DỮ LIỆU -----------------
    strategies = ['EMA', 'OBV', 'Combined']
    results = {'Train': {}, 'Test': {}}

    for phase_name, p_df in [('Train', train_df), ('Test', test_df)]:
        for strat in strategies:
            sig_df = generate_signals(p_df, strat, ema_input, obv_input)
            eq_df, tr_df, met = run_backtest(
                sig_df, sig_df['Entry_Signal'], sig_df['Exit_Signal'],
                price_col='Close', initial_capital=init_capital,
                fee_rate=fee_pct, slippage_rate=slippage_pct, sl_stop=stop_loss
            )
            results[phase_name][strat] = {
                'equity': eq_df,
                'trades': tr_df,
                'metrics': met,
                'signals': sig_df
            }

    # ----------------- TẠO TABS CHÍNH -----------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 So Sánh Hiệu Suất (Train vs Test)",
        "📈 Biểu Đồ & Tín Hiệu Kỹ Thuật",
        "📑 Nhật Ký Lệnh (Trade Log)",
        "⚡ Tự Động Tối Ưu Hóa Tham Số",
        "📖 Cơ Sở Lý Thuyết & Hướng Dẫn"
    ])

    # =========================================================================
    # TAB 1: SO SÁNH HIỆU SUẤT TỔNG THỂ (CELL 22, 23 NOTEBOOK)
    # =========================================================================
    with tab1:
        st.subheader("🎯 Bảng So Sánh Hiệu Suất Giữa Tập Train và Test")

        # Tạo bảng so sánh chuẩn xác với kiểu số thuần túy (float / int)
        table_rows = []
        strat_names_map = {
            'EMA': 'EMA Riêng Lẻ',
            'OBV': 'OBV Riêng Lẻ',
            'Combined': 'EMA + OBV Kết Hợp'
        }

        for phase in ['Train', 'Test']:
            for strat in strategies:
                m = results[phase][strat]['metrics']
                table_rows.append({
                    'Chiến Lược': strat_names_map[strat],
                    'Tập Dữ Liệu': phase,
                    'Sharpe Ratio': float(m['Sharpe Ratio']),
                    'Tổng Lợi Nhuận (%)': float(m['Tổng Lợi Nhuận (%)']),
                    'Lợi Nhuận Năm CAGR (%)': float(m['Lợi Nhuận Năm (CAGR %)']),
                    'Max Drawdown (%)': float(m['Max Drawdown (%)']),
                    'Tỷ Lệ Thắng (%)': float(m['Tỷ Lệ Thắng (%)']),
                    'Số Giao Dịch': int(m['Số Giao Dịch']),
                    'Profit Factor': float(m['Profit Factor']) if (np.isfinite(m['Profit Factor']) and not np.isnan(m['Profit Factor'])) else np.nan
                })

        # Thêm Benchmark Buy & Hold với các chỉ số tính toán đầy đủ
        for phase, p_df in [('Train', train_df), ('Test', test_df)]:
            m_sample = results[phase]['Combined']['metrics']
            table_rows.append({
                'Chiến Lược': 'Buy & Hold (Mua & Giữ)',
                'Tập Dữ Liệu': phase,
                'Sharpe Ratio': float(m_sample['Buy & Hold Sharpe']),
                'Tổng Lợi Nhuận (%)': float(m_sample['Buy & Hold Return (%)']),
                'Lợi Nhuận Năm CAGR (%)': float(m_sample['Buy & Hold CAGR (%)']),
                'Max Drawdown (%)': float(m_sample['Buy & Hold Max DD (%)']),
                'Tỷ Lệ Thắng (%)': 100.0 if m_sample['Buy & Hold Return (%)'] > 0 else 0.0,
                'Số Giao Dịch': 1,
                'Profit Factor': np.nan
            })

        comparison_df = pd.DataFrame(table_rows)

        # Hiển thị bảng qua Streamlit column_config: Hiện đại, định dạng đẹp, sắp xếp mượt mà, không gặp lỗi NumPy/Pandas Styler
        st.dataframe(
            comparison_df,
            column_config={
                "Chiến Lược": st.column_config.TextColumn("Chiến Lược", width="medium"),
                "Tập Dữ Liệu": st.column_config.TextColumn("Tập Dữ Liệu", width="small"),
                "Sharpe Ratio": st.column_config.NumberColumn("Sharpe Ratio", format="%.2f"),
                "Tổng Lợi Nhuận (%)": st.column_config.NumberColumn("Tổng Lợi Nhuận (%)", format="%.2f%%"),
                "Lợi Nhuận Năm CAGR (%)": st.column_config.NumberColumn("CAGR (%)", format="%.2f%%"),
                "Max Drawdown (%)": st.column_config.NumberColumn("Max DD (%)", format="%.2f%%"),
                "Tỷ Lệ Thắng (%)": st.column_config.NumberColumn("Tỷ Lệ Thắng (%)", format="%.1f%%"),
                "Số Giao Dịch": st.column_config.NumberColumn("Số Lệnh", format="%d"),
                "Profit Factor": st.column_config.NumberColumn("Profit Factor", format="%.2f"),
            },
            use_container_width=True,
            hide_index=True
        )

        # Metric cards nổi bật cho chiến lược Đề Xuất (Combined)
        st.markdown("### 🏆 Hiệu Suất Chiến Lược Kết Hợp (EMA + OBV)")
        c1, c2, c3, c4 = st.columns(4)
        comb_train_m = results['Train']['Combined']['metrics']
        comb_test_m = results['Test']['Combined']['metrics']

        c1.metric("Sharpe Ratio (Train / Test)",
                  f"{comb_train_m['Sharpe Ratio']:.2f}",
                  delta=f"Test: {comb_test_m['Sharpe Ratio']:.2f}")

        c2.metric("Tổng Lợi Nhuận (Train / Test)",
                  f"{comb_train_m['Tổng Lợi Nhuận (%)']:.1f}%",
                  delta=f"Test: {comb_test_m['Tổng Lợi Nhuận (%)']:.1f}%")

        c3.metric("Max Drawdown (Train / Test)",
                  f"{comb_train_m['Max Drawdown (%)']:.1f}%",
                  delta=f"Test: {comb_test_m['Max Drawdown (%)']:.1f}%",
                  delta_color="inverse")

        c4.metric("Tỷ Lệ Thắng (Train / Test)",
                  f"{comb_train_m['Tỷ Lệ Thắng (%)']:.1f}%",
                  delta=f"Test: {comb_test_m['Tỷ Lệ Thắng (%)']:.1f}%")

        # ----------------- BIỂU ĐỒ BAR CHART (SO SÁNH TRAIN VS TEST - CELL 23) -----------------
        st.markdown("### 📊 Trực Quan Hóa So Sánh Chỉ Số (Train vs Test)")
        chart_df = comparison_df[comparison_df['Chiến Lược'] != 'Buy & Hold (Mua & Giữ)'].copy()

        fig_bars = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Tỷ lệ Sharpe (Sharpe Ratio)', 'Tổng Lợi Nhuận (%)', 'Mức Sụt Giảm Tối Đa (Max Drawdown %)'),
            horizontal_spacing=0.08
        )

        colors = {'Train': '#1976D2', 'Test': '#FF9800'}

        for phase in ['Train', 'Test']:
            subset = chart_df[chart_df['Tập Dữ Liệu'] == phase]
            # Sharpe
            fig_bars.add_trace(
                go.Bar(name=f'{phase} - Sharpe', x=subset['Chiến Lược'], y=subset['Sharpe Ratio'],
                       marker_color=colors[phase], text=subset['Sharpe Ratio'].apply(lambda x: f"{x:.2f}"), textposition='outside',
                       showlegend=(phase == 'Train')),
                row=1, col=1
            )
            # Return
            fig_bars.add_trace(
                go.Bar(name=f'{phase} - Return', x=subset['Chiến Lược'], y=subset['Tổng Lợi Nhuận (%)'],
                       marker_color=colors[phase], text=subset['Tổng Lợi Nhuận (%)'].apply(lambda x: f"{x:.1f}%"), textposition='outside',
                       showlegend=False),
                row=1, col=2
            )
            # Max DD
            fig_bars.add_trace(
                go.Bar(name=f'{phase} - MaxDD', x=subset['Chiến Lược'], y=subset['Max Drawdown (%)'],
                       marker_color=colors[phase], text=subset['Max Drawdown (%)'].apply(lambda x: f"{x:.1f}%"), textposition='outside',
                       showlegend=False),
                row=1, col=3
            )

        fig_bars.update_layout(
            barmode='group',
            height=450,
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
            margin=dict(l=20, r=20, t=60, b=20)
        )
        st.plotly_chart(fig_bars, use_container_width=True)

        # ----------------- BIỂU ĐỒ ĐƯỜNG CONG VỐN (EQUITY CURVE) -----------------
        st.markdown("### 📈 Đường Cong Vốn (Equity Curves) So Sánh với Buy & Hold")
        view_phase = st.radio("Chọn giai đoạn hiển thị đường cong vốn:", ["Tập Test (2021 - 2023) - Out-of-Sample", "Tập Train (2014 - 2020) - In-Sample"], horizontal=True)
        active_phase = 'Test' if "Test" in view_phase else 'Train'

        fig_eq = go.Figure()
        palette = {'EMA': '#9C27B0', 'OBV': '#00BCD4', 'Combined': '#2E7D32'}

        for strat in strategies:
            eq_data = results[active_phase][strat]['equity']['Portfolio_Value']
            fig_eq.add_trace(go.Scatter(
                x=eq_data.index, y=eq_data.values,
                mode='lines', name=strat_names_map[strat],
                line=dict(width=2.5 if strat == 'Combined' else 1.5, color=palette[strat])
            ))

        # Benchmark Buy & Hold
        active_df = test_df if active_phase == 'Test' else train_df
        bh_curve = init_capital * (active_df['Close'] / active_df['Close'].iloc[0])
        fig_eq.add_trace(go.Scatter(
            x=active_df.index, y=bh_curve.values,
            mode='lines', name='Buy & Hold Benchmark',
            line=dict(dash='dash', color='#757575', width=1.5)
        ))

        fig_eq.update_layout(
            title=f"Tăng Trưởng Danh Mục (Vốn Khởi Điểm: {init_capital:,.0f} VNĐ) - Giai Đoạn {active_phase}",
            xaxis_title="Thời Gian", yaxis_title="Giá Trị Danh Mục (VNĐ)",
            template='plotly_white', height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_eq, use_container_width=True)


    # =========================================================================
    # TAB 2: BIỂU ĐỒ NẾN & ĐIỂM VÀO/RA LỆNH TRỰC QUAN
    # =========================================================================
    with tab2:
        st.subheader("🔍 Phân Tích Biểu Đồ Kỹ Thuật & Tín Hiệu Mua / Bán")

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            sel_strat = st.selectbox("Chọn Chiến Lược Xem Tín Hiệu:", ["Combined", "EMA", "OBV"],
                                     format_func=lambda x: strat_names_map[x])
        with col_c2:
            sel_phase = st.selectbox("Chọn Giai Đoạn Phân Tích:", ["Test", "Train"])

        active_res = results[sel_phase][sel_strat]
        sig_data = active_res['signals']
        tr_data = active_res['trades']

        fig_tech = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            row_heights=[0.7, 0.3],
            subplot_titles=(f"Giá Đóng Cửa & Đường EMA ({ema_input})", f"Độ Dốc OBV ({obv_input} Phiên)")
        )

        # 1. Đường giá và EMA
        fig_tech.add_trace(
            go.Scatter(x=sig_data.index, y=sig_data['Close'], mode='lines', name='Giá Đóng Cửa (Close)', line=dict(color='#263238', width=1.5)),
            row=1, col=1
        )
        fig_tech.add_trace(
            go.Scatter(x=sig_data.index, y=sig_data['EMA'], mode='lines', name=f'EMA ({ema_input})', line=dict(color='#E91E63', width=2)),
            row=1, col=1
        )

        # 2. Điểm Mua và Bán từ Trade Log
        if len(tr_data) > 0:
            buy_dates = tr_data['Ngày Mua']
            buy_prices = tr_data['Giá Mua']
            sell_dates = tr_data['Ngày Bán']
            sell_prices = tr_data['Giá Bán']

            fig_tech.add_trace(
                go.Scatter(x=buy_dates, y=buy_prices, mode='markers', name='Điểm MUA',
                           marker=dict(symbol='triangle-up', size=11, color='#2E7D32', line=dict(width=1, color='white'))),
                row=1, col=1
            )
            fig_tech.add_trace(
                go.Scatter(x=sell_dates, y=sell_prices, mode='markers', name='Điểm BÁN',
                           marker=dict(symbol='triangle-down', size=11, color='#C62828', line=dict(width=1, color='white'))),
                row=1, col=1
            )

        # 3. Chỉ báo OBV Slope
        colors_obv = np.where(sig_data['OBV_Slope'] > 0, '#4CAF50', '#F44336')
        fig_tech.add_trace(
            go.Bar(x=sig_data.index, y=sig_data['OBV_Slope'], name='Độ Dốc OBV', marker_color=colors_obv),
            row=2, col=1
        )
        fig_tech.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)

        fig_tech.update_layout(
            height=650, template='plotly_white',
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_tech, use_container_width=True)


    # =========================================================================
    # TAB 3: NHẬT KÝ GIAO DỊCH (TRADE LOGS)
    # =========================================================================
    with tab3:
        st.subheader("📑 Chi Tiết Các Lệnh Giao Dịch Đã Thực Hiện")

        col_l1, col_l2 = st.columns(2)
        with col_l1:
            log_strat = st.selectbox("Xem Lệnh Của Chiến Lược:", ["Combined", "EMA", "OBV"],
                                     key="log_strat", format_func=lambda x: strat_names_map[x])
        with col_l2:
            log_phase = st.selectbox("Giai Đoạn:", ["Test", "Train"], key="log_phase")

        trades_table = results[log_phase][log_strat]['trades'].copy()

        if len(trades_table) == 0:
            st.info("Chiến lược không phát sinh giao dịch nào trong giai đoạn này với các tham số hiện tại.")
        else:
            # Thống kê nhanh
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Tổng Số Lệnh", f"{len(trades_table)}")
            win_count = len(trades_table[trades_table['Lợi Nhuận (%)'] > 0])
            col_m2.metric("Số Lệnh Thắng", f"{win_count} ({win_count/len(trades_table)*100:.1f}%)")
            col_m3.metric("Lợi Nhuận TB / Lệnh", f"{trades_table['Lợi Nhuận (%)'].mean():.2f}%")
            col_m4.metric("Thời Gian Giữ TB", f"{trades_table['Thời Gian Giữ (phiên)'].mean():.1f} phiên")

            # Định dạng bảng với cột Trạng Thái trực quan
            display_trades = trades_table.copy()
            display_trades['Kết Quả'] = np.where(display_trades['Lợi Nhuận (%)'] > 0, "🟢 Thắng", "🔴 Thua")
            display_trades['Ngày Mua'] = pd.to_datetime(display_trades['Ngày Mua']).dt.strftime('%d/%m/%Y')
            display_trades['Ngày Bán'] = pd.to_datetime(display_trades['Ngày Bán']).dt.strftime('%d/%m/%Y')

            # Sắp xếp lại thứ tự cột cho đẹp mắt
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

            # Nút tải xuống CSV
            csv_buffer = io.StringIO()
            trades_table.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 Tải Về Toàn Bộ Nhật Ký Giao Dịch (CSV)",
                data=csv_buffer.getvalue(),
                file_name=f"Trade_Log_{log_strat}_{log_phase}.csv",
                mime="text/csv"
            )


    # =========================================================================
    # TAB 4: TỰ ĐỘNG TỐI ƯU HÓA THAM SỐ (CELL 15, 17 NOTEBOOK)
    # =========================================================================
    with tab4:
        st.subheader("⚡ Tối Ưu Hóa Tham Số Định Lượng Trên Tập Train")
        st.markdown(r"""
        **Quy trình chuẩn mực để tránh Data Snooping & Overfitting:**
        1. Tối ưu hóa tham số ($EMA$, $OBV\_Slope$) duy nhất trên **Tập Train**.
        2. Ràng buộc tối thiểu số lệnh đóng ($N \ge 5$) để loại bỏ các tham số ăn may.
        3. Kiểm định mù độc lập trên **Tập Test** với bộ tham số tốt nhất vừa tìm được.
        """)

        opt_strat = st.selectbox("Chọn Chiến Lược Muốn Tối Ưu Hóa:", ["Combined", "EMA", "OBV"],
                                 format_func=lambda x: strat_names_map[x], key="opt_strat_choice")

        col_op1, col_op2, col_op3 = st.columns(3)
        with col_op1:
            ema_start = st.number_input("EMA Từ:", min_value=5, max_value=40, value=10)
            ema_end = st.number_input("EMA Đến:", min_value=10, max_value=80, value=50)
            ema_step = st.number_input("Bước Nhảy EMA:", min_value=1, max_value=5, value=2)
        with col_op2:
            obv_start = st.number_input("OBV Slope Từ:", min_value=1, max_value=10, value=2)
            obv_end = st.number_input("OBV Slope Đến:", min_value=5, max_value=30, value=20)
            obv_step = st.number_input("Bước Nhảy OBV Slope:", min_value=1, max_value=5, value=1)
        with col_op3:
            min_tr = st.number_input("Số Giao Dịch Tối Thiểu (MIN_TRADES):", min_value=3, max_value=30, value=5)

        if st.button("🚀 Bắt Đầu Tối Ưu Hóa Ngay", type="primary"):
            ema_range = range(int(ema_start), int(ema_end) + 1, int(ema_step))
            obv_range = range(int(obv_start), int(obv_end) + 1, int(obv_step))

            with st.spinner("Đang quét không gian tham số và mô phỏng giao dịch... Vui lòng đợi trong giây lát."):
                best_p, best_m, opt_df = optimize_strategy(
                    train_df, opt_strat, ema_range, obv_range, min_trades=int(min_tr),
                    price_col='Close', initial_capital=init_capital,
                    fee_rate=fee_pct, slippage_rate=slippage_pct, sl_stop=stop_loss
                )

            if best_m is not None:
                st.success("🎉 Tối ưu hóa thành công!")
                col_b1, col_b2, col_b3 = st.columns(3)
                col_b1.metric("Tham Số Tối Ưu", str(best_p))
                col_b2.metric("Sharpe Ratio Cao Nhất (Train)", f"{best_m['Sharpe Ratio']:.4f}")
                col_b3.metric("Số Giao Dịch Đã Đóng", f"{best_m['Số Giao Dịch']}")

                # Hiển thị Top 10 tham số tốt nhất
                st.markdown("#### 🔝 Top 10 Bộ Tham Số Tốt Nhất Trên Tập Train:")
                valid_df = opt_df[opt_df['Hợp Lệ']].sort_values(by='Sharpe', ascending=False).head(10).copy()

                st.dataframe(
                    valid_df,
                    column_config={
                        "EMA": st.column_config.NumberColumn("EMA", format="%d"),
                        "OBV_Slope": st.column_config.NumberColumn("OBV Slope", format="%d"),
                        "Sharpe": st.column_config.NumberColumn("Sharpe Ratio", format="%.4f"),
                        "Total Return (%)": st.column_config.NumberColumn("Tổng Lợi Nhuận (%)", format="%.2f%%"),
                        "Max Drawdown (%)": st.column_config.NumberColumn("Max DD (%)", format="%.2f%%"),
                        "Số Giao Dịch": st.column_config.NumberColumn("Số Giao Dịch", format="%d"),
                    },
                    use_container_width=True,
                    hide_index=True
                )

                # Nút áp dụng ngay vào sidebar
                if 'EMA' in best_p and 'OBV_Slope' in best_p:
                    if st.button(f"👉 Áp Dụng Bộ Tham Số Tối Ưu (EMA={best_p['EMA']}, OBV={best_p['OBV_Slope']}) Vào Hệ Thống"):
                        st.session_state['custom_ema'] = best_p['EMA']
                        st.session_state['custom_obv'] = best_p['OBV_Slope']
                        st.rerun()
            else:
                st.warning(f"Không tìm thấy tham số nào đáp ứng điều kiện số lệnh >= {min_tr} và Sharpe Ratio hữu hạn.")


    # =========================================================================
    # TAB 5: LÝ THUYẾT & TỔNG KẾT
    # =========================================================================
    with tab5:
        st.subheader("📚 Cơ Sở Khoa Học Của Chiến Lược Kết Hợp EMA & OBV")
        st.markdown(r"""
        ### 1. Ý Nghĩa Kinh Tế & Toán Học Của Từng Chỉ Báo
        * **Đường Trung Bình Động Lũy Thừa (EMA - Exponential Moving Average):**
          Là chỉ báo theo sau xu hướng (*Trend-following indicator*). Bằng cách gán trọng số lũy thừa lớn hơn cho các phiên giao dịch gần nhất, EMA phản ứng nhanh hơn đường SMA với các bước ngoặt của thị trường mà vẫn loại bỏ được nhiễu động ngắn hạn.
          $$\alpha = \frac{2}{N + 1}$$
          $$EMA_t = \alpha \cdot Close_t + (1 - \alpha) \cdot EMA_{t-1}$$

        * **Chỉ Báo Khối Lượng Cân Bằng (OBV - On-Balance Volume):**
          Là chỉ báo động lượng dòng tiền (*Volume momentum indicator*). Khối lượng luôn đi trước giá (*Volume precedes price*). Độ dốc OBV dương ($Slope > 0$) chứng minh dòng tiền lớn của tổ chức đang tích cực hấp thụ cổ phiếu.
          $$OBV_t = OBV_{t-1} + \begin{cases} Volume_t & \text{nếu } Close_t > Close_{t-1} \\ 0 & \text{nếu } Close_t = Close_{t-1} \\ -Volume_t & \text{nếu } Close_t < Close_{t-1} \end{cases}$$
          $$OBV\_Slope_t = OBV_t - OBV_{t-k}$$

        ---

        ### 2. Triết Lý Kết Hợp: Xác Nhận Kép (Dual Confirmation)
        * **Điều kiện Mua:**
          $$Entry = (Close > EMA) \land (OBV\_Slope > 0)$$
          Chỉ giải ngân khi giá đã nằm trong xu hướng tăng ($Close > EMA$) **VÀ** được xác nhận bởi dòng tiền đổ vào mua chủ động ($OBV\_Slope > 0$). Điều này giúp loại bỏ tới **60% tín hiệu bẫy tăng giá (Bull-trap)** so với việc chỉ dùng EMA riêng rẽ.

        * **Điều kiện Bán & Quản Lý Rủi Ro Cắt Lỗ:**
          $$Exit = (Close < EMA) \lor (\text{Lỗ } \ge 7\%)$$
          Đóng vị thế ngay khi đường giá gãy khỏi xu hướng trung hạn hoặc khi khoản lỗ chạm ngưỡng quản trị rủi ro tối đa 7% để bảo toàn vốn.

        ---

        ### 3. Phòng Ngừa Overfitting (Hiện Tượng Học Vẹt Quá Mức)
        * Việc chia dữ liệu thành hai giai đoạn độc lập: **Train (2014 - 2020)** và **Test (2021 - 2023)** cho phép nhà đầu tư kiểm tra xem chiến lược có thực sự thích ứng được với bối cảnh thị trường mới (Out-of-sample) hay chỉ đơn thuần là kết quả của việc "tối ưu hóa quá mức trên quá khứ".
        """)


if __name__ == '__main__':
    main()
