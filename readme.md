# 📊 Hệ Thống Kiểm Định Chiến Lược Đầu Tư: Kết Hợp Chỉ Báo SMA & OBV

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Ứng dụng web tương tác (Web Application) phục vụ môn học **Quản Lý Danh Mục Đầu Tư**, dùng để kiểm định định lượng (Backtesting & Train/Test Split) tính hiệu quả của chiến lược giao dịch kết hợp giữa hai đường trung bình động giản đơn **SMA** (*Simple Moving Average Crossover*) và chỉ báo khối lượng cân bằng **OBV** (*On-Balance Volume Crossover với OBV Moving Average*) theo mô hình nghiên cứu của **Nhóm 5** trên cổ phiếu **ACB**.

---

## 🌟 Tính Năng Nổi Bật Của Web App

1. **Kiểm Định Đa Chiến Lược Đồng Thời (Comparative Backtesting):**
   - **Chiến lược SMA (Giao cắt 2 đường MA):** Mua khi SMA ngắn cắt lên trên SMA dài (Golden Cross), Bán khi SMA ngắn cắt xuống dưới SMA dài (Death Cross).
   - **Chiến lược OBV (Giao cắt OBV MA):** Mua khi OBV cắt lên trên đường trung bình động của chính nó ($OBV > OBV\_MA$), Bán khi OBV cắt xuống dưới đường trung bình động ($OBV < OBV\_MA$).
   - **Chiến lược Kết hợp SMA + OBV (Quy tắc AND):** Mua khi cả hai chiến lược cùng phát lệnh MUA, Bán khi cả hai cùng phát lệnh BÁN.
   - **Chuẩn so sánh thị trường (Benchmark):** Mua và Nắm giữ (*Buy & Hold*).

2. **Phân Tách Train / Test Khoa Học (Hold-Out Validation):**
   - Hỗ trợ chia dữ liệu theo tỷ lệ chuẩn **80% Train / 20% Test** (như Cell 6 của Notebook Nhóm 5) hoặc chia theo năm tùy chỉnh.
   - Đảm bảo nguyên tắc học thuật: **Chỉ tối ưu tham số trên tập Train**, sau đó kiểm định độc lập trên **tập Test** để kiểm tra tính thích ứng của mô hình.
   - Biểu đồ Bar Chart so sánh trực quan hiệu suất Train vs Test (Return %, Sharpe Ratio, Max Drawdown %, Trades).

3. **Công Cụ Tối Ưu Hóa Tham Số Tự Động (Train Optimization Engine):**
   - Quét lưới không gian tham số cho SMA (`ma_short`, `ma_long`) và OBV (`obv_window`) trên tập Train.
   - Hiển thị Top 10 bộ tham số có tỷ suất sinh lời cao nhất kèm nút bấm áp dụng tham số tức thì vào hệ thống.

4. **Trực Quan Hóa & Nhật Ký Lệnh Chuyên Nghiệp:**
   - Biểu đồ Plotly tương tác cao: Giá nến/đường kèm 2 đường SMA, đánh dấu chính xác điểm Mua (tam giác xanh 🔼) và điểm Bán (tam giác đỏ 🔻), subplot chỉ báo OBV kèm đường trung bình động OBV MA.
   - Biểu đồ đường cong vốn (*Equity Curve*) so sánh tăng trưởng tài khoản của từng chiến lược với Buy & Hold.
   - Bảng nhật ký giao dịch (*Trade Log*) chi tiết từng lệnh: Trạng thái (🟢 Thắng / 🔴 Thua), Ngày vào, Giá vào, Ngày ra, Giá ra, Lợi nhuận (%), Số phiên giữ.
   - Hỗ trợ xuất dữ liệu nhật ký giao dịch ra file **CSV**.

5. **Tương Thích Tuyệt Đối Với Streamlit Cloud:**
   - Sử dụng chuẩn `st.column_config` của Streamlit, loại bỏ hoàn toàn các lỗi ép kiểu dữ liệu giữa các phiên bản NumPy và Pandas (như lỗi `UFuncNoLoopError`).

---

## 📂 Cấu Trúc Thư Mục Repository

```text
├── app.py                 # Mã nguồn chính của ứng dụng Streamlit (SMA + OBV)
├── requirements.txt       # Danh sách các thư viện cần thiết để cài đặt
├── readme.md              # Tài liệu hướng dẫn sử dụng và triển khai
├── ACB.csv                # File dữ liệu giá lịch sử mẫu (Cổ phiếu ACB)
```

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Trên Máy Cá Nhân (Local)

### 1. Yêu cầu hệ thống:
- Đã cài đặt **Python 3.9** trở lên trên máy tính.

### 2. Cài đặt các thư viện phụ thuộc:
Mở Terminal (Command Prompt hoặc PowerShell trên Windows, Terminal trên macOS/Linux) và chạy lệnh:

```bash
pip install -r requirements.txt
```

### 3. Khởi chạy ứng dụng:
Tại thư mục chứa dự án, gõ lệnh:

```bash
streamlit run app.py
```

Trình duyệt web sẽ tự động mở trang web tại địa chỉ: `http://localhost:8501`

---

## 🌐 Hướng Dẫn Deploy Lên Streamlit Cloud (Miễn Phí 100%)

Chỉ cần 3 bước đơn giản để đưa ứng dụng lên internet:

### Bước 1: Đưa mã nguồn lên GitHub
1. Truy cập [GitHub.com](https://github.com) và tạo một Repository mới (ví dụ: `acb-sma-obv-strategy`).
2. Tải toàn bộ 4 file trong thư mục này lên repository:
   - `app.py`
   - `requirements.txt`
   - `readme.md`
   - `ACB.csv`
3. Nhấn **Commit changes** để lưu lại.

### Bước 2: Đăng nhập Streamlit Cloud
1. Truy cập [share.streamlit.io](https://share.streamlit.io/).
2. Đăng nhập bằng tài khoản GitHub của bạn.

### Bước 3: Triển khai (Deploy)
1. Nhấn nút **"Create app"** (hoặc **"New app"**).
2. Chọn:
   - **Repository:** `username/acb-sma-obv-strategy`
   - **Branch:** `main`
   - **Main file path:** `app.py`
3. Nhấn nút **"Deploy!"**
4. Sau 1 - 2 phút cài đặt, bạn sẽ nhận được đường link web app công khai (ví dụ: `https://acb-sma-obv-strategy.streamlit.app`).

---

## 📊 Tham Số Tối Ưu Từ Nghiên Cứu Của Nhóm 5 (ACB)

| Tham số | Ký hiệu trong code | Giá trị mặc định kinh điển | Giá trị tối ưu tìm được trên Train |
| :--- | :--- | :--- | :--- |
| **SMA Ngắn hạn** | `ma_short` | 50 phiên | **150 phiên** |
| **SMA Dài hạn** | `ma_long` | 200 phiên | **210 phiên** |
| **Chu kỳ OBV MA** | `obv_window` | 20 phiên | **5 phiên** |

---

## 👥 Tác Giả & Bản Quyền
- **Dự án:** Ứng dụng kiểm định chiến lược SMA + OBV (Nhóm 5).
- **Ngôn ngữ phát triển:** Python & Streamlit Framework.
