# 📈 Hệ Thống Kiểm Định Chiến Lược Đầu Tư: Kết Hợp Chỉ Báo EMA & OBV

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Ứng dụng web tương tác (Web Application) phục vụ môn học **Quản Lý Danh Mục Đầu Tư**, dùng để kiểm định định lượng (Backtesting & Out-of-Sample Testing) tính hiệu quả của chiến lược giao dịch kết hợp giữa đường trung bình động lũy thừa **EMA** (*Exponential Moving Average*) và chỉ báo khối lượng cân bằng **OBV** (*On-Balance Volume*), có hạch toán quản lý rủi ro cắt lỗ cố định (**Stop-loss 7%**), phí giao dịch và trượt giá thực tế.

---

## 🌟 Tính Năng Nổi Bật Của Web App

1. **Kiểm Định Đa Chiến Lược Đồng Thời (Comparative Backtesting):**
   - **Chiến lược EMA Riêng lẻ:** Mua khi $Close > EMA$, Bán khi $Close < EMA$.
   - **Chiến lược OBV Riêng lẻ:** Mua khi $OBV\_Slope > 0$, Bán khi $OBV\_Slope < 0$.
   - **Chiến lược Kết hợp EMA + OBV (Đề xuất):** Mua khi $Close > EMA \land OBV\_Slope > 0$, Bán khi $Close < EMA$.
   - **Chuẩn so sánh (Benchmark):** Mua và Nắm giữ (*Buy & Hold*).

2. **Kiểm Định Chống Học Vẹt (Overfitting & Walk-Forward Validation):**
   - Phân chia 2 giai đoạn độc lập theo đúng nghiên cứu học thuật:
     * **Tập Huấn Luyện (In-Sample / Train):** 2014 – 2020 (dùng để hiệu chỉnh và tìm kiếm tham số tối ưu).
     * **Tập Kiểm Định Mù (Out-of-Sample / Test):** 2021 – 2023 (dùng để đánh giá khả năng thích ứng thị trường mới).
   - Biểu đồ Bar Chart so sánh trực quan hiệu suất Train vs Test (Sharpe Ratio, Total Return %, Max Drawdown %) giúp nhận diện ngay nguy cơ Overfitting.

3. **Mô Phỏng Giao Dịch Sát Thực Tế (Realistic Backtest Engine):**
   - Loại trừ hoàn toàn lỗi thiên kiến nhìn trước tương lai (*Look-ahead bias*) bằng việc làm trễ tín hiệu 1 phiên (`shift(1)`).
   - Tích hợp chi phí giao dịch thực tế: **Phí giao dịch (0.2%)** và **Trượt giá (0.1%)**.
   - Hạch toán quy tắc quản lý vị thế: **Long-only**, không tích lũy dồn vốn (`accumulate=False`).
   - Tự động kích hoạt lệnh **Cắt lỗ cứng (Stop-loss 7%)** bảo vệ vốn khi giá giảm quá ngưỡng từ mức mua.

4. **Công Cụ Tối Ưu Hóa Tham Số Tự Động (Grid Search Optimization):**
   - Quét không gian tham số trên tập Train nhằm tối đa hóa **Tỷ lệ Sharpe (*Sharpe Ratio*)**.
   - Thiết lập điều kiện ràng buộc $N \ge 5$ giao dịch để loại bỏ các bộ tham số "ăn may" ít giao dịch.
   - Bảng xếp hạng Top 10 bộ tham số tối ưu và nút bấm áp dụng tham số tức thì.

5. **Trực Quan Hóa & Nhật Ký Giao Dịch Chuyên Nghiệp:**
   - Biểu đồ tương tác cao với **Plotly**: Zoom, hover, đánh dấu chính xác điểm Mua (tam giác xanh 🔼) và điểm Bán (tam giác đỏ 🔻).
   - Biểu đồ đường cong vốn (*Equity Curve*) và mức sụt giảm (*Underwater Drawdown*).
   - Bảng nhật ký giao dịch (*Trade Log*) chi tiết từng lệnh: Ngày vào, Giá vào, Ngày ra, Giá ra, Lợi nhuận (%), Lý do thoát (Tín hiệu hay Stop-loss).
   - Hỗ trợ xuất dữ liệu ra file **CSV** phục vụ báo cáo.

---

## 📂 Cấu Trúc Thư Mục Repository

```text
├── app.py                 # Mã nguồn chính của ứng dụng Streamlit
├── requirements.txt       # Danh sách các thư viện cần thiết để cài đặt
├── readme.md              # Tài liệu hướng dẫn sử dụng và triển khai
├── ACB.csv                # File dữ liệu giá lịch sử mẫu (Cổ phiếu ACB 2014 - 2023)
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

Chỉ cần 3 bước đơn giản để đưa ứng dụng lên internet để bất kỳ ai cũng có thể truy cập:

### Bước 1: Đưa mã nguồn lên GitHub
1. Truy cập [GitHub.com](https://github.com) và tạo một Repository mới (ví dụ: `ema-obv-trading-strategy`).
2. Tải toàn bộ 4 file trong thư mục này lên repository:
   - `app.py`
   - `requirements.txt`
   - `readme.md`
   - `ACB.csv`
3. Nhấn **Commit changes** để lưu lại.

### Bước 2: Đăng nhập Streamlit Cloud
1. Truy cập [share.streamlit.io](https://share.streamlit.io/).
2. Đăng nhập bằng tài khoản GitHub vừa tạo.

### Bước 3: Triển khai (Deploy)
1. Nhấn nút **"Create app"** (hoặc **"New app"**).
2. Chọn:
   - **Repository:** `username/ema-obv-trading-strategy`
   - **Branch:** `main`
   - **Main file path:** `app.py`
3. Nhấn nút **"Deploy!"**
4. Đợi khoảng 1 - 2 phút để hệ thống tự động cài đặt thư viện và khởi chạy web app. Sau khi hoàn tất, bạn sẽ nhận được một đường link chia sẻ công khai (ví dụ: `https://your-app-name.streamlit.app`).

---

## 📊 Giải Thích Các Tham Số & Chỉ Số Đánh Giá

| Chỉ số / Tham số | Ký hiệu / Tên | Ý nghĩa kinh tế & định lượng |
| :--- | :--- | :--- |
| **EMA Period** | $N_{EMA}$ | Chu kỳ số phiên tính đường trung bình động lũy thừa (Mặc định: 20 phiên, Tối ưu: 36 phiên). |
| **OBV Slope** | $k_{OBV}$ | Số phiên tính độ dốc khối lượng cân bằng (Mặc định: 3 phiên, Tối ưu: 19 - 20 phiên). |
| **Stop-Loss** | $SL$ | Ngưỡng cắt lỗ cố định (7%), bảo toàn vốn khi giá giảm ngoài dự kiến. |
| **Sharpe Ratio** | $S_p$ | Thước đo lợi nhuận điều chỉnh theo rủi ro biến động: $\frac{R_p - R_f}{\sigma_p} \times \sqrt{252}$. Càng cao càng tốt (> 1.0 là xuất sắc). |
| **Max Drawdown** | $MDD$ | Mức sụt giảm vốn lớn nhất từ đỉnh đến đáy sâu nhất. Càng thấp càng an toàn. |
| **Win Rate** | $WR$ | Tỷ lệ số lệnh có lợi nhuận trên tổng số lệnh đã thực hiện (%). |
| **Profit Factor** | $PF$ | Tỷ số giữa Tổng lợi nhuận gộp trên Tổng lỗ gộp. $PF > 1.5$ thể hiện chiến lược có lợi thế vượt trội. |

---

## 👥 Tác Giả & Bản Quyền
- **Dự án:** Ứng dụng kiểm định chiến lược EMA + OBV.
- **Ngôn ngữ phát triển:** Python & Streamlit Framework.
- Mọi thắc mắc hoặc đóng góp vui lòng mở Issue hoặc tạo Pull Request trên GitHub.
