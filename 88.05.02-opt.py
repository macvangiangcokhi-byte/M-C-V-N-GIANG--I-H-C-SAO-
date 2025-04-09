import numpy as np
from scipy.optimize import minimize

# --- 1. Định nghĩa hàm dự đoán Ra dựa trên phương trình của bạn ---
def predict_ra(params):
    """
    Tính toán Ra dự đoán từ bộ thông số cắt [V, Fz, ap, ae].

    Args:
        params (list or np.array): Danh sách hoặc mảng chứa [V, Fz, ap, ae].

    Returns:
        float: Giá trị Ra dự đoán.
    """
    V, Fz, ap, ae = params # Giải nén các tham số

    # Áp dụng chính xác phương trình hồi quy bạn đã cung cấp
    Ra_pred = (
        0.350000
        + 0.001583 * V
        + 3.333333 * Fz
        + 0.116667 * ap
        + 0.003333 * ae
        + 0.000379 * (V**2)
        - 0.025000 * V * Fz
        # + 0.000000 * V * ap  # Bỏ qua vì hệ số là 0
        # + 0.000000 * V * ae  # Bỏ qua vì hệ số là 0
        + 466.666667 * (Fz**2)
        - 5.000000 * Fz * ap
        # + 0.000000 * Fz * ae  # Bỏ qua vì hệ số là 0
        + 14.166667 * (ap**2)
        - 0.400000 * ap * ae
        + 0.166667 * (ae**2)
    )
    return Ra_pred

# --- 2. Định nghĩa hàm mục tiêu để tối ưu hóa ---
def objective_function(params, target_ra):
    """
    Hàm mục tiêu: Tính bình phương sai số giữa Ra dự đoán và Ra mục tiêu.

    Args:
        params (list or np.array): Danh sách hoặc mảng chứa [V, Fz, ap, ae].
        target_ra (float): Giá trị Ra mục tiêu mong muốn.

    Returns:
        float: Bình phương sai số (Ra_predict - target_ra)^2.
    """
    # Áp đặt ràng buộc tối thiểu cứng nếu cần (tránh giá trị âm/quá nhỏ từ tối ưu hóa)
    # Sử dụng các giá trị min đã định nghĩa trong bounds
    params[0] = max(params[0], 7.85)  # V >= 7.85
    params[1] = max(params[1], 0.01)  # Fz >= 0.01
    params[2] = max(params[2], 0.001) # ap >= 0.001 (hoặc ap_min dựa trên D)
    params[3] = max(params[3], 0.001) # ae >= 0.001 (hoặc ae_min dựa trên D)

    ra_predicted = predict_ra(params)
    error_sq = (ra_predicted - target_ra)**2
    return error_sq

# --- 3. Nhập thông tin cần thiết ---
print("--- Tìm bộ thông số cắt cho Ra mục tiêu ---")

# Nhập Ra mục tiêu
while True:
    try:
        target_ra = float(input("Nhập giá trị Ra mục tiêu mong muốn (ví dụ: 0.4): "))
        if target_ra > 0:
            break
        else:
            print("Ra mục tiêu phải là số dương.")
    except ValueError:
        print("Vui lòng nhập một số hợp lệ.")

# --- *** PHẦN SỬA ĐỔI THEO YÊU CẦU MỚI NHẤT *** ---
# Nhập đường kính dao
while True:
    try:
        cutter_diameter_D = float(input("Nhập đường kính dao phay D (mm): "))
        if cutter_diameter_D > 0:
            break
        else:
            print("Đường kính dao phải là số dương.")
    except ValueError:
        print("Vui lòng nhập một số hợp lệ.")

# Gán trực tiếp giới hạn (Bounds) dựa trên yêu cầu MỚI NHẤT
print(f"\nSử dụng giới hạn (bounds) được tính toán với D = {cutter_diameter_D} mm:")

# Giới hạn cố định cho V và Fz (THEO YÊU CẦU MỚI NHẤT)
v_min = 7.85      # m/phút (Đã thay đổi)
v_max = 1979.2    # m/phút
fz_min = 0.01     # mm/răng
fz_max = 0.25     # mm/răng

# Giới hạn cho ap và ae phụ thuộc vào D (như trước)
ap_min = 0.01 * cutter_diameter_D  # mm (0.01 * D)
ap_max = 1.0 * cutter_diameter_D   # mm (1.0 * D)
ae_min = 0.02 * cutter_diameter_D  # mm (0.02 * D)
ae_max = 0.75 * cutter_diameter_D  # mm (0.75 * D)

# Đảm bảo giá trị min không lớn hơn max (trường hợp D quá nhỏ)
# và không nhỏ hơn một giá trị sàn hợp lý (ví dụ 0.001 hoặc giá trị min đã cho)
ap_min = max(0.001, ap_min) # Giữ mức sàn tối thiểu cho ap
ae_min = max(0.001, ae_min) # Giữ mức sàn tối thiểu cho ae
ap_max = max(ap_min, ap_max) # Đảm bảo max >= min
ae_max = max(ae_min, ae_max) # Đảm bảo max >= min

bounds = [
    (v_min, v_max),
    (fz_min, fz_max),
    (ap_min, ap_max),
    (ae_min, ae_max),
]
print(f" - Giới hạn V:  ({bounds[0][0]:.2f}, {bounds[0][1]:.1f}) m/phút") # Cập nhật format cho v_min
print(f" - Giới hạn Fz: ({bounds[1][0]:.3f}, {bounds[1][1]:.3f}) mm/răng")
print(f" - Giới hạn ap: ({bounds[2][0]:.3f}, {bounds[2][1]:.3f}) mm (Phụ thuộc D)")
print(f" - Giới hạn ae: ({bounds[3][0]:.3f}, {bounds[3][1]:.3f}) mm (Phụ thuộc D)")
# --- *** KẾT THÚC PHẦN SỬA ĐỔI *** ---


# Nhập điểm bắt đầu (Initial Guess) - Nên nằm trong khoảng bounds
print("\nNhập giá trị ban đầu (dự đoán) cho các thông số:")
try:
    # Sử dụng bounds đã được định nghĩa ở trên để hiển thị khoảng gợi ý
    v_guess = float(input(f" - V ban đầu (trong khoảng [{bounds[0][0]:.2f}, {bounds[0][1]:.1f}]): "))
    fz_guess = float(input(f" - Fz ban đầu (trong khoảng [{bounds[1][0]:.3f}, {bounds[1][1]:.3f}]): "))
    ap_guess = float(input(f" - ap ban đầu (trong khoảng [{bounds[2][0]:.3f}, {bounds[2][1]:.3f}]): "))
    ae_guess = float(input(f" - ae ban đầu (trong khoảng [{bounds[3][0]:.3f}, {bounds[3][1]:.3f}]): "))
    initial_guess = [v_guess, fz_guess, ap_guess, ae_guess]
    # Kiểm tra xem initial guess có nằm trong bounds không (tùy chọn)
    valid_guess = all(b[0] <= g <= b[1] for g, b in zip(initial_guess, bounds))
    if not valid_guess:
        print("Cảnh báo: Giá trị ban đầu nằm ngoài giới hạn!")
        # Đặt lại giá trị mặc định là trung bình của bounds
        initial_guess = [(b[0] + b[1]) / 2.0 for b in bounds]
        print(f"Sử dụng điểm bắt đầu là trung bình giới hạn: {initial_guess}")

except ValueError:
    print("Lỗi nhập liệu. Sử dụng giá trị trung bình của giới hạn làm điểm bắt đầu.")
    initial_guess = [(b[0] + b[1]) / 2.0 for b in bounds]
    print(f"Điểm bắt đầu (ví dụ): {initial_guess}")


# --- 4. Thực hiện tối ưu hóa ---
print("\nĐang thực hiện tối ưu hóa...")
result = minimize(
    objective_function,      # Hàm cần tối thiểu hóa
    initial_guess,           # Điểm bắt đầu
    args=(target_ra,),       # Tham số phụ cho hàm mục tiêu (Ra mục tiêu)
    method='L-BFGS-B',       # Phương pháp tối ưu hóa hỗ trợ bounds
    bounds=bounds            # Giới hạn cho các thông số
)

# --- 5. Hiển thị kết quả ---
print("\n--- Kết quả tối ưu hóa ---")
if result.success:
    optimal_params = result.x
    # Đảm bảo các giá trị tối ưu không vi phạm cận dưới (do sai số tính toán)
    optimal_params[0] = max(bounds[0][0], optimal_params[0]) # V >= v_min
    optimal_params[1] = max(bounds[1][0], optimal_params[1]) # Fz >= fz_min
    optimal_params[2] = max(bounds[2][0], optimal_params[2]) # ap >= ap_min
    optimal_params[3] = max(bounds[3][0], optimal_params[3]) # ae >= ae_min

    final_predicted_ra = predict_ra(optimal_params)

    print(f"Tìm thấy bộ thông số cắt tối ưu cho Ra mục tiêu ≈ {target_ra:.4f} (với D = {cutter_diameter_D} mm):")
    print(f"  - V  (m/phút): {optimal_params[0]:.3f}")
    print(f"  - Fz (mm/răng): {optimal_params[1]:.6f}")
    print(f"  - ap (mm):      {optimal_params[2]:.3f}")
    print(f"  - ae (mm):      {optimal_params[3]:.3f}")
    print("-" * 20)
    print(f"Giá trị Ra dự đoán với bộ thông số này: {final_predicted_ra:.4f}")
    print(f"Độ lệch so với mục tiêu: {abs(final_predicted_ra - target_ra):.6f}")
    print("\nLưu ý: Có thể có các bộ thông số khác cũng cho kết quả Ra tương tự.")
else:
    print(f"Tối ưu hóa không thành công.")
    print(f"Lý do: {result.message}")
