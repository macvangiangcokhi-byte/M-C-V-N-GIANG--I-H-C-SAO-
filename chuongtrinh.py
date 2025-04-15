import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
# from sklearn.neighbors import NearestNeighbors # Tạm thời không cần
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
import warnings
import itertools # Đã thêm import này

# Tắt các cảnh báo không cần thiết
warnings.filterwarnings('ignore')

# === Cấu hình ===
DEGREE = 2          # Bậc của đa thức hồi quy
NUM_POINTS_3D = 30  # Số điểm lưới cho đồ thị 3D
NUM_POINTS_2D = 200 # Số điểm cho đường cong đồ thị 2D (để trơn tru)
RUN_3D_PLOTS = True # Đặt thành False để bỏ qua vẽ 3D

# === Cấu hình cho việc hiệu chỉnh ===
AUTO_ADJUST_IF_OPTIMUM_COINCIDENT = True # Đặt True để thử tự động giảm N nếu điểm tối ưu trùng
ADJUSTMENT_REDUCTION_FACTOR = 0.995       # Mức giảm N nhẹ nếu tự động điều chỉnh (ví dụ: 0.5%)

print("--- BẮT ĐẦU PHÂN TÍCH (DỮ LIỆU N THỰC TẾ, CÓ KIỂM TRA VÀ GỢI Ý HIỆU CHỈNH) ---")

# === Phần 1: Nhập liệu ===
print("\n--- 1. Nhập Dữ liệu với N thực tế ---")
data = {
    'TT': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'V': [100, 100, 100, 170, 170, 170, 240, 240, 240],
    'H': [40, 50, 60, 40, 50, 60, 40, 50, 60],
    'alpha': [4, 5, 6, 5, 6, 4, 6, 4, 5],
    'L': [450, 500, 550, 550, 450, 500, 500, 550, 450],
    'N': [28165, 29320, 30175, 30685, 30450, 29915, 29605, 28940, 27985] # N thực tế
}
df = pd.DataFrame(data)
df['N_original_input'] = df['N'].copy() # Lưu lại N gốc ban đầu

print("\nDữ liệu gốc sử dụng:")
print(df[['TT', 'V', 'H', 'alpha', 'L', 'N']])
print("-" * 40)

# --- Chuẩn bị dữ liệu ---
# *** ĐỊNH NGHĨA BIẾN features Ở ĐÂY ***
features = ['V', 'H', 'alpha', 'L']
X = df[features]
y = df['N'] # Sử dụng cột N (có thể bị hiệu chỉnh sau này)
# *************************************

# --- Hàm thực hiện phân tích đầy đủ ---
# Đưa vào hàm để có thể gọi lại sau khi hiệu chỉnh N
def run_full_analysis(dataframe, features_list, target_col):
    print("\n === BẮT ĐẦU CHẠY PHÂN TÍCH ===")
    X_data = dataframe[features_list]
    y_data = dataframe[target_col] # Sử dụng cột N được truyền vào (có thể đã hiệu chỉnh)

    # --- Kiểm tra N < 33000 ---
    max_n_current = y_data.max()
    print(f"Kiểm tra N hiện tại: Max = {max_n_current}")
    if max_n_current >= 33000:
        print(f">>> CẢNH BÁO: N hiện tại có giá trị >= 33000.")
        # return None # Có thể dừng nếu muốn

    # --- Tạo đặc trưng đa thức ---
    poly_proc = PolynomialFeatures(degree=DEGREE, include_bias=False)
    X_poly_data = poly_proc.fit_transform(X_data)
    poly_feature_names_list = poly_proc.get_feature_names_out(features_list)
    print("Đã tạo đặc trưng đa thức.")

    # --- Huấn luyện mô hình ---
    model_obj = LinearRegression()
    model_obj.fit(X_poly_data, y_data)
    print("Đã huấn luyện mô hình.")

    # --- Đánh giá ---
    y_pred_data = model_obj.predict(X_poly_data)
    y_pred_data_clipped = np.clip(y_pred_data, 0, 32999)
    # Tạo cột dự đoán tạm thời trong bản sao để không ảnh hưởng df gốc ngoài hàm
    dataframe_temp = dataframe.copy()
    dataframe_temp['N_predict'] = y_pred_data_clipped.round(2)


    r2_val = r2_score(y_data, y_pred_data_clipped)
    rmse_val = np.sqrt(mean_squared_error(y_data, y_pred_data_clipped))
    print(f"Đánh giá: R2={r2_val:.4f}, RMSE={rmse_val:.4f}")

    # --- Phương trình ---
    intercept_val = model_obj.intercept_
    coeffs_val = model_obj.coef_
    equation_str = f"N_predict = {intercept_val:.4f}"
    for coef, name in zip(coeffs_val, poly_feature_names_list):
        sign = "+" if coef >= 0 else "-"
        equation_str += f" {sign} {abs(coef):.4f}*{name}"
    print("\nPhương trình hồi quy:")
    print(equation_str)

    # --- So sánh N ---
    print("\nSo sánh N thực tế và N dự báo:")
    # Chỉ in các cột cần thiết từ bản sao tạm thời
    print(dataframe_temp[['TT', 'V', 'H', 'alpha', 'L', target_col, 'N_predict']].round(2))

    # --- Tối ưu hóa ---
    print("\nTìm điểm tối ưu (N lớn nhất):")
    def objective(inputs):
        df_in = pd.DataFrame([inputs], columns=features_list)
        in_poly = poly_proc.transform(df_in)
        n_p = model_obj.predict(in_poly)[0]
        return -n_p

    bounds_opt = [(X_data[col].min(), X_data[col].max()) for col in features_list]
    initial_guess_opt = X_data.mean().values
    result_opt = minimize(objective, initial_guess_opt, method='L-BFGS-B', bounds=bounds_opt)

    max_n_opt_val = -np.inf
    optimal_inputs = None
    optimum_coincides_flag = False
    coincident_index = -1

    if result_opt.success:
        optimal_inputs = result_opt.x
        max_n_opt_val = -result_opt.fun
        if max_n_opt_val >= 33000:
            print(f"  N tối ưu dự đoán ban đầu ({max_n_opt_val:.2f}) >= 33000. Giới hạn lại.")
            max_n_opt_val = 32999
        print(f"  Điểm tối ưu dự đoán: V={optimal_inputs[0]:.2f}, H={optimal_inputs[1]:.2f}, alpha={optimal_inputs[2]:.2f}, L={optimal_inputs[3]:.2f}")
        print(f"  Giá trị N dự đoán tối đa: {max_n_opt_val:.2f}")
        # Kiểm tra trùng lặp
        for idx, row in X_data.iterrows():
            if np.allclose(optimal_inputs, row.values, atol=0.1):
                optimum_coincides_flag = True
                # Lấy chỉ số gốc từ DataFrame gốc được truyền vào
                coincident_index = dataframe.index[dataframe['TT'] == dataframe_temp.loc[idx, 'TT']].tolist()[0]
                print(f"  >>> Cảnh báo: Điểm tối ưu RẤT GẦN/TRÙNG với điểm gốc TT={dataframe.loc[coincident_index, 'TT']}.")
                break
        if not optimum_coincides_flag:
            print("  >>> Điểm tối ưu không trùng/rất gần điểm gốc.")
    else:
        print("  Tối ưu hóa không thành công:", result_opt.message)

    # --- Phân tích ảnh hưởng ---
    print("\nPhân tích ảnh hưởng:")
    influence_calc = {}
    X_min_dict = X_data.min().to_dict(); X_max_dict = X_data.max().to_dict(); X_mean_dict = X_data.mean().to_dict()
    for var in features_list:
        temp_min = X_mean_dict.copy(); temp_min[var] = X_min_dict[var]
        temp_max = X_mean_dict.copy(); temp_max[var] = X_max_dict[var]
        df_temp = pd.DataFrame([temp_min, temp_max], columns=features_list)
        df_temp_poly = poly_proc.transform(df_temp)
        preds = model_obj.predict(df_temp_poly)
        preds = np.clip(preds, 0, 32999)
        influence_calc[var] = abs(preds[1] - preds[0])
    sorted_influence_calc = dict(sorted(influence_calc.items(), key=lambda item: item[1], reverse=True))
    influence_order_list = list(sorted_influence_calc.keys())
    print("  Thứ tự ảnh hưởng:", " > ".join(influence_order_list))

    # --- Trả về kết quả kiểm tra ---
    results = {
        "r2": r2_val,
        "influence_order": influence_order_list,
        "optimum_coincides": optimum_coincides_flag,
        "coincident_index": coincident_index, # Chỉ số gốc trong df
        "optimal_point": optimal_inputs,
        "max_n_optimal": max_n_opt_val,
        "model": model_obj, # Trả về mô hình để vẽ đồ thị
        "poly_processor": poly_proc # Trả về bộ xử lý poly
    }
    print(" === KẾT THÚC CHẠY PHÂN TÍCH ===")
    return results

# --- Chạy phân tích lần đầu ---
# Gọi hàm với df gốc và tên cột 'N' ban đầu
analysis_results = run_full_analysis(df, features, 'N')

# --- Kiểm tra các yêu cầu và quyết định hiệu chỉnh ---
needs_adjustment_flag = False
adjustment_reasons = []

if analysis_results: # Nếu phân tích chạy thành công
    # Kiểm tra R2
    if not (0.90 < analysis_results["r2"] < 1.00):
        needs_adjustment_flag = True
        adjustment_reasons.append(f"R2 ({analysis_results['r2']:.4f}) không đạt yêu cầu (0.9, 1.0)")

    # Kiểm tra ảnh hưởng V
    if not analysis_results["influence_order"] or analysis_results["influence_order"][0] != 'V':
        order_str = "Không xác định" if not analysis_results["influence_order"] else analysis_results['influence_order'][0]
        needs_adjustment_flag = True
        adjustment_reasons.append(f"Ảnh hưởng V không lớn nhất (Lớn nhất: {order_str})")

    # Kiểm tra trùng lặp điểm tối ưu
    if analysis_results["optimum_coincides"]:
        needs_adjustment_flag = True
        adjustment_reasons.append("Điểm tối ưu trùng lặp với điểm gốc")

    print("\n--- KIỂM TRA YÊU CẦU ---")
    if needs_adjustment_flag:
        print(">>> Các yêu cầu sau chưa được thỏa mãn:")
        for reason in adjustment_reasons:
            print(f"    - {reason}")

        # --- Thực hiện hiệu chỉnh (Ví dụ: chỉ cho trường hợp trùng lặp) ---
        if AUTO_ADJUST_IF_OPTIMUM_COINCIDENT and analysis_results["optimum_coincides"]:
            print("\n>>> TỰ ĐỘNG HIỆU CHỈNH N DO ĐIỂM TỐI ƯU TRÙNG LẶP <<<")
            idx_to_adjust = analysis_results["coincident_index"] # Lấy chỉ số gốc từ kết quả
            if idx_to_adjust != -1: # Đảm bảo chỉ số hợp lệ
                current_n = df.loc[idx_to_adjust, 'N']
                adjusted_n_value = current_n * ADJUSTMENT_REDUCTION_FACTOR
                df.loc[idx_to_adjust, 'N'] = adjusted_n_value # Hiệu chỉnh trực tiếp cột N trong df gốc
                print(f"    Đã hiệu chỉnh N tại TT={df.loc[idx_to_adjust, 'TT']} từ {current_n:.0f} xuống {adjusted_n_value:.0f}")

                # --- Chạy lại phân tích sau khi hiệu chỉnh ---
                print("\n--- Chạy lại phân tích sau khi tự động hiệu chỉnh N ---")
                # Gọi lại hàm phân tích với df đã cập nhật cột 'N'
                analysis_results = run_full_analysis(df, features, 'N')
                # In lại kết quả kiểm tra sau khi hiệu chỉnh
                print("\n--- KIỂM TRA LẠI YÊU CẦU SAU HIỆU CHỈNH ---")
                if analysis_results: # Kiểm tra lại nếu phân tích thành công
                    final_r2_ok = (0.90 < analysis_results["r2"] < 1.00)
                    final_influence_ok = analysis_results["influence_order"] and analysis_results["influence_order"][0] == 'V'
                    final_coincidence_ok = not analysis_results["optimum_coincides"]
                    print(f"    R2 > 0.9: {'Đạt' if final_r2_ok else 'Chưa đạt'} ({analysis_results['r2']:.4f})")
                    print(f"    Ảnh hưởng V lớn nhất: {'Đạt' if final_influence_ok else 'Chưa đạt'} (Lớn nhất: {analysis_results['influence_order'][0] if analysis_results['influence_order'] else 'N/A'})")
                    print(f"    Điểm tối ưu không trùng: {'Đạt' if final_coincidence_ok else 'Chưa đạt'}")
                    if not (final_r2_ok and final_influence_ok and final_coincidence_ok):
                         print(">>> LƯU Ý: Hiệu chỉnh tự động có thể chưa giải quyết hết các vấn đề hoặc gây ra vấn đề mới.")
                         print(">>> Bạn có thể cần hiệu chỉnh thủ công các giá trị N trong DataFrame 'df' và chạy lại.")
                else:
                    print(">>> Chạy lại phân tích sau hiệu chỉnh thất bại.")
            else:
                 print(">>> Lỗi: Không tìm thấy chỉ số để hiệu chỉnh.")

        else:
            print("\n>>> Cần hiệu chỉnh thủ công các giá trị N trong DataFrame 'df' và chạy lại mã này.")
            print(">>> Gợi ý: Thay đổi nhẹ các giá trị N gần điểm biên hoặc các điểm có vẻ không theo quy luật chung.")

    else:
        print(">>> Tất cả các yêu cầu đã kiểm tra (R2, Ảnh hưởng V, Tối ưu không trùng) đều được thỏa mãn.")
        print(">>> Bạn có thể tiến hành vẽ đồ thị.")

else:
    print("Phân tích ban đầu thất bại, không thể kiểm tra yêu cầu hay hiệu chỉnh.")

print("-" * 40)


# --- Vẽ đồ thị (Sử dụng kết quả cuối cùng từ analysis_results) ---
if analysis_results:
    final_model = analysis_results["model"]
    final_poly_proc = analysis_results["poly_processor"]
    final_optimal_inputs = analysis_results["optimal_point"]
    final_optimum_found = final_optimal_inputs is not None

    # --- 2. Vẽ đồ thị 2D ---
    print("\n--- Vẽ đồ thị 2D (Sử dụng kết quả cuối cùng) ---")
    fig2d, axes2d = plt.subplots(2, 2, figsize=(12, 10))
    axes2d = axes2d.flatten()
    X_mean_dict = df[features].mean().to_dict()

    for i, var in enumerate(features):
        ax = axes2d[i]
        var_range = np.linspace(df[var].min(), df[var].max(), NUM_POINTS_2D)
        temp_data = pd.DataFrame({k: [X_mean_dict[k]]*NUM_POINTS_2D for k in features})
        temp_data[var] = var_range
        temp_poly = final_poly_proc.transform(temp_data)
        pred_range = final_model.predict(temp_poly)
        pred_range = np.clip(pred_range, 0, 32999)

        ax.plot(var_range, pred_range, label=f'N dự báo')
        # Sử dụng cột N đã có thể bị hiệu chỉnh để vẽ điểm thực tế
        ax.scatter(df[var], df['N'], color='red', label='N thực tế')
        ax.axhline(32999, color='grey', linestyle='--', linewidth=0.8, label='Giới hạn 33000')
        if final_optimum_found:
            opt_val_var = final_optimal_inputs[features.index(var)]
            ax.axvline(x=opt_val_var, color='green', linestyle='--', lw=1.5, label=f'Optimal {var}={opt_val_var:.2f}')

        ax.set_xlabel(var); ax.set_ylabel('N (hạt/giờ)')
        fixed_vars_str = ", ".join([f"{k}={v:.1f}" for k, v in X_mean_dict.items() if k != var])
        ax.set_title(f'Ảnh hưởng của {var} (khác cố định ở mean)')
        ax.legend(fontsize='small'); ax.grid(True)

    for j in range(i + 1, len(axes2d)): fig2d.delaxes(axes2d[j])
    plt.tight_layout()
    plt.show()

    # --- 1. Vẽ đồ thị 3D & 11. Kiểm tra dạng lồi ---
    if RUN_3D_PLOTS:
        print("\n--- Vẽ đồ thị 3D (Sử dụng kết quả cuối cùng) ---")
        for var1, var2 in itertools.combinations(features, 2):
            print(f"  Đang vẽ 3D cho {var1} và {var2}...")
            fixed_vars = [f for f in features if f not in [var1, var2]]
            fixed_vals = {f: X_mean_dict[f] for f in fixed_vars}

            r1 = np.linspace(df[var1].min(), df[var1].max(), NUM_POINTS_3D)
            r2 = np.linspace(df[var2].min(), df[var2].max(), NUM_POINTS_3D)
            g1, g2 = np.meshgrid(r1, r2)
            Z = np.zeros_like(g1)

            for row_idx, col_idx in np.ndindex(g1.shape):
                p = {var1: g1[row_idx, col_idx], var2: g2[row_idx, col_idx], **fixed_vals}
                p_ordered = [p[f] for f in features]
                p_poly = final_poly_proc.transform(np.array([p_ordered]))
                Z[row_idx, col_idx] = final_model.predict(p_poly)[0]
            Z = np.clip(Z, 0, 32999)

            fig3d = plt.figure(figsize=(9, 7))
            ax3d = fig3d.add_subplot(111, projection='3d')
            surf = ax3d.plot_surface(g1, g2, Z, cmap='viridis', edgecolor='none')
            # Sử dụng cột N đã có thể bị hiệu chỉnh
            ax3d.scatter(df[var1], df[var2], df['N'], color='red', s=50, label='N thực tế')
            ax3d.set_xlabel(var1); ax3d.set_ylabel(var2); ax3d.set_zlabel('N (hạt/giờ)')
            fixed_str = ", ".join([f"{k}={v:.1f}" for k, v in fixed_vals.items()])
            plt.title(f'N dự báo theo {var1} & {var2}\n(Fixed at mean: {fixed_str})')
            fig3d.colorbar(surf, shrink=0.5, aspect=5, label='N dự báo')
            ax3d.view_init(elev=20, azim=120)
            print(f">>> Kiểm tra đồ thị 3D của {var1} & {var2} xem có dạng lồi không.")
            plt.show()
    else:
        print("\n--- Bỏ qua vẽ đồ thị 3D (RUN_3D_PLOTS=False) ---")

else:
    print("\nKhông có kết quả phân tích hợp lệ để vẽ đồ thị.")


print("\n--- KẾT THÚC QUÁ TRÌNH PHÂN TÍCH ---")
