# -*- bai bao 88.05: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
from scipy.optimize import minimize
import warnings

# === Cấu hình ===
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")
DEGREE = 2          # Bậc của đa thức hồi quy
NUM_POINTS_3D = 30  # Số điểm lưới cho đồ thị 3D
NUM_POINTS_2D = 50  # Số điểm cho đồ thị 2D
RUN_3D_PLOTS = True # Đặt thành False để bỏ qua vẽ 3D

print("--- BẮT ĐẦU QUÁ TRÌNH PHÂN TÍCH (DỮ LIỆU MỚI CUNG CẤP) ---")

# === Phần 1: Nhập liệu ===
print("\n--- 1. Nhập Dữ liệu Mới ---")
data = {
    'V':  [70, 90, 70, 90, 70, 90, 70, 90, 80, 80, 80, 80, 70, 90, 70, 90, 80, 80, 80, 80, 80, 80, 80, 80, 80],
    'Fz': [0.02, 0.02, 0.04, 0.04, 0.03, 0.03, 0.03, 0.03, 0.02, 0.04, 0.02, 0.04, 0.03, 0.03, 0.03, 0.03, 0.02, 0.04, 0.02, 0.04, 0.03, 0.03, 0.03, 0.03, 0.03],
    'ap': [0.25, 0.25, 0.25, 0.25, 0.2, 0.2, 0.3, 0.3, 0.2, 0.2, 0.3, 0.3, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.2, 0.3, 0.2, 0.3, 0.25],
    'ae': [2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.0, 2.0, 3.0, 3.0, 2.0, 2.0, 3.0, 3.0, 2.0, 2.0, 3.0, 3.0, 2.5],
    'Ra': [0.38, 0.42, 0.45, 0.48, 0.40, 0.43, 0.42, 0.45, 0.39, 0.46, 0.41, 0.47, 0.41, 0.44, 0.42, 0.45, 0.40, 0.47, 0.41, 0.48, 0.42, 0.44, 0.43, 0.41, 0.35] 
}
df = pd.DataFrame(data)
print("Dữ liệu sử dụng:")
print(df.head())
print("-" * 40)

# Xác định biến đầu vào và mục tiêu
features = ['V', 'Fz', 'ap', 'ae']
feature_indices = {name: i for i, name in enumerate(features)}
X = df[features]
y = df['Ra'] # <--- Sử dụng Ra làm biến mục tiêu

# === Phần 2: Tiền xử lý, Mô hình hóa và Đánh giá ===
print("\n--- 2. Tiền xử lý (Centering) ---")
# Sử dụng giá trị trung tâm Fixed 
# center_values = {f: df[f].mean() for f in features} # Tính mean từ dữ liệu mới
center_values = {'V': 80, 'Fz': 0.03, 'ap': 0.25, 'ae': 2.5} # Hoặc giữ nguyên giá trị trung tâm cũ
print("Giá trị trung tâm dùng để centering:", center_values)
X_centered = X.copy()
for col, center_val in center_values.items():
    X_centered[col] = X_centered[col] - center_val
print("-" * 40)

print("\n--- 3. Tạo đặc trưng đa thức bậc 2 ---")
poly = PolynomialFeatures(degree=DEGREE, include_bias=False)
X_poly = poly.fit_transform(X_centered)
poly_feature_names = poly.get_feature_names_out(features)
print("-" * 40)

print("\n--- 4. Xây dựng và Huấn luyện Mô hình Hồi quy ---")
model = LinearRegression()
model.fit(X_poly, y)
print("Huấn luyện xong mô hình.")
print("-" * 40)

# -- Yêu cầu 1: Phương trình hồi quy --
print("\n--- 5. Phương trình hồi quy bậc 2 ---")
equation = f"Ra = {model.intercept_:.6f}"
for coef, name in zip(model.coef_, poly_feature_names):
    sign = "+" if coef >= 0 else "-"
    equation += f" {sign} {abs(coef):.6f}*{name}"
print(equation); print("-" * 40)

# -- Yêu cầu 2: Đánh giá mô hình --
print("\n--- 6. Đánh giá mô hình ---")
y_pred = model.predict(X_poly)
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
print(f"R-squared (R²): {r2:.4f}"); print(f"RMSE: {rmse:.4f}"); print("-" * 40)

# -- Yêu cầu 3: So sánh Ra thực tế và Ra dự báo --
print("\n--- 7. So sánh Ra thực tế và Ra dự báo ---")
df['Ra_predict'] = y_pred
print(df[['V', 'Fz', 'ap', 'ae', 'Ra', 'Ra_predict']].round(4)); print("-" * 40)

# -- Yêu cầu 4: Tìm điểm Optimal (Ra nhỏ nhất) --
print("\n--- 8. Tìm điểm Optimal ---")
def objective_func(x):
    p_cent = np.array([x[i] - center_values[f] for i, f in enumerate(features)])
    p_poly = poly.transform(p_cent.reshape(1, -1))
    return model.predict(p_poly)[0]
bounds = [(df[f].min(), df[f].max()) for f in features]
opt_result = minimize(objective_func, X.mean().values, method='L-BFGS-B', bounds=bounds)
optimal_point = opt_result.x if opt_result.success else None
min_ra_pred = opt_result.fun if opt_result.success else None
optimal_point_found = opt_result.success
if optimal_point_found:
    print(f"Giá trị Ra_predict nhỏ nhất: {min_ra_pred:.4f} µm")
    print("Tại bộ thông số (Experimental Value):")
    for i, f in enumerate(features): print(f"- {f}: {optimal_point[i]:.3f}")
    # Kiểm tra trùng lặp thông số đầu vào
    if any(np.allclose(row.values, optimal_point, atol=1e-2) for _, row in X.iterrows()): print("!!! Cảnh báo: Điểm Optimal rất gần với điểm TN.")
    else: print("Thông số Optimal không trùng/rất gần với các điểm TN.")
    # Kiểm tra trùng lặp giá trị Ra Optimal với Ra gốc
    if any(np.isclose(min_ra_pred, ra_orig, atol=1e-4) for ra_orig in y): print(f"!!! Cảnh báo: Ra Optimal ({min_ra_pred:.4f}) rất gần với giá trị Ra gốc đo được.")
    else: print("Giá trị Ra Optimal dự đoán không trùng/rất gần với các giá trị Ra gốc đo được.")
else: print("Không tìm thấy điểm Optimal:", opt_result.message)
print("-" * 40)

# === Phần 3: Trực quan hóa ===
# -- Yêu cầu 5: Đồ thị 3D --
if RUN_3D_PLOTS:
    print("\n--- 9. Vẽ đồ thị 3D ---")
    fixed_vals_cent = {f: 0.0 for f in features}
    for var1, var2 in itertools.combinations(features, 2):
        fixed_vars = [f for f in features if f not in [var1, var2]]
        r1 = np.linspace(df[var1].min(), df[var1].max(), NUM_POINTS_3D); r2 = np.linspace(df[var2].min(), df[var2].max(), NUM_POINTS_3D)
        g1, g2 = np.meshgrid(r1, r2); Z = np.zeros_like(g1)
        for i, j in np.ndindex(g1.shape):
            p_cent = fixed_vals_cent.copy(); p_cent[var1] = g1[i, j] - center_values[var1]; p_cent[var2] = g2[i, j] - center_values[var2]
            ord_p_cent = [p_cent[f] for f in features]
            Z[i, j] = model.predict(poly.transform(np.array([ord_p_cent])))[0]
        fig = plt.figure(figsize=(9, 7)); ax = fig.add_subplot(111, projection='3d'); surf = ax.plot_surface(g1, g2, Z, cmap='viridis', edgecolor='none')
        ax.set_xlabel(var1); ax.set_ylabel(var2); ax.set_zlabel('Ra_predict (µm)')
        fixed_str = ", ".join([f"{v}={center_values[v]:.2f}" for v in fixed_vars]); plt.title(f'Effect of {var1} & {var2}\n(Fixed : {fixed_str})')
        fig.colorbar(surf, shrink=0.5, aspect=5, label='Ra_predict (µm)'); plt.tight_layout(); plt.show()
    print("-" * 40)

# -- Yêu cầu 6: Đồ thị 2D và Xếp hạng Effect of --
print("\n--- 10&11. Phân tích Effect of 2D và Xếp hạng ---")
ra_effects = {} # Dictionary để lưu kết quả phân tích ảnh hưởng

# Lặp qua từng thông số đầu vào
for feature_vary in features:
    # Tạo dải giá trị cho thông số đang xét
    vary_range = np.linspace(df[feature_vary].min(), df[feature_vary].max(), NUM_POINTS_2D)
    pred_ra = np.zeros(NUM_POINTS_2D) # Mảng để lưu Ra_predict tương ứng

    # Xác định các giá trị cố định (là giá trị trung tâm) cho các thông số khác
    fixed_vals = {f: center_values[f] for f in features if f != feature_vary}

    # Tính Ra_predict cho từng điểm trong dải giá trị của thông số đang xét
    for i, val in enumerate(vary_range):
        # Tạo điểm dữ liệu đầu vào: kết hợp giá trị đang xét và các giá trị cố định
        p = {**fixed_vals, feature_vary: val}
        # Centering điểm dữ liệu này
        p_cent = np.array([p[f] - center_values[f] for f in features])
        # Chuyển đổi sang dạng đa thức
        p_poly = poly.transform(p_cent.reshape(1, -1))
        # Dự đoán Ra bằng mô hình
        pred_ra[i] = model.predict(p_poly)[0]

    # Lưu kết quả: dải giá trị, Ra dự đoán, và độ chênh lệch (ảnh hưởng)
    ra_effects[feature_vary] = {
        'range': vary_range,
        'pred': pred_ra,
        'delta': pred_ra.max() - pred_ra.min() # Tính delta Ra
    }

# Sắp xếp các thông số theo mức độ ảnh hưởng (delta) giảm dần
sorted_influence = sorted(ra_effects.items(), key=lambda item: item[1]['delta'], reverse=True)

# --- Vẽ đồ thị 2D ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
print("Đang vẽ đồ thị 2D tổng hợp...")

for idx, (feature_vary, effect_data) in enumerate(sorted_influence):
    ax = axes[idx]
    fixed_features = [f for f in features if f != feature_vary]
    # Vẽ đường biểu diễn ảnh hưởng
    ax.plot(effect_data['range'], effect_data['pred'], marker='.', markersize=5, linestyle='-')
    ax.set_xlabel(f"{feature_vary} (Experimental Value)")
    ax.set_ylabel("Ra_predict (µm)")
    fixed_str = ", ".join([f"{v}={center_values[v]:.2f}" for v in fixed_features])
    ax.set_title(f"Effect of {feature_vary}\n(Fixed : {fixed_str[:40]}...)")
    ax.grid(True)
    # Vẽ đường thẳng đứng tại giá trị tối ưu (nếu có)
    if optimal_point_found:
        opt_val = optimal_point[feature_indices[feature_vary]]
        ax.axvline(x=opt_val, color='r', linestyle='--', lw=1.5, label=f'Optimal {feature_vary}={opt_val:.3f}')
        ax.legend(fontsize='small')

# Xóa các trục thừa (nếu có ít hơn 4 thông số)
for i in range(len(sorted_influence), len(axes)):
    fig.delaxes(axes[i])

fig.suptitle("Phân tích Effect of từng yếu tố đến Ra_predict", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# --- In xếp hạng ảnh hưởng ---
print("\nTổng kết mức độ Effect of (dựa trên khoảng biến thiên Ra_predict):")
for rank, (feature, effect_data) in enumerate(sorted_influence, 1):
    print(f"{rank}. {feature}: Khoảng biến thiên Ra ≈ {effect_data['delta']:.4f} µm")
print("-" * 40)

