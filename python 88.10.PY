import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

# Example Data Table
data = np.array([
    [120, 0.1, 0.8, 1.81],
    [120, 0.07, 1.1, 1.62],
    [120, 0.07, 0.8, 1.51],
    [90, 0.1, 1.1, 1.87],
    [90, 0.1, 0.8, 1.69],
    [90, 0.07, 1.1, 1.41],
    [90, 0.085, 0.95, 1.21],
    [120, 0.085, 0.95, 1.29],
    [105, 0.07, 0.95, 0.89],
    [105, 0.1, 0.95, 1.48],
    [105, 0.085, 0.8, 1.11],
    [105, 0.085, 1.1, 1.19],
    [105, 0.085, 0.95, 0.74]
])

V = data[:, 0]
F = data[:, 1]
t = data[:, 2]
Ra = data[:, 3]

# Data Normalization
V_coded = (V - 105) / 15
F_coded = (F - 0.085) / 0.015
t_coded = (t - 0.95) / 0.15

# Creating a Design Matrix
X = np.column_stack((V_coded, F_coded, t_coded))
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Xây dựng mô hình
model = LinearRegression()
model.fit(X_poly, Ra)

# In phương trình hồi quy
coef = model.coef_
intercept = model.intercept_
print("Phương trình hồi quy:")
print(f"Ra = {intercept:.4f} + {coef[1]:.4f}X1 + {coef[2]:.4f}X2 + {coef[3]:.4f}X3 + "
      f"{coef[4]:.4f}X1² + {coef[5]:.4f}X1X2 + {coef[6]:.4f}X1X3 + "
      f"{coef[7]:.4f}X2² + {coef[8]:.4f}X2X3 + {coef[9]:.4f}X3²")

# Tính R-squared
R2 = model.score(X_poly, Ra)
print(f"\nHệ số R-squared: {R2:.4f}")

# Vẽ đồ thị
def plot_response_surface(x1, x2, param_name1, param_name2, fixed_param_name, fixed_value):
    fig = plt.figure(figsize=(15, 5))
    
    # 3D Surface plot
    ax1 = fig.add_subplot(131, projection='3d')
    X1, X2 = np.meshgrid(x1, x2)
    zeros = np.zeros_like(X1)
    
    if fixed_param_name == 't':
        X_plot = np.column_stack((X1.ravel(), X2.ravel(), zeros.ravel()))
    elif fixed_param_name == 'F':
        X_plot = np.column_stack((X1.ravel(), zeros.ravel(), X2.ravel()))
    else:  # fixed_param_name == 'V'
        X_plot = np.column_stack((zeros.ravel(), X1.ravel(), X2.ravel()))
    
    X_plot_poly = poly.transform(X_plot)
    Ra_plot = model.predict(X_plot_poly).reshape(X1.shape)
    
    surf = ax1.plot_surface(X1, X2, Ra_plot, cmap='viridis')
    ax1.set_xlabel(param_name1)
    ax1.set_ylabel(param_name2)
    ax1.set_zlabel('Ra (µm)')
    plt.colorbar(surf)
    
    # Contour plot
    ax2 = fig.add_subplot(132)
    cont = ax2.contour(X1, X2, Ra_plot, levels=15)
    ax2.clabel(cont, inline=True, fontsize=8)
    ax2.set_xlabel(param_name1)
    ax2.set_ylabel(param_name2)
    plt.colorbar(cont)
    
    # Individual effects
    ax3 = fig.add_subplot(133)
    x1_range = np.linspace(x1.min(), x1.max(), 100)
    x2_range = np.linspace(x2.min(), x2.max(), 100)
    
    # Plot effect of first parameter
    zeros = np.zeros_like(x1_range)
    if fixed_param_name == 't':
        X_effect1 = np.column_stack((x1_range, zeros, zeros))
    elif fixed_param_name == 'F':
        X_effect1 = np.column_stack((x1_range, zeros, zeros))
    else:
        X_effect1 = np.column_stack((zeros, x1_range, zeros))
    
    X_effect1_poly = poly.transform(X_effect1)
    Ra_effect1 = model.predict(X_effect1_poly)
    ax3.plot(x1_range, Ra_effect1, 'b-', label=param_name1)
    
    # Plot effect of second parameter
    if fixed_param_name == 't':
        X_effect2 = np.column_stack((zeros, x2_range, zeros))
    elif fixed_param_name == 'F':
        X_effect2 = np.column_stack((zeros, zeros, x2_range))
    else:
        X_effect2 = np.column_stack((zeros, zeros, x2_range))
    
    X_effect2_poly = poly.transform(X_effect2)
    Ra_effect2 = model.predict(X_effect2_poly)
    ax3.plot(x2_range, Ra_effect2, 'r-', label=param_name2)
    
    ax3.set_xlabel('Parameter value')
    ax3.set_ylabel('Ra (µm)')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()

# Vẽ các đồ thị
V_range = np.linspace(-1, 1, 50)
F_range = np.linspace(-1, 1, 50)
t_range = np.linspace(-1, 1, 50)

plot_response_surface(V_range, F_range, 'V', 'F', 't', 0)
plot_response_surface(V_range, t_range, 'V', 't', 'F', 0)
plot_response_surface(F_range, t_range, 'F', 't', 'V', 0)

# Tìm giá trị tối ưu
from scipy.optimize import minimize

def objective(X):
    X_reshaped = X.reshape(1, -1)
    X_poly = poly.transform(X_reshaped)
    return model.predict(X_poly)[0]

constraints = (
    {'type': 'ineq', 'fun': lambda x: x[0] + 1},  # V >= -1
    {'type': 'ineq', 'fun': lambda x: 1 - x[0]},  # V <= 1
    {'type': 'ineq', 'fun': lambda x: x[1] + 1},  # F >= -1
    {'type': 'ineq', 'fun': lambda x: 1 - x[1]},  # F <= 1
    {'type': 'ineq', 'fun': lambda x: x[2] + 1},  # t >= -1
    {'type': 'ineq', 'fun': lambda x: 1 - x[2]}   # t <= 1
)

result = minimize(objective, x0=[0, 0, 0], constraints=constraints)

# Chuyển đổi giá trị tối ưu về thang đo thực
V_opt = result.x[0] * 15 + 105
F_opt = result.x[1] * 0.015 + 0.085
t_opt = result.x[2] * 0.15 + 0.95
Ra_opt = objective(result.x)

print("\nGiá trị tối ưu:")
print(f"V = {V_opt:.2f} m/ph")
print(f"F = {F_opt:.4f} mm/ph")
print(f"t = {t_opt:.2f} mm")
print(f"Ra = {Ra_opt:.4f} µm")

# Phân tích ảnh hưởng của các thông số
print("\nPhân tích ảnh hưởng:")
print(f"Ảnh hưởng của V: {abs(coef[1]):.4f}")
print(f"Ảnh hưởng của F: {abs(coef[2]):.4f}")
print(f"Ảnh hưởng của t: {abs(coef[3]):.4f}")
print(f"Ảnh hưởng của V²: {abs(coef[4]):.4f}")
print(f"Ảnh hưởng của VF: {abs(coef[5]):.4f}")
print(f"Ảnh hưởng của Vt: {abs(coef[6]):.4f}")
print(f"Ảnh hưởng của F²: {abs(coef[7]):.4f}")
print(f"Ảnh hưởng của Ft: {abs(coef[8]):.4f}")
print(f"Ảnh hưởng của t²: {abs(coef[9]):.4f}")

# Tính tổng các hệ số ảnh hưởng (giá trị tuyệt đối)
total_effect = (abs(coef[1]) + abs(coef[2]) + abs(coef[3]) + 
               abs(coef[4]) + abs(coef[7]) + abs(coef[9]))

# Tính phần trăm ảnh hưởng của từng yếu tố
print("\nPhần trăm ảnh hưởng của các yếu tố:")
print(f"V (tuyến tính): {abs(coef[1])/total_effect*100:.2f}%")
print(f"F (tuyến tính): {abs(coef[2])/total_effect*100:.2f}%")
print(f"t (tuyến tính): {abs(coef[3])/total_effect*100:.2f}%")
print(f"V² (phi tuyến): {abs(coef[4])/total_effect*100:.2f}%")
print(f"F² (phi tuyến): {abs(coef[7])/total_effect*100:.2f}%")
print(f"t² (phi tuyến): {abs(coef[9])/total_effect*100:.2f}%")

# Equation of Individual Effects of Each Factor
print("\nEquation of Individual Effects:")

# Cho V (khi F=0, t=0)
print(f"Ra(V) = {intercept:.4f} + {coef[1]:.4f}V + {coef[4]:.4f}V²")

# Cho F (khi V=0, t=0)
print(f"Ra(F) = {intercept:.4f} + {coef[2]:.4f}F + {coef[7]:.4f}F²")

# Cho t (khi V=0, F=0)
print(f"Ra(t) = {intercept:.4f} + {coef[3]:.4f}t + {coef[9]:.4f}t²")

# Tính độ ảnh hưởng tổng hợp của từng yếu tố
V_total = abs(coef[1]) + abs(coef[4])
F_total = abs(coef[2]) + abs(coef[7])
t_total = abs(coef[3]) + abs(coef[9])
total = V_total + F_total + t_total

print("\nTotal Effect Analysis:")
print(f"V (tổng): {V_total/total*100:.2f}%")
print(f"F (tổng): {F_total/total*100:.2f}%")
print(f"t (tổng): {t_total/total*100:.2f}%")

# After model fitting, add this code:

# Calculate predictions and errors
y_pred = model.predict(X_poly)
mse = np.mean((Ra - y_pred) ** 2)
r2 = model.score(X_poly, Ra)

# Print header statistics
print("Linear Regression:")
print(f"R² = {r2:.4f}")
print(f"MSE = {mse:.6f}\n")

# Create results table
print("Prediction Results:")
print(f"{'Index':>5} {'Ra_actual':>10} {'Ra_predicted':>12} {'Difference':>12}")
print("-" * 42)

for i in range(len(Ra)):
    difference = abs(Ra[i] - y_pred[i])
    print(f"{i+1:>5} {Ra[i]:>10.6f} {y_pred[i]:>12.6f} {difference:>12.6f}")

# Optional: Create a pandas DataFrame for nicer formatting
import pandas as pd

results_df = pd.DataFrame({
    'Ra_actual': Ra,
    'Ra_predicted': y_pred,
    'Difference': abs(Ra - y_pred)
})

# Save to CSV or display in notebook
results_df.to_csv('prediction_results.csv', float_format='%.6f', index=True)
