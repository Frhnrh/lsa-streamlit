import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import io

# ===============================
# STREAMLIT INTERFACE WRAPPER
# ===============================
st.set_page_config(page_title="Least Squares Adjustment (LSA)", layout="wide")
st.title("📐 Least Squares Adjustment (LSA) Elevation Analyzer")
st.markdown("Upload CSV or enter data manually to perform LSA and plot results.")

uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

# Input containers
input_method = "manual"
if uploaded_file:
    input_method = "csv"

unknown_points = []
known_points = {}
observations = []

if input_method == "csv":
    section = None
    content = uploaded_file.read().decode("utf-8")
    reader = csv.reader(io.StringIO(content))
    for row in reader:
        if not row: continue
        if row[0].startswith('#'):
            if 'UNKNOWN' in row[0].upper():
                section = 'unknown'
            elif 'BENCHMARK' in row[0].upper():
                section = 'benchmark'
            elif 'OBSERVATION' in row[0].upper():
                section = 'obs'
            continue

        if section == 'unknown':
            unknown_points = [x.strip() for x in row if x.strip()]
        elif section == 'benchmark' and len(row) >= 2:
            known_points[row[0].strip()] = float(row[1])
        elif section == 'obs' and len(row) >= 3:
            observations.append((row[0].strip(), row[1].strip(), float(row[2])))

else:
    unknown_str = st.text_input("Enter unknown points (e.g., A,B,C):")
    unknown_points = [pt.strip() for pt in unknown_str.split(",") if pt.strip()]

    st.markdown("**Enter Benchmarks (Known Point Name & Value)**")
    bm_data = st.text_area("Example: BM1 100.00\nBM2 105.00")
    for line in bm_data.splitlines():
        try:
            name, val = line.strip().split()
            known_points[name] = float(val)
        except:
            continue

    st.markdown("**Enter Observations (From To Δh)**")
    obs_data = st.text_area("Example: A B 1.25\nB C -0.50")
    for line in obs_data.splitlines():
        try:
            frm, to, dh = line.strip().split()
            observations.append((frm, to, float(dh)))
        except:
            continue

if st.button("Run LSA") and unknown_points and known_points and observations:

    # STEP 1: DIMENSIONS
    u = len(unknown_points)
    n = len(observations)
    r = n - u
    point_index = {pt: i for i, pt in enumerate(unknown_points)}

    if r <= 0:
        st.error("LSA cannot be performed. Redundancy (r) ≤ 0")
        st.stop()

    # STEP 2 & 3: Matrix A and L
    A = np.zeros((n, u))
    L = np.zeros((n, 1))

    for i, (frm, to, dh) in enumerate(observations):
        if frm in point_index:
            A[i, point_index[frm]] = -1
        elif frm in known_points:
            L[i] += known_points[frm]

        if to in point_index:
            A[i, point_index[to]] = 1
        elif to in known_points:
            L[i] -= known_points[to]

        L[i] += dh

    # STEP 4: LSA
    AT = A.T
    N = AT @ A
    U = AT @ L
    X = np.linalg.inv(N) @ U
    V = A @ X - L
    sigma0_squared = (V.T @ V)[0, 0] / r
    Cov = sigma0_squared * np.linalg.inv(N)
    std_dev = np.sqrt(np.diag(Cov))

    # STEP 5: Display Results
    
    st.subheader("📊 Results")
    
    # Dimensions
    st.markdown("#### 🔢 Dimensions")
    st.write(f"Number of observations (n): {n}")
    st.write(f"Number of unknowns (u): {u}")
    st.write(f"Redundancy (r): {r}")
    
    # Matrix L
    st.markdown("#### 📥 Matrix L")
    st.text(L)
    
    # Matrix A
    st.markdown("#### 🧱 Matrix A")
    st.text(A)
    
    # Adjusted Parameters
    st.markdown("#### 📐 Adjusted Parameters")
    for i, pt in enumerate(unknown_points):
        st.write(f"{pt} = {X[i, 0]:.4f} m")
    
    # Residuals
    st.markdown("#### 📉 Residuals (v)")
    residual_str = "\n".join([f"v{i+1} = {v[0]:.5f}" for i, v in enumerate(V)])
    st.text(residual_str)
    
    # Aposteriori Variance
    st.markdown("#### 📈 Aposteriori Variance (σ₀²)")
    st.write(f"σ₀² = {sigma0_squared:.6f}")
    
    # Covariance Matrix
    st.markdown("#### 🧮 Covariance Matrix")
    st.text(Cov)
    
    # Standard Deviations
    st.markdown("#### 📏 Standard Deviations")
    for i, pt in enumerate(unknown_points):
        st.write(f"σ({pt}) = {std_dev[i]:.6f} m")
    
    # Final Result ± σ
    st.markdown("#### ✅ Final Adjusted Parameters ± Std Dev")
    for i, pt in enumerate(unknown_points):
        st.write(f"{pt} = {X[i, 0]:.4f} ± {std_dev[i]:.4f} m")
    
    # Nota
    st.markdown("#### 📝 Note")
    st.info("The closer the standard deviation is to 0, the higher the accuracy of the adjusted data.")


    # STEP 6: Plot Elevation Profile
    st.subheader("📈 Elevation Profile")
    points = {pt: X[i, 0] for i, pt in enumerate(unknown_points)}
    points.update(known_points)
    plot_order = list(dict.fromkeys(list(known_points.keys()) + unknown_points))
    elevations = [points[pt] for pt in plot_order]
    symbols = {pt: "o" if pt in unknown_points else "s" for pt in plot_order}
    colors = {pt: "royalblue" if pt in unknown_points else "firebrick" for pt in plot_order}

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, pt in enumerate(plot_order):
        ax.scatter(i, elevations[i], marker=symbols[pt], s=100, color=colors[pt])
        ax.text(i, elevations[i] + 0.1, f"{elevations[i]:.3f} m", ha='center', fontsize=9)
    for i in range(len(plot_order) - 1):
        ax.annotate("", xy=(i + 1, elevations[i + 1]), xytext=(i, elevations[i]),
                    arrowprops=dict(arrowstyle="->", color='gray', lw=1.2))
    ax.set_title("Elevation Profile (LSA)", fontsize=13, weight='bold')
    ax.set_xticks(range(len(plot_order)))
    ax.set_xticklabels(plot_order)
    ax.set_ylabel("Elevation (m)")
    ax.set_xlabel("Point")
    ax.grid(True)
    ax.legend(handles=[
        plt.Line2D([0], [0], marker='s', color='w', label='Benchmark (Known)', markerfacecolor='firebrick', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='TBM (Adjusted)', markerfacecolor='royalblue', markersize=10)
    ])
    st.pyplot(fig)

    # STEP 7: Plot Residuals
    st.subheader("🔍 Residuals and Outlier Detection")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    residual_values = [v[0] for v in V]
    ax2.bar(range(1, len(residual_values) + 1), residual_values, color='skyblue')
    ci_limit = 2.576 * np.sqrt(sigma0_squared)
    ax2.axhline(ci_limit, color='red', linestyle='--', label='99% CI Upper Limit')
    ax2.axhline(-ci_limit, color='red', linestyle='--', label='99% CI Lower Limit')
    ax2.set_title("Residuals and Outlier Detection", fontsize=13, weight='bold')
    ax2.set_xlabel("Observation")
    ax2.set_ylabel("Residual (m)")
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)

    st.success("✅ LSA Completed Successfully")
