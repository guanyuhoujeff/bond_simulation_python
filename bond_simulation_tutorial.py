# -*- coding: utf-8 -*-
"""
債券利率蒙地卡羅模擬教學腳本 (Bond Simulation Tutorial)

本腳本旨在教學演示如何使用 Python 進行債券利率的蒙地卡羅模擬。
程式碼中的變數名稱、公式與註解都對標 'equation.md' 文件，以利於學習與對照。

包含四種利率模型：
1. Vasicek 模型：用於模擬無風險利率 (drf)，具有均值回歸特性。
2. Cox-Ingersoll-Ross (CIR) 模型：Vasicek 的變體，確保利率為正。
3. 幾何布朗運動 (GBM) 模型：一個常用於股價的簡化利率模型。
4. 有風險利率模型：在無風險利率上疊加一個隨機的信用利差。

並根據模擬出的利率路徑，使用「修正存續期間」與「價格曲度」來近似計算債券價格的變化。
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# --- 1. 利率模擬函數 ---

def simulate_vasicek(r_0, k, theta, sigma, T, dt, n_simulations, dW_t):
    """
    使用 Vasicek 模型模擬利率路徑。

    根據隨機微分方程 (SDE):
    dr_t = k(θ - r_t)dt + σdW_t

    參數說明 (對標 equation.md):
    :param r_0: float, 初始利率 (r_t 在 t=0 的值)。
    :param k: float, 均值回歸速度 (kappa)，代表利率回復到長期均值的速度。
    :param theta: float, 長期平均利率或均衡水平 (θ)。
    :param sigma: float, 波動率 (σ)，代表利率隨機波動的幅度。
    :param T: float, 總模擬時長(年)。
    :param dt: float, 每個時間步長(年)。
    :param n_simulations: int, 模擬路徑的數量。
    :param dW_t: np.ndarray, 預先生成的維納過程增量 (dW_t)，代表隨機衝擊。
                 其維度應為 (n_simulations, num_steps)。
    :return: np.ndarray, 模擬的利率路徑，形狀為 (n_simulations, num_steps + 1)。
    """
    num_steps = int(T / dt)
    # 初始化利率路徑數組，第一欄設為初始利率 r_0
    r_paths = np.zeros((n_simulations, num_steps + 1))
    r_paths[:, 0] = r_0

    # 迭代計算每一步的利率
    for t in range(1, num_steps + 1):
        # 根據 Vasicek 公式的離散化形式進行計算
        # r(t) = r(t-1) + k * (theta - r(t-1)) * dt + sigma * dW_t
        # dW_t 是一個服從 N(0, sqrt(dt)) 的隨機變數，我們在傳入前已處理好
        r_paths[:, t] = r_paths[:, t-1] + k * (theta - r_paths[:, t-1]) * dt + sigma * dW_t[:, t-1]
        # 確保利率不會變成不切實際的負值
        r_paths[:, t] = np.maximum(0.0001, r_paths[:, t])
        
    return r_paths

def simulate_cir(r_0, k, theta, sigma, T, dt, n_simulations, dW_t):
    """
    使用 Cox-Ingersoll-Ross (CIR) 模型模擬利率路徑。

    根據隨機微分方程 (SDE):
    dr_t = k(θ - r_t)dt + σ√r_t dW_t
    此模型確保利率恆為正值（在 2kθ > σ² 的條件下）。

    參數說明 (對標 equation.md):
    :param r_0: float, 初始利率 (r_t 在 t=0 的值)。
    :param k: float, 均值回歸速度 (kappa_cir)。
    :param theta: float, 長期平均利率 (theta_cir)。
    :param sigma: float, 波動率 (sigma_cir)。
    :param T: float, 總模擬時長(年)。
    :param dt: float, 每個時間步長(年)。
    :param n_simulations: int, 模擬路徑的數量。
    :param dW_t: np.ndarray, 預先生成的維納過程增量 (dW_t)。
    :return: np.ndarray, 模擬的利率路徑。
    """
    num_steps = int(T / dt)
    r_paths = np.zeros((n_simulations, num_steps + 1))
    r_paths[:, 0] = r_0

    for t in range(1, num_steps + 1):
        # 為了避免對負數開根號，先取 r(t-1) 和 0 之間的最大值
        sqrt_r = np.sqrt(np.maximum(0, r_paths[:, t-1]))
        
        # 根據 CIR 公式的離散化形式進行計算
        # r(t) = r(t-1) + k * (theta - r(t-1)) * dt + sigma * sqrt(r(t-1)) * dW_t
        r_paths[:, t] = r_paths[:, t-1] + k * (theta - r_paths[:, t-1]) * dt + sigma * sqrt_r * dW_t[:, t-1]
        # 再次確保利率不會變為負數
        r_paths[:, t] = np.maximum(0.0001, r_paths[:, t])
        
    return r_paths

def simulate_gbm(r_0, mu, sigma, T, dt, n_simulations, dW_t):
    """
    使用幾何布朗運動 (GBM) 模型模擬利率路徑。

    根據隨機微分方程 (SDE):
    dr_t = μr_t dt + σr_t dW_t
    其離散化形式為 r_t = r_{t-1} * exp((μ - 0.5σ²)Δt + σdW_t)。

    參數說明 (對標 equation.md):
    :param r_0: float, 初始利率 (r_t 在 t=0 的值)。
    :param mu: float, 長期漂移趨勢 (μ, mu_r)。
    :param sigma: float, 年化波動率 (σ, sigma_r)。
    :param T: float, 總模擬時長(年)。
    :param dt: float, 每個時間步長(年)。
    :param n_simulations: int, 模擬路徑的數量。
    :param dW_t: np.ndarray, 預先生成的維納過程增量 (dW_t)。
    :return: np.ndarray, 模擬的利率路徑。
    """
    num_steps = int(T / dt)
    r_paths = np.zeros((n_simulations, num_steps + 1))
    r_paths[:, 0] = r_0

    for t in range(1, num_steps + 1):
        # 根據 GBM 公式的離散化形式進行計算
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * dW_t[:, t-1]
        r_paths[:, t] = r_paths[:, t-1] * np.exp(drift + diffusion)
        # 確保利率不會變成不切實際的負值
        r_paths[:, t] = np.maximum(0.0001, r_paths[:, t])
        
    return r_paths

def simulate_risky_rate(drf_paths, dW_f, initial_spread, sigma_s, rho, T, dt, n_simulations):
    """
    模擬有風險利率 (drs)，即在無風險利率(drf)上疊加信用利差(spread)。

    公式:
    drs_t = drf_t + spread_t
    d(spread_t) = σ_s dW_s
    dW_s = ρ * dW_f + √(1 - ρ²) * dZ_t  (其中 dZ_t 是獨立的維納過程)

    參數說明 (對標 equation.md):
    :param drf_paths: np.ndarray, 已模擬好的無風險利率路徑 (drf_t)。
    :param dW_f: np.ndarray, 生成 drf_paths 所使用的隨機衝擊 (dW_f)。
    :param initial_spread: float, 初始信用利差。
    :param sigma_s: float, 信用利差的波動率 (σ_s, sigma_spread)。
    :param rho: float, drf 與 spread 變動之間的相關係數 (ρ, correlation)。
    :param T: float, 總模擬時長(年)。
    :param dt: float, 每個時間步長(年)。
    :param n_simulations: int, 模擬路徑的數量。
    :return: tuple (np.ndarray, np.ndarray), 分別為模擬的有風險利率路徑(drs)和信用利差路徑(spread)。
    """
    num_steps = int(T / dt)
    
    # 1. 生成一個獨立的隨機衝擊 dZ_t
    dZ_t = np.random.normal(0, 1, (n_simulations, num_steps)) * np.sqrt(dt)
    
    # 2. 根據相關係數 rho，合成信用利差的隨機衝擊 dW_s
    dW_s = rho * dW_f + np.sqrt(1 - rho**2) * dZ_t
    
    # 3. 模擬信用利差的路徑
    spread_paths = np.zeros((n_simulations, num_steps + 1))
    spread_paths[:, 0] = initial_spread
    for t in range(1, num_steps + 1):
        # 根據 d(spread_t) = σ_s dW_s 進行計算
        spread_paths[:, t] = spread_paths[:, t-1] + sigma_s * dW_s[:, t-1]
        # 確保利差為正
        spread_paths[:, t] = np.maximum(0.0001, spread_paths[:, t])
        
    # 4. 有風險利率 = 無風險利率 + 信用利差
    drs_paths = drf_paths + spread_paths
    
    return drs_paths, spread_paths

# --- 2. 債券價格計算函數 ---

def calculate_price_paths(r_paths, D_mod, convT, initial_price=100.0):
    """
    使用修正存續期間(D_mod)和價格曲度(convT)來近似計算債券價格路徑。

    近似公式:
    ΔP/P ≈ -D_mod * Δy + (C_mod / 2) * (Δy)²
    其中 y 是殖利率(yield)，此處用模擬的利率 r_paths 作為替代。

    參數說明:
    :param r_paths: np.ndarray, 模擬的利率路徑 (y)。
    :param D_mod: float, 債券的修正存續期間 (Modified Duration)。
    :param convT: float, 來自資料庫的價格曲度值。
                   在我們的系統中，convT 的定義是 (P * C_mod / 2)，
                   其中 P 是初始價格，C_mod 是標準的曲度定義。
    :param initial_price: float, 債券的初始價格，預設為100。
    :return: np.ndarray, 模擬的債券價格路徑。
    """
    num_steps = r_paths.shape[1] - 1
    price_paths = np.zeros_like(r_paths)
    price_paths[:, 0] = initial_price
    
    # 從 convT 反解出公式中需要的 (C_mod / 2) 項。
    # 根據定義 convT = initial_price * C_mod / 2
    # 因此 C_mod / 2 = convT / initial_price
    # 這個因子在整個模擬中被視為常數。
    convexity_factor = convT / initial_price
    
    for t in range(1, num_steps + 1):
        # 計算兩期之間的利率變化 Δy
        delta_y = r_paths[:, t] - r_paths[:, t-1]
        
        # 抓取前一期的價格 P(t-1)
        prev_price = price_paths[:, t-1]
        
        # 1. 計算價格變動百分比 (ΔP/P)
        # ΔP/P = -D_mod * Δy + (C_mod / 2) * (Δy)²
        price_change_percentage = -D_mod * delta_y + convexity_factor * (delta_y ** 2)
        
        # 2. 計算新價格 P(t) = P(t-1) * (1 + ΔP/P)
        price_paths[:, t] = prev_price * (1 + price_change_percentage)
        
    return price_paths

# --- 3. 主執行區塊 ---

if __name__ == "__main__":
    
    # --- A. 設定模擬參數 ---
    
    # 債券基本資料 (此處為範例，實際應用中從資料庫讀取)
    bond_data = {
        'name': 'CGB10Y',
        'avgYld': 2.2,      # 平均殖利率 (%)
        'mdurT': 8.9,       # 修正存續期間 (D_mod)
        'convT': 45.5,      # 價格曲度 (P * C_mod / 2)
        'maturity': datetime(2034, 5, 15)
    }
    
    # 模擬通用參數
    n_simulations = 100      # 模擬次數
    dt = 1/252               # 時間步長 (年)，假設一年252個交易日
    
    # 計算剩餘到期年限，作為模擬總時長 T
    time_to_maturity = (bond_data['maturity'] - datetime.now()).days / 365.25
    T = max(time_to_maturity, 0.1) # 確保至少模擬一小段時間
    
    num_steps = int(T / dt)
    time_points = np.linspace(0, T, num_steps + 1) # 模擬的時間點數列
    
    # 初始利率 (將百分比轉為小數)
    r_0 = bond_data['avgYld'] / 100.0

    # --- B. 設定各模型參數 (將百分比轉為小數) ---

    # 1. Vasicek & CIR 模型參數
    k_vasicek = 0.3
    theta_vasicek = 2.2 / 100.0
    sigma_vasicek = 0.5 / 100.0
    
    k_cir = 0.2
    theta_cir = 2.5 / 100.0
    sigma_cir = 0.4 / 100.0

    # 2. GBM 模型參數
    mu_r = 0.05 / 100.0
    sigma_r = 0.6 / 100.0

    # 3. 有風險利率模型參數
    initial_spread = 0.8 / 100.0
    sigma_s = 0.3 / 100.0
    rho = 0.6

    # --- C. 生成隨機衝擊 ---
    # 為每個需要獨立隨機性的模型預先生成維納過程增量 dW_t
    # dW_t ~ N(0, dt)  等價於  sqrt(dt) * N(0, 1)
    np.random.seed(42) # 設定隨機種子以確保結果可重現
    dW_vasicek = np.random.normal(0, 1, (n_simulations, num_steps)) * np.sqrt(dt)
    dW_cir = np.random.normal(0, 1, (n_simulations, num_steps)) * np.sqrt(dt)
    dW_gbm = np.random.normal(0, 1, (n_simulations, num_steps)) * np.sqrt(dt)
    
    # --- D. 執行模擬 ---

    print("開始執行利率模擬...")
    # 1. 模擬無風險利率 drf (Vasicek)
    drf_paths = simulate_vasicek(r_0, k_vasicek, theta_vasicek, sigma_vasicek, T, dt, n_simulations, dW_vasicek)
    
    # 2. 模擬有風險利率 drs (在 drf 基礎上增加信用利差)
    drs_paths, spread_paths = simulate_risky_rate(drf_paths, dW_vasicek, initial_spread, sigma_s, rho, T, dt, n_simulations)
    
    # 3. 模擬綜合利率 dr (GBM)
    dr_paths = simulate_gbm(r_0, mu_r, sigma_r, T, dt, n_simulations, dW_gbm)
    
    # 4. 模擬 CIR 利率 dr_cir
    dr_cir_paths = simulate_cir(r_0, k_cir, theta_cir, sigma_cir, T, dt, n_simulations, dW_cir)
    print("利率模擬完成。")

    print("開始計算債券價格路徑...")
    # 5. 計算對應的債券價格路徑
    price_drf_paths = calculate_price_paths(drf_paths, bond_data['mdurT'], bond_data['convT'])
    price_drs_paths = calculate_price_paths(drs_paths, bond_data['mdurT'], bond_data['convT'])
    price_dr_paths = calculate_price_paths(dr_paths, bond_data['mdurT'], bond_data['convT'])
    price_dr_cir_paths = calculate_price_paths(dr_cir_paths, bond_data['mdurT'], bond_data['convT'])
    print("價格計算完成。")

    # --- E. 結果可視化 ---
    
    print("正在生成圖表...")
    # 設定 Matplotlib 以正確顯示中文和負號
    try:
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # For Windows
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("未找到 'Microsoft JhengHei' 字體，圖表中的中文可能無法正常顯示。")
        print("您可以嘗試安裝 'Microsoft JhengHei' 或替換為系統中已有的中文字體，例如 'SimHei' (黑體) 或 'KaiTi' (楷體)。")

    
    # 繪製利率模擬路徑圖 (只畫前50條以保持清晰)
    fig_rates, axes_rates = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fig_rates.suptitle(f'四種利率模型的模擬路徑 (前 {min(50, n_simulations)} 條)', fontsize=16)

    axes_rates[0, 0].plot(time_points, drf_paths[:50, :].T * 100, lw=0.5)
    axes_rates[0, 0].set_title('1. 無風險利率 (drf) - Vasicek')
    axes_rates[0, 0].set_ylabel('利率 (%)')
    axes_rates[0, 0].grid(True, linestyle='--', alpha=0.6)

    axes_rates[0, 1].plot(time_points, drs_paths[:50, :].T * 100, lw=0.5)
    axes_rates[0, 1].set_title('2. 有風險利率 (drs) - Vasicek + Spread')
    axes_rates[0, 1].grid(True, linestyle='--', alpha=0.6)
    
    axes_rates[1, 0].plot(time_points, dr_paths[:50, :].T * 100, lw=0.5)
    axes_rates[1, 0].set_title('3. 綜合利率 (dr) - GBM')
    axes_rates[1, 0].set_xlabel('時間 (年)')
    axes_rates[1, 0].set_ylabel('利率 (%)')
    axes_rates[1, 0].grid(True, linestyle='--', alpha=0.6)

    axes_rates[1, 1].plot(time_points, dr_cir_paths[:50, :].T * 100, lw=0.5)
    axes_rates[1, 1].set_title('4. CIR 利率 (dr_cir)')
    axes_rates[1, 1].set_xlabel('時間 (年)')
    axes_rates[1, 1].grid(True, linestyle='--', alpha=0.6)

    fig_rates.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("tutorial_rate_paths.png")
    
    # 繪製價格模擬路徑圖
    fig_prices, axes_prices = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    fig_prices.suptitle(f'對應的債券價格模擬路徑 (前 {min(50, n_simulations)} 條)', fontsize=16)

    axes_prices[0, 0].plot(time_points, price_drf_paths[:50, :].T, lw=0.5)
    axes_prices[0, 0].set_title('基於 drf (Vasicek) 的價格')
    axes_prices[0, 0].set_ylabel('債券價格')
    axes_prices[0, 0].grid(True, linestyle='--', alpha=0.6)

    axes_prices[0, 1].plot(time_points, price_drs_paths[:50, :].T, lw=0.5)
    axes_prices[0, 1].set_title('基於 drs (Risky) 的價格')
    axes_prices[0, 1].grid(True, linestyle='--', alpha=0.6)

    axes_prices[1, 0].plot(time_points, price_dr_paths[:50, :].T, lw=0.5)
    axes_prices[1, 0].set_title('基於 dr (GBM) 的價格')
    axes_prices[1, 0].set_xlabel('時間 (年)')
    axes_prices[1, 0].set_ylabel('債券價格')
    axes_prices[1, 0].grid(True, linestyle='--', alpha=0.6)

    axes_prices[1, 1].plot(time_points, price_dr_cir_paths[:50, :].T, lw=0.5)
    axes_prices[1, 1].set_title('基於 dr_cir (CIR) 的價格')
    axes_prices[1, 1].set_xlabel('時間 (年)')
    axes_prices[1, 1].grid(True, linestyle='--', alpha=0.6)

    fig_prices.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("tutorial_price_paths.png")

    # 顯示最終價格分佈
    plt.figure(figsize=(10, 6))
    plt.hist(price_drf_paths[:, -1], bins=50, alpha=0.7, label='基於 drf (Vasicek)', density=True)
    plt.hist(price_drs_paths[:, -1], bins=50, alpha=0.7, label='基於 drs (Risky)', density=True)
    plt.hist(price_dr_paths[:, -1], bins=50, alpha=0.7, label='基於 dr (GBM)', density=True)
    plt.hist(price_dr_cir_paths[:, -1], bins=50, alpha=0.7, label='基於 dr_cir (CIR)', density=True)
    plt.title('模擬結束時的債券價格分佈')
    plt.xlabel('最終價格')
    plt.ylabel('機率密度')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("tutorial_final_price_dist.png")
    
    print("圖表已儲存為 tutorial_rate_paths.png, tutorial_price_paths.png, 和 tutorial_final_price_dist.png")
    
    # 顯示繪圖
    plt.show()

    # --- F. 簡單統計分析 ---
    final_prices_drf = price_drf_paths[:, -1]
    final_prices_drs = price_drs_paths[:, -1]
    
    print("\n--- 最終價格統計分析 ---")
    print(f"模型: 基於 drf (Vasicek)")
    print(f"  平均最終價格: {np.mean(final_prices_drf):.4f}")
    print(f"  價格標準差: {np.std(final_prices_drf):.4f}")
    print(f"  5% 分位數價格: {np.percentile(final_prices_drf, 5):.4f}")
    print(f"  95% VaR (從初始價100計算的潛在最大損失): {100 - np.percentile(final_prices_drf, 5):.4f}")
    
    print(f"\n模型: 基於 drs (Risky)")
    print(f"  平均最終價格: {np.mean(final_prices_drs):.4f}")
    print(f"  價格標準差: {np.std(final_prices_drs):.4f}")
    print(f"  5% 分位數價格: {np.percentile(final_prices_drs, 5):.4f}")
    print(f"  95% VaR (從初始價100計算的潛在最大損失): {100 - np.percentile(final_prices_drs, 5):.4f}")
    print("------------------------")