import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import tempfile
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def create_bond_database():
    """
    創建一個包含多個債券樣本的字典，作為模擬資料庫。
    注意：此處的 avgYld 和 coupon 以「百分比」形式儲存 (例如 2.2 代表 2.2%)。
    
    :return: dict，鍵為債券名稱，值為包含該債券詳細參數的字典。
    
    例如：
    平均殖利率 (avgYld): 1.553%	
    票面利率 (coupon): 2.125%
    發行日 (issueDate): 2011-01-13	
    到期日 (maturity): 2031-01-13
    每年付息 (payPerYr): 1 次	
    每年複利 (compPerYr): 1 次
    存續期間 (durT): 5.3548	
    修正存續期間 (mdurT): 5.273
    價格曲度 (convT): 1762.7325	
    """
    bond_table = pd.read_csv(os.path.join(THIS_DIR,  "bond_db.csv"))
    bond_table["issueDate"] = bond_table["issueDate"].apply(lambda x: pd.to_datetime(str(x)))
    bond_table["maturity"] = bond_table["maturity"].apply(lambda x: pd.to_datetime(str(x)))
    db = {
        row["name"]: row 
        for row in bond_table.to_dict("records")
    }
    return db

# --- 模擬函數 ---
def simulate_vasicek(r0, kappa, theta, sigma, T, dt, n_simulations, dW_shocks):
    """
    使用 Vasicek 模型模擬債券利率(drf)的路徑。
    此模型包含均值回歸特性。

    :param r0: float, 初始利率 (小數形式)。
    :param kappa: float, 均值回歸速度。
    :param theta: float, 長期平均利率。
    :param sigma: float, 波動率。
    :param T: float, 總模擬時長(年)。
    :param dt: float, 每個時間步長(年)。
    :param n_simulations: int, 模擬路徑的數量。
    :param dW_shocks: np.ndarray, 預先生成的隨機衝擊。
    :return: np.ndarray, 模擬的利率路徑，形狀為 (n_simulations, num_steps + 1)。
    """
    num_steps = int(T / dt)
    rates = np.zeros((n_simulations, num_steps + 1)); rates[:, 0] = r0
    for t in range(1, num_steps + 1):
        rates[:, t] = np.maximum(0.001, rates[:, t-1] + kappa * (theta - rates[:, t-1]) * dt + sigma * dW_shocks[:, t-1])
    return rates

def simulate_cir(r0, kappa, theta, sigma, T, dt, n_simulations, dW_shocks):
    """
    使用 Cox-Ingersoll-Ross (CIR) 模型模擬利率路徑。
    此模型確保利率永遠為正 (若 2*kappa*theta > sigma**2)。

    :param r0: float, 初始利率 (小數形式)。
    :param kappa: float, 均值回歸速度。
    :param theta: float, 長期平均利率。
    :param sigma: float, 波動率。
    :param T: float, 總模擬時長(年)。
    :param dt: float, 每個時間步長(年)。
    :param n_simulations: int, 模擬路徑的數量。
    :param dW_shocks: np.ndarray, 預先生成的隨機衝擊。
    :return: np.ndarray, 模擬的利率路徑，形狀為 (n_simulations, num_steps + 1)。
    """
    num_steps = int(T / dt)
    rates = np.zeros((n_simulations, num_steps + 1))
    rates[:, 0] = r0
    for t in range(1, num_steps + 1):
        # 使用 np.maximum 確保根號內的值非負，避免數值計算問題
        sqrt_r = np.sqrt(np.maximum(0, rates[:, t-1]))
        rates[:, t] = rates[:, t-1] + kappa * (theta - rates[:, t-1]) * dt + sigma * sqrt_r * dW_shocks[:, t-1]
        # 再次確保利率不會變為負數
        rates[:, t] = np.maximum(0.0001, rates[:, t])
    return rates

def simulate_risky_rate(rf_paths, dW_rf, initial_spread, sigma_spread, correlation, T, dt, n_simulations, dW_independent):
    """
    在先前已模擬的利率的基礎上，模擬有風險利率(drs)的路徑。
    主要模擬與債券利率相關的信用利差(credit spread)的變化。

    :param rf_paths: np.ndarray, 已模擬好的債券利率路徑。
    :param dW_rf: np.ndarray, 生成債券利率路徑所使用的隨機衝擊。
    :param initial_spread: float, 初始信用利差。
    :param sigma_spread: float, 信用利差的波動率。
    :param correlation: float, 債券利率與信用利差變動的相關係數。
    :param T: float, 總模擬時長(年)。
    :param dt: float, 每個時間步長(年)。
    :param n_simulations: int, 模擬路徑的數量。
    :param dW_independent: np.ndarray, 獨立的隨機衝擊，用於生成利差的非系統性風險部分。
    :return: tuple (np.ndarray, np.ndarray), 分別為模擬的有風險利率路徑和信用利差路徑。
    """
    num_steps = int(T / dt); 
    spread_paths = np.zeros((n_simulations, num_steps + 1)); 
    spread_paths[:, 0] = initial_spread
    dW_spread = correlation * dW_rf + np.sqrt(1 - correlation**2) * dW_independent
    for t in range(1, num_steps + 1):
        spread_paths[:, t] = np.maximum(0.001, spread_paths[:, t-1] + sigma_spread * dW_spread[:, t-1])
    return rf_paths + spread_paths, spread_paths

def simulate_gbm(r0, mu, sigma, T, dt, n_simulations, dW_shocks):
    """
    使用幾何布朗運動(Geometric Brownian Motion)模型模擬**綜合利率 (dr)** 的路徑。
    公式: dr_t = μr_t dt + σr_t dW_t
    此模型常用於模擬股價，這裡用來作為一個簡化的利率模型。

    :param r0: float, 初始利率 (小數形式)。
    :param mu: float, 長期漂移趨勢 (μ, 在此對應 mu_r)。
    :param sigma: float, 年化波動率 (σ, 在此對應 sigma_r)。
    :param T: float, 總模擬時長(年)。
    :param dt: float, 每個時間步長(年)。
    :param n_simulations: int, 模擬路徑的數量。
    :param dW_shocks: np.ndarray, 預先生成的隨機衝擊。
    :return: np.ndarray, 模擬的利率路徑。
    """
    num_steps = int(T / dt); rates = np.zeros((n_simulations, num_steps + 1)); rates[:, 0] = r0
    for t in range(1, num_steps + 1):
        rates[:, t] = np.maximum(0.001, rates[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW_shocks[:, t-1]))
    return rates

def calculate_price_paths(rate_paths, mdurT, convT, initial_price=100.0):
    """
    使用修正存續期間(mdurT)和價格曲度(convT)近似法，根據利率路徑計算債券價格路徑。
    
    重要：
    此處的 convT 參數假定為您資料庫中的 "1/2價格曲度" (如 288.3137)，
    其金融定義為 (P * C_Mod / 2)，且 P 為 initial_price (通常為 100)。

    :param rate_paths: np.ndarray, 模擬的利率路徑，刻度為小數點形式。
    :param mdurT: float, 債券的修正存續期間 (D_Mod)。
    :param convT: float, 債券的 "1/2 價格曲度" (P * C_Mod / 2)。
    :param initial_price: float, 債券的初始價格，預設為100。
    :return: np.ndarray, 模擬的債券價格路徑。
    """
    n_simulations, num_steps = rate_paths.shape[0], rate_paths.shape[1] - 1
    price_paths = np.zeros_like(rate_paths)
    price_paths[:, 0] = initial_price
    
    # --- 邏輯修正 ---
    # 根據定義，convT = (initial_price * C_Mod / 2)
    # 我們需要 C_Mod / 2 這一項
    # 因此，C_Mod / 2 = convT / initial_price
    # 
    # 這個值在近似法中被假定為常數，不應隨 P(t-1) 變動。
    # 【錯誤的邏輯】: convexity_factor = convT / prev_price
    # 【修正後的邏輯】:
    convexity_factor = convT / initial_price
    # --- 修正結束 ---
    
    for t in range(1, num_steps + 1):
        delta_y = rate_paths[:, t] - rate_paths[:, t-1]
        
        # 抓取前一期的價格 P(t-1)
        prev_price = price_paths[:, t-1]
        
        # 1. 計算價格變動百分比 (dP/P)
        # dP/P = -mdurT * dy + (C_Mod / 2) * (dy^2)
        #
        # 【注意】: 這裡的 convexity_factor 是我們在迴圈外
        #          就已經計算好的常數 (convT / initial_price)
        price_change_factor = -mdurT * delta_y + convexity_factor * (delta_y ** 2)
        
        # 2. 計算新價格 P(t) = P(t-1) * (1 + dP/P)
        price_paths[:, t] = prev_price * (1 + price_change_factor)
        
    return price_paths

# --- 主執行函數 ---
def run_simulation(bond_data, n_simulations, dt, params):
    """
    程式核心執行函數。
    整合所有模擬、計算、繪圖與報告生成步驟。

    :param bond_data: dict, 選定債券的詳細資料。
    :param n_simulations: int, 模擬總次數。 e.g., 100
    :param dt: float, 每個時間步長(年)。 e.g., 1/252
    :param params: dict, 所有模擬模型的參數。
    :return: tuple, 包含所有結果的元組 (數據幀, 圖表, 報告路徑)。
    """
    # 設定隨機種子以確保每次模擬結果可重現
    np.random.seed(42)
    
    # --- 1. 參數準備 ---
    # 將來自資料庫的百分比格式 (如 2.2%) 轉換為小數格式 (0.022) 以進行計算
    initial_yield = bond_data['avgYld'] / 100.0
    
    # 計算債券的剩餘到期時間(年)，作為模擬的總時長 T
    time_to_maturity = (bond_data['maturity'] - datetime.now()).days / 365.25
    T = max(time_to_maturity, 0.1) # 確保至少模擬一小段時間
    num_steps = int(T / dt)
    time_points = np.linspace(0, T, num_steps + 1) # 模擬的時間點數列
    
    # --- 2. 生成隨機衝擊 ---
    # 為每個模型預先生成符合標準常態分佈的隨機衝擊 (Wiener Process increments)
    # dW ~ N(0, dt)
    dW_rf = np.random.normal(0, 1, (n_simulations, num_steps)) * np.sqrt(dt)
    dW_spread_independent = np.random.normal(0, 1, (n_simulations, num_steps)) * np.sqrt(dt)
    dW_dr = np.random.normal(0, 1, (n_simulations, num_steps)) * np.sqrt(dt)
    dW_cir = np.random.normal(0, 1, (n_simulations, num_steps)) * np.sqrt(dt)

    # --- 3. 利率路徑模擬 ---
    # 模擬三種不同模型下的利率路徑
    # drf: 債券風險利率 (Vasicek 模型)
    rf_paths = simulate_vasicek(initial_yield, params['kappa_rf'], params['theta_rf'], params['sigma_rf'], T, dt, n_simulations, dW_rf)
    # drs: 有風險利率 (Vasicek + 相關信用利差)
    drs_paths, _ = simulate_risky_rate(rf_paths, dW_rf, params['initial_spread'], params['sigma_spread'], params['correlation'], T, dt, n_simulations, dW_spread_independent)
    # dr: 綜合利率 (幾何布朗運動模型)
    dr_paths = simulate_gbm(initial_yield, params['mu_r'], params['sigma_r'], T, dt, n_simulations, dW_dr)
    # dr_cir: CIR 模型
    cir_paths = simulate_cir(initial_yield, params['kappa_cir'], params['theta_cir'], params['sigma_cir'], T, dt, n_simulations, dW_cir)

    # --- 4. 債券價格路徑模擬 ---
    # 根據每條利率路徑，使用存續期間和曲度來估算債券價格的變化
    price_rf_paths = calculate_price_paths(rf_paths, bond_data['mdurT'], bond_data['convT'])
    price_drs_paths = calculate_price_paths(drs_paths, bond_data['mdurT'], bond_data['convT'])
    price_dr_paths = calculate_price_paths(dr_paths, bond_data['mdurT'], bond_data['convT'])
    price_cir_paths = calculate_price_paths(cir_paths, bond_data['mdurT'], bond_data['convT'])

    # --- 5. 統計數據分析 ---
    # 計算利率每日變化的基本統計量
    drf_changes = np.diff(rf_paths, axis=1).flatten()
    drs_changes = np.diff(drs_paths, axis=1).flatten()
    dr_changes = np.diff(dr_paths, axis=1).flatten()
    cir_changes = np.diff(cir_paths, axis=1).flatten()
    
    # 計算 drf 和 drs 變化之間的共變異數和相關係數
    cov_matrix = np.cov(drf_changes, drs_changes)
    corr_matrix = np.corrcoef(drf_changes, drs_changes)
    
    rate_stats_df = pd.DataFrame({
        "指標": ["drf 變化量平均", "drf 變化量標準差", "drs 變化量平均", "drs 變化量標準差", "dr 變化量平均", "dr 變化量標準差", "dr_cir 變化量平均", "dr_cir 變化量標準差", "drf-drs 共變異數", "drf-drs 相關係數"],
        "數值": [f"{np.mean(drf_changes):.8f}", f"{np.std(drf_changes):.8f}", f"{np.mean(drs_changes):.8f}", f"{np.std(drs_changes):.8f}", f"{np.mean(dr_changes):.8f}", f"{np.std(dr_changes):.8f}", f"{np.mean(cir_changes):.8f}", f"{np.std(cir_changes):.8f}", f"{cov_matrix[0, 1]:.8f}", f"{corr_matrix[0, 1]:.6f}"]
    })

    # 計算最終價格的統計數據，包括風險價值 (VaR)
    final_prices_rf = price_rf_paths[:, -1]
    final_prices_drs = price_drs_paths[:, -1]
    final_prices_dr = price_dr_paths[:, -1]
    final_prices_cir = price_cir_paths[:, -1]
    
    price_stats_df = pd.DataFrame({
        "統計量": ["平均價格", "價格標準差", "最大價格", "最小價格", "95%分位數 (Q95)", "5%分位數 (Q5)", "95% VaR (價值損失)"],
        "基於 drf": [f"{np.mean(final_prices_rf):.4f}", f"{np.std(final_prices_rf):.4f}", f"{np.max(final_prices_rf):.4f}", f"{np.min(final_prices_rf):.4f}", f"{np.percentile(final_prices_rf, 95):.4f}", f"{np.percentile(final_prices_rf, 5):.4f}", f"{100 - np.percentile(final_prices_rf, 5):.4f}"],
        "基於 drs": [f"{np.mean(final_prices_drs):.4f}", f"{np.std(final_prices_drs):.4f}", f"{np.max(final_prices_drs):.4f}", f"{np.min(final_prices_drs):.4f}", f"{np.percentile(final_prices_drs, 95):.4f}", f"{np.percentile(final_prices_drs, 5):.4f}", f"{100 - np.percentile(final_prices_drs, 5):.4f}"],
        "基於 dr": [f"{np.mean(final_prices_dr):.4f}", f"{np.std(final_prices_dr):.4f}", f"{np.max(final_prices_dr):.4f}", f"{np.min(final_prices_dr):.4f}", f"{np.percentile(final_prices_dr, 95):.4f}", f"{np.percentile(final_prices_dr, 5):.4f}", f"{100 - np.percentile(final_prices_dr, 5):.4f}"],
        "基於 dr_cir": [f"{np.mean(final_prices_cir):.4f}", f"{np.std(final_prices_cir):.4f}", f"{np.max(final_prices_cir):.4f}", f"{np.min(final_prices_cir):.4f}", f"{np.percentile(final_prices_cir, 95):.4f}", f"{np.percentile(final_prices_cir, 5):.4f}", f"{100 - np.percentile(final_prices_cir, 5):.4f}"]
    })

    # --- 6. 生成 Excel 報告 ---
    # 創建一個暫存的 Excel 檔案來儲存所有詳細結果
    fd, report_filepath = tempfile.mkstemp(suffix=".xlsx")
    os.close(fd)
    os.makedirs("excel_reports", exist_ok=True)
    report_filepath = os.path.join("excel_reports", f"bond_simulation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
    
    with pd.ExcelWriter(report_filepath) as writer:
        # 寫入債券基本資訊
        pd.DataFrame(list(bond_data.items()), columns=['項目', '數值']).applymap(lambda x: x.strftime('%Y-%m-%d') if isinstance(x, datetime) else x).to_excel(writer, sheet_name='bond_info', index=False)
        # 寫入本次模擬使用的參數
        pd.DataFrame(list(params.items()), columns=['參數', '設定值']).to_excel(writer, sheet_name='simulation_parameters', index=False)
        # 寫入利率和價格的統計結果
        rate_stats_df.to_excel(writer, sheet_name='rate_statistics', index=False)
        price_stats_df.to_excel(writer, sheet_name='price_statistics', index=False)
        
        # 如果模擬次數不多，則將詳細的模擬路徑寫入 Excel
        if n_simulations <= 1000:
            pd.DataFrame(rf_paths.T, index=time_points).to_excel(writer, sheet_name='drf_rate_paths')
            pd.DataFrame(drs_paths.T, index=time_points).to_excel(writer, sheet_name='drs_rate_paths')
            pd.DataFrame(dr_paths.T, index=time_points).to_excel(writer, sheet_name='dr_rate_paths')
            pd.DataFrame(cir_paths.T, index=time_points).to_excel(writer, sheet_name='dr_cir_rate_paths')
            pd.DataFrame(price_rf_paths.T, index=time_points).to_excel(writer, sheet_name='drf_price_paths')
            pd.DataFrame(price_drs_paths.T, index=time_points).to_excel(writer, sheet_name='drs_price_paths')
            pd.DataFrame(price_dr_paths.T, index=time_points).to_excel(writer, sheet_name='dr_price_paths')
            pd.DataFrame(price_cir_paths.T, index=time_points).to_excel(writer, sheet_name='dr_cir_price_paths')
        else:
            # 如果模擬次數過多，為避免 Excel 檔案過大，改為生成多個 CSV 檔案
            print(f"模擬次數過多 ({n_simulations})，僅生成 CSV 格式的詳細路徑報告以節省空間。")
            report_dir = os.path.dirname(report_filepath)
            base_name = os.path.splitext(os.path.basename(report_filepath))[0]
            pd.DataFrame(rf_paths.T, index=time_points).to_csv(os.path.join(report_dir, f'{base_name}_drf_rate_paths.csv'))
            pd.DataFrame(drs_paths.T, index=time_points).to_csv(os.path.join(report_dir, f'{base_name}_drs_rate_paths.csv'))
            pd.DataFrame(dr_paths.T, index=time_points).to_csv(os.path.join(report_dir, f'{base_name}_dr_rate_paths.csv'))
            pd.DataFrame(cir_paths.T, index=time_points).to_csv(os.path.join(report_dir, f'{base_name}_dr_cir_rate_paths.csv'))
            pd.DataFrame(price_rf_paths.T, index=time_points).to_csv(os.path.join(report_dir, f'{base_name}_drf_price_paths.csv'))
            pd.DataFrame(price_drs_paths.T, index=time_points).to_csv(os.path.join(report_dir, f'{base_name}_drs_price_paths.csv'))
            pd.DataFrame(price_dr_paths.T, index=time_points).to_csv(os.path.join(report_dir, f'{base_name}_dr_price_paths.csv'))
            pd.DataFrame(price_cir_paths.T, index=time_points).to_csv(os.path.join(report_dir, f'{base_name}_dr_cir_price_paths.csv'))
            
        print("report_filepath:", report_filepath)
            
    # --- 7. 繪圖 ---
    # 設定 Matplotlib 以正確顯示中文和負號
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 繪製利率模擬路徑圖
    fig_rate_paths = plt.figure(figsize=(12, 10))
    plt.suptitle('利率模擬路徑 (前50條)', fontsize=16)
    
    ax1 = fig_rate_paths.add_subplot(2, 2, 1)
    ax1.plot(time_points, rf_paths[:50, :].T * 100, lw=0.5)
    ax1.set_title('債券風險利率 (drf) - Vasicek')
    ax1.set_ylabel('利率 (%)')
    
    ax2 = fig_rate_paths.add_subplot(2, 2, 2)
    ax2.plot(time_points, drs_paths[:50, :].T * 100, lw=0.5)
    ax2.set_title('有風險利率 (drs) - Vasicek + Spread')
    ax2.set_ylabel('利率 (%)')

    ax3 = fig_rate_paths.add_subplot(2, 2, 3)
    ax3.plot(time_points, dr_paths[:50, :].T * 100, lw=0.5)
    ax3.set_title('綜合利率 (dr) - GBM')
    ax3.set_xlabel('時間 (年)')
    ax3.set_ylabel('利率 (%)')

    ax4 = fig_rate_paths.add_subplot(2, 2, 4)
    ax4.plot(time_points, cir_paths[:50, :].T * 100, lw=0.5)
    ax4.set_title('CIR 利率 (dr_cir) - Cox-Ingersoll-Ross')
    ax4.set_xlabel('時間 (年)')
    ax4.set_ylabel('利率 (%)')
    
    fig_rate_paths.tight_layout(rect=[0, 0, 1, 0.96])

    # 繪製利率變化分佈圖
    fig_rate_dist = plt.figure(figsize=(12, 5))
    plt.hist(drf_changes * 100, bins=100, alpha=0.6, label='drf (Vasicek)', density=True)
    plt.hist(drs_changes * 100, bins=100, alpha=0.6, label='drs (Risky)', density=True)
    plt.hist(dr_changes * 100, bins=100, alpha=0.6, label='dr (GBM)', density=True)
    plt.hist(cir_changes * 100, bins=100, alpha=0.6, label='dr_cir (CIR)', density=True)
    plt.title('每日利率變化分佈')
    plt.xlabel('利率變化 (%)')
    plt.ylabel('機率密度')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # 繪製價格模擬路徑圖
    fig_price_paths = plt.figure(figsize=(12, 10))
    plt.suptitle('債券價格模擬路徑 (前50條)', fontsize=16)

    ax_p1 = fig_price_paths.add_subplot(2, 2, 1)
    ax_p1.plot(time_points, price_rf_paths[:50, :].T, lw=0.5)
    ax_p1.set_title('基於 drf (Vasicek) 的價格')
    ax_p1.set_ylabel('價格')

    ax_p2 = fig_price_paths.add_subplot(2, 2, 2)
    ax_p2.plot(time_points, price_drs_paths[:50, :].T, lw=0.5)
    ax_p2.set_title('基於 drs (Risky) 的價格')
    ax_p2.set_ylabel('價格')

    ax_p3 = fig_price_paths.add_subplot(2, 2, 3)
    ax_p3.plot(time_points, price_dr_paths[:50, :].T, lw=0.5)
    ax_p3.set_title('基於 dr (GBM) 的價格')
    ax_p3.set_xlabel('時間 (年)')
    ax_p3.set_ylabel('價格')

    ax_p4 = fig_price_paths.add_subplot(2, 2, 4)
    ax_p4.plot(time_points, price_cir_paths[:50, :].T, lw=0.5)
    ax_p4.set_title('基於 dr_cir (CIR) 的價格')
    ax_p4.set_xlabel('時間 (年)')
    ax_p4.set_ylabel('價格')
    
    fig_price_paths.tight_layout(rect=[0, 0, 1, 0.96])

    # 繪製最終價格分佈圖
    fig_price_dist = plt.figure(figsize=(12, 5))
    plt.hist(final_prices_rf, bins=100, alpha=0.6, label='基於 drf (Vasicek)', density=True)
    plt.hist(final_prices_drs, bins=100, alpha=0.6, label='基於 drs (Risky)', density=True)
    plt.hist(final_prices_dr, bins=100, alpha=0.6, label='基於 dr (GBM)', density=True)
    plt.hist(final_prices_cir, bins=100, alpha=0.6, label='基於 dr_cir (CIR)', density=True)
    plt.title('最終債券價格分佈')
    plt.xlabel('價格')
    plt.ylabel('機率密度')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
        
    return rate_stats_df, price_stats_df, fig_rate_paths, fig_rate_dist, fig_price_paths, fig_price_dist, report_filepath
