import gradio as gr
import pandas as pd
from datetime import datetime

# 從 bond.py 導入重構後的函數和數據
from bond import create_bond_database, run_simulation

# 1. 數據準備
BOND_DATABASE = create_bond_database()
BOND_CHOICES = list(BOND_DATABASE.keys())

# --- Gradio 介面輔助函數 ---

def get_bond_info_html(bond_name):
    """
    根據下拉選單選擇的債券名稱，生成用於顯示其詳細資訊的 HTML 字串。
    
    :param bond_name: str, 使用者從下拉選單選擇的債券名稱。
    :return: str, 一個 HTML 格式的字串，用於在 Gradio 介面中顯示。
    """
    if not bond_name:
        return "<p>請從上方選擇一個債券</p>"
    
    bond = BOND_DATABASE[bond_name]
    issue_date_str = bond['issueDate'].strftime('%Y-%m-%d')
    maturity_date_str = bond['maturity'].strftime('%Y-%m-%d')
    
    # **重要**: 根據資料庫中的百分比格式，調整顯示方式
    html = f"""
    <div style="border: 1px solid #e0e0e0; padding: 15px; border-radius: 5px;">
        <h4 style="margin-top:0;">債券: {bond_name}</h4>
        <table style="width:100%;">
            <tr><td style="width:50%;"><strong>平均殖利率 (avgYld):</strong> {bond['avgYld']:.3f}%</td><td style="width:50%;"><strong>票面利率 (coupon):</strong> {bond['coupon']:.3f}%</td></tr>
            <tr><td><strong>發行日 (issueDate):</strong> {issue_date_str}</td><td><strong>到期日 (maturity):</strong> {maturity_date_str}</td></tr>
            <tr><td><strong>每年付息 (payPerYr):</strong> {bond['payPerYr']} 次</td><td><strong>每年複利 (compPerYr):</strong> {bond['compPerYr']} 次</td></tr>
            <tr><td><strong>存續期間 (durT):</strong> {bond['durT']}</td><td><strong>修正存續期間 (mdurT):</strong> {bond['mdurT']}</td></tr>
            <tr><td><strong>價格曲度 (convT):</strong> {bond['convT']}</td><td></td></tr>
        </table>
    </div>
    """
    return html

def run_monte_carlo(
    bond_name, n_simulations, dt,
    kappa_rf, theta_rf, sigma_rf,
    initial_spread, sigma_spread, correlation,
    mu_r, sigma_r,
    kappa_cir, theta_cir, sigma_cir
):
    """
    Gradio 的主要執行函數。
    它負責收集所有來自介面的輸入參數，將百分比單位轉換為小數，
    調用後端的 `run_simulation` 核心函式，並將計算結果回傳給介面的各個輸出元件。

    :param bond_name: str, 選擇的債券名稱。 (e.g., 'CGB10Y')
    :param n_simulations: int, 模擬次數。 (e.g., 100)
    :param dt: float, 時間步長。 (e.g., 1/252)
    :param kappa_rf: float, 均值回歸速度。 (e.g., 0.3)
    :param theta_rf: float, 長期平均利率 (%)。 (e.g., 2.2)
    :param sigma_rf: float, 波動率 (%)。 (e.g., 0.5)
    :param initial_spread: float, 初始信用利差 (%)。 (e.g., 0.8)
    :param sigma_spread: float, 利差波動率 (%)。 (e.g., 0.3)
    :param correlation: float, drf與利差的相關係數。 (e.g., 0.6)
    :param mu_r: float, 長期漂移趨勢 (%)。 (e.g., 0.05)
    :param sigma_r: float, 年化波動率 (%)。 (e.g., 0.6)
    :param kappa_cir: float, CIR 均值回歸速度。
    :param theta_cir: float, CIR 長期平均利率 (%)。
    :param sigma_cir: float, CIR 波動率 (%)。
    :return: tuple, 包含所有結果的元組，用於更新 Gradio 介面的各個輸出元件。
    """
    if not bond_name:
        raise gr.Error("請先選擇一個債券！")

    bond_data = BOND_DATABASE[bond_name]
    
    # **重要**: 將來自UI的百分比參數轉換為小數以進行計算
    params = {
        'kappa_rf': kappa_rf, 
        'theta_rf': theta_rf / 100.0, 
        'sigma_rf': sigma_rf / 100.0,
        'initial_spread': initial_spread / 100.0, 
        'sigma_spread': sigma_spread / 100.0, 
        'correlation': correlation,
        'mu_r': mu_r / 100.0, 
        'sigma_r': sigma_r / 100.0,
        'kappa_cir': kappa_cir,
        'theta_cir': theta_cir / 100.0,
        'sigma_cir': sigma_cir / 100.0,
        "n_simulations":int(n_simulations),
    }
    
    rate_stats_df, price_stats_df, fig_rate_paths, fig_rate_dist, fig_price_paths, fig_price_dist, report_filepath = run_simulation(
        bond_data=bond_data, n_simulations=int(n_simulations), dt=dt, params=params
    )
    
    return (
        rate_stats_df, price_stats_df,
        fig_rate_paths, fig_rate_dist,
        fig_price_paths, fig_price_dist,
        gr.update(value=report_filepath, visible=True)
    )

# --- 建立 Gradio 介面 ---

with gr.Blocks(theme=gr.themes.Soft(), title="債券利率蒙地卡羅模擬") as app:
    gr.Markdown("# 債券利率與價格蒙地卡羅模擬")
    
    # --- 步驟 1 & 2: 參數設定 ---
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("## 步驟 1: 選擇債券")
            bond_selector = gr.Dropdown(BOND_CHOICES, label="選擇債券", info="從模擬的資料庫中選擇一個債券以載入其初始參數。", value=BOND_CHOICES[0])
            bond_info_display = gr.HTML(get_bond_info_html(BOND_CHOICES[0]))
        
        with gr.Column(scale=3):
            gr.Markdown("## 步驟 2: 設定模擬參數")
            with gr.Tabs():
                with gr.Tab("債券風險利率 (drf)"):
                    kappa_rf = gr.Slider(0.01, 1.0, value=0.3, step=0.01, label="均值回歸速度 (kappa_rf)")
                    theta_rf = gr.Slider(1.0, 5.0, value=2.2, step=0.1, label="長期平均利率 (%)")
                    sigma_rf = gr.Slider(0.1, 2.0, value=0.5, step=0.1, label="波動率 (%)")
                with gr.Tab("有風險利率 (drs)"):
                    initial_spread = gr.Slider(0.1, 3.0, value=0.8, step=0.1, label="初始信用利差 (%)")
                    sigma_spread = gr.Slider(0.1, 1.0, value=0.3, step=0.05, label="利差波動率 (%)")
                    correlation = gr.Slider(-1.0, 1.0, value=0.6, step=0.05, label="drf與利差的相關係數")
                with gr.Tab("綜合利率 (dr)"):
                    mu_r = gr.Slider(0.0, 0.2, value=0.05, step=0.01, label="長期漂移趨勢 (%)")
                    sigma_r = gr.Slider(0.1, 2.0, value=0.6, step=0.1, label="年化波動率 (%)")
                with gr.Tab("CIR 利率 (dr_cir)"):
                    gr.Markdown("#### Cox-Ingersoll-Ross (CIR) 模型\n此模型 `dr = k(θ - r)dt + σ√r dW` 確保利率為正。")
                    kappa_cir = gr.Slider(0.01, 1.0, value=0.2, step=0.01, label="均值回歸速度 (kappa_cir)")
                    theta_cir = gr.Slider(1.0, 5.0, value=2.5, step=0.1, label="長期平均利率 (%)")
                    sigma_cir = gr.Slider(0.1, 2.0, value=0.4, step=0.1, label="波動率 (%)")
            n_simulations = gr.Slider(50, 100000, value=100, step=50, label="模擬次數 (n_simulations)")
            dt = gr.Number(value=1/252, label="時間步長 (dt)", info="以年為單位，預設為一個交易日(1/252年)。")

    # --- 步驟 3: 執行 ---
    run_button = gr.Button("開始模擬", variant="primary")

    # --- 步驟 4: 結果呈現 ---
    with gr.Tabs():
        with gr.Tab("利率分析"):
            gr.Markdown("## 利率模擬結果")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### 利率統計數據")
                    rate_results_table = gr.DataFrame(headers=["指標", "數值"], datatype=["str", "str"])
                with gr.Column(scale=2):
                    gr.Markdown("#### 利率路徑與相關性圖")
                    plot_rate_paths = gr.Plot()
                    gr.Markdown("#### 利率變化分佈圖")
                    plot_rate_distribution = gr.Plot()
        
        with gr.Tab("債券價格分析"):
            gr.Markdown("## 債券價格模擬結果")
            gr.Markdown("使用修正存續期間與曲度來估算價格變化。初始價格假設為 100。")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### 價格統計數據 (含 VaR)")
                    price_results_table = gr.DataFrame()
                with gr.Column(scale=2):
                    gr.Markdown("#### 價格模擬路徑")
                    plot_price_paths = gr.Plot()
                    gr.Markdown("#### 最終價格分佈")
                    plot_price_distribution = gr.Plot()

    # --- 步驟 5: 下載資料 ---
    with gr.Column():
        gr.Markdown("--- \n ## 下載完整報告，(若模擬次數超過1000，則不會包含模擬路徑與價格資料，以避免檔案過大)")
        report_file = gr.File(label="下載完整報告 (Excel)", visible=False)

    # --- 事件監聽 ---
    bond_selector.change(fn=get_bond_info_html, inputs=bond_selector, outputs=bond_info_display)
    
    all_inputs = [
        bond_selector, n_simulations, dt,
        kappa_rf, theta_rf, sigma_rf,
        initial_spread, sigma_spread, correlation,
        mu_r, sigma_r,
        kappa_cir, theta_cir, sigma_cir
    ]
    
    run_button.click(
        fn=run_monte_carlo,
        inputs=all_inputs,
        outputs=[
            rate_results_table, price_results_table,
            plot_rate_paths, plot_rate_distribution,
            plot_price_paths, plot_price_distribution,
            report_file
        ]
    )

if __name__ == "__main__":
    app.launch()
