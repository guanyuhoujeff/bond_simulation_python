# 債券利率蒙地卡羅模擬 (Bond Interest Rate Monte Carlo Simulation)

---

### 課程資訊
*   **課程**：114學年度 國立高雄科技大學 金融系碩博課程 - 金融工程
*   **指導老師**：洪志興 老師
*   **整理人**：侯冠宇

---

### 專案簡介

本專案為債券利率蒙地卡羅模擬。可透過模擬未來的利率路徑，評估利率變動對債券價格可能造成的影響，並進行風險分析。

主要功能包含：
1.  **多種利率模型實作**：
    *   **Vasicek 模型**：具有均值回歸特性的債券風險利率模型。
    *   **幾何布朗運動 (GBM) 模型**：常用於資產價格模擬的基礎模型。
    *   **有風險利率模型**：在無風險利率的基礎上，疊加一個隨機變動的信用利差。
    *   **Cox-Ingersoll-Ross (CIR) 模型**：確保利率為正的均值回歸模型。

2.  **債券價格近似計算**：
    *   利用「修正存續期間 (Modified Duration)」與「價格曲度 (Convexity)」來近似計算因利率變動所導致的債券價格變化。

3.  **教學 Jupyter Notebook**：
    *   `bond_simulation_tutorial.ipynb` 提供了一個互動式的教學腳本，詳細解釋了各個模型的原理與程式碼實作，並將結果可視化。

---

### 如何在 Google Colab 中開啟教學 Notebook

您可以直接在瀏覽器中透過 Google Colab 執行本專案的教學 Notebook，無需在本機安裝任何環境。

[![在 Colab 中開啟](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guanyuhoujeff/bond_simulation_python/blob/main/bond_simulation_tutorial.ipynb)
點擊上方按鈕即可直接在 Google Colab 中開啟 `bond_simulation_tutorial.ipynb`。

---

### 聯絡方式

若有任何問題或建議，歡迎來信至：[jeff7522553@gmail.com](mailto:jeff7522553@gmail.com)