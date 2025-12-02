import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.optimize as sco
import plotly.graph_objects as go

# === 設定網頁標題與排版 ===
st.set_page_config(page_title="台股效率前沿計算機", page_icon="📈")
st.title("📈 台股效率前沿計算機")
st.write("輸入代號，自動計算最佳資產配置 (Max Sharpe)")

# === 側邊欄：輸入區 ===
st.sidebar.header("參數設定")

# 1. 輸入代號
default_tickers = "2330.TW, 2317.TW, 2454.TW, 0050.TW"
tickers_input = st.sidebar.text_area("輸入股票代號 (用逗號隔開, 上市.TW/上櫃.TWO)", default_tickers)

# 2. 選擇日期
col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("開始日期", value=pd.to_datetime("2024-01-01"))
end_date = col2.date_input("結束日期", value=pd.to_datetime("today"))

# 3. 無風險利率
rf_input = st.sidebar.number_input("無風險利率 (%)", value=2.0, step=0.1)
risk_free_rate = rf_input / 100

# === 核心運算邏輯 ===
def run_optimization():
    # 處理代號
    tickers = [t.strip().upper() for t in tickers_input.split(',')]
    
    with st.spinner(f"正在下載 {len(tickers)} 檔股票資料..."):
        try:
            # 下載資料
            data = yf.download(tickers, start=start_date, end=end_date)['Close']
            
            # 資料檢查
            if data.empty:
                st.error("❌ 下載失敗：找不到資料，請檢查代號或日期。")
                return
            
            # 移除資料不足的股票
            data = data.dropna(axis=1, how='all').dropna()
            
            if data.shape[1] < 2:
                st.error("⚠️ 有效股票少於 2 檔，無法計算效率前沿。")
                return
                
            used_tickers = data.columns.tolist()
            st.success(f"✅ 成功載入 {len(used_tickers)} 檔股票資料！")
            
            # 計算報酬與風險
            returns = data.pct_change().dropna()
            mean_returns = returns.mean() * 252
            cov_matrix = returns.cov() * 252
            num_assets = len(used_tickers)

            # --- 定義函數 ---
            def portfolio_performance(weights):
                weights = np.array(weights)
                ret = np.sum(mean_returns * weights) * 252
                std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
                return ret, std

            def neg_sharpe_ratio(weights):
                p_ret, p_std = portfolio_performance(weights)
                return -(p_ret - risk_free_rate) / p_std

            # --- 規劃求解 (Solver) ---
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for _ in range(num_assets))
            init_guess = num_assets * [1. / num_assets,]

            result = sco.minimize(neg_sharpe_ratio, init_guess, 
                                  method='SLSQP', bounds=bounds, constraints=constraints)
            
            best_w = result.x
            opt_ret, opt_std = portfolio_performance(best_w)

            # === 顯示結果 ===
            st.markdown("---")
            st.subheader("🏆 最佳投組建議 (Max Sharpe)")
            
            # 建立結果表格
            res_df = pd.DataFrame({
                "股票": used_tickers,
                "建議權重": [f"{w:.2%}" for w in best_w],
                "原始數值": best_w # 用於排序
            }).sort_values("原始數值", ascending=False)
            
            # 顯示指標卡片
            c1, c2, c3 = st.columns(3)
            c1.metric("預期年化報酬", f"{opt_ret:.2%}")
            c2.metric("預期年化波動", f"{opt_std:.2%}")
            c3.metric("夏普比率", f"{(opt_ret - risk_free_rate)/opt_std:.2f}")

            # 顯示圓餅圖與表格
            c_chart, c_table = st.columns([1, 1])
            with c_table:
                st.table(res_df[["股票", "建議權重"]])
            
            with c_chart:
                fig_pie = go.Figure(data=[go.Pie(labels=res_df["股票"], values=res_df["原始數值"], hole=.4)])
                fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=250)
                st.plotly_chart(fig_pie, use_container_width=True)

            # === 蒙地卡羅模擬圖表 ===
            st.subheader("📊 效率前沿模擬圖")
            
            # 隨機模擬 2000 次
            sim_n = 2000
            results = np.zeros((3, sim_n))
            for i in range(sim_n):
                w = np.random.random(num_assets)
                w /= np.sum(w)
                pr, ps = portfolio_performance(w)
                results[0,i] = ps # X: Risk
                results[1,i] = pr # Y: Return
                results[2,i] = (pr - risk_free_rate) / ps # Color: Sharpe

            # 繪製互動圖表 (Plotly)
            fig = go.Figure()
            
            # 1. 散佈點
            fig.add_trace(go.Scatter(
                x=results[0,:], y=results[1,:],
                mode='markers',
                marker=dict(
                    size=6, color=results[2,:], colorscale='Viridis', showscale=True,
                    colorbar=dict(title="Sharpe")
                ),
                name='模擬組合',
                text=[f"Sharpe: {s:.2f}" for s in results[2,:]]
            ))
            
            # 2. 最佳點
            fig.add_trace(go.Scatter(
                x=[opt_std], y=[opt_ret],
                mode='markers+text',
                marker=dict(color='red', size=15, symbol='star'),
                name='最佳配置點',
                text=["★ Max Sharpe"],
                textposition="top center"
            ))

            fig.update_layout(
                xaxis_title="風險 (年化波動率)",
                yaxis_title="預期年化報酬率",
                height=500,
                hovermode="closest"
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"發生錯誤: {e}")

# 按鈕觸發
if st.button("🚀 開始計算", type="primary"):
    run_optimization()
