import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.optimize as sco
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# === 1. 設定網頁標題 ===
st.set_page_config(page_title="台股效率前沿計算機", page_icon="📈", layout="wide")
st.title("📈 台股效率前沿計算機")
st.write("自動計算：Max Sharpe、甜蜜點 (左上角最佳平衡)、指定區間極大值")

# === 2. 側邊欄參數 ===
st.sidebar.header("參數設定")
default_tickers = "5403.TWO, 2412.TW, 5903.TWO, 6803.TWO"
tickers_input = st.sidebar.text_area("輸入代號 (逗號隔開)", default_tickers)

col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("開始日期", value=pd.to_datetime("2025-01-01"))
end_date = col2.date_input("結束日期", value=pd.to_datetime("today"))

rf_input = st.sidebar.number_input("無風險利率 (%)", value=2.0, step=0.1)
risk_free_rate = rf_input / 100

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 指定風險區間")
c_min, c_max = st.sidebar.columns(2)
req_min_risk = c_min.number_input("最低風險", value=0.04, step=0.01)
req_max_risk = c_max.number_input("最高風險", value=0.16, step=0.01)

# === 3. 核心運算 ===
def run_optimization():
    tickers = [t.strip().upper() for t in tickers_input.split(',')]
    
    with st.spinner("資料下載與運算中..."):
        try:
            # 下載資料
            data = yf.download(tickers, start=start_date, end=end_date)['Close']
            if data.empty:
                st.error("❌ 找不到資料")
                return
            data = data.dropna(axis=1, how='all').dropna()
            if data.shape[1] < 2:
                st.error("⚠️ 股票少於 2 檔")
                return
                
            used_tickers = data.columns.tolist()
            st.success(f"✅ 成功載入 {len(used_tickers)} 檔")
            
            # 計算年化數據 (修正重複年化 bug)
            returns = data.pct_change().dropna()
            mean_returns = returns.mean() * 252
            cov_matrix = returns.cov() * 252
            num_assets = len(used_tickers)

            def portfolio_performance(weights):
                weights = np.array(weights)
                ret = np.sum(mean_returns * weights)
                std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                return ret, std

            def neg_sharpe_ratio(weights):
                p_ret, p_std = portfolio_performance(weights)
                return -(p_ret - risk_free_rate) / p_std if p_std != 0 else 0

            # 1. Max Sharpe
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for _ in range(num_assets))
            init_guess = num_assets * [1. / num_assets,]
            result = sco.minimize(neg_sharpe_ratio, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
            sharpe_w = result.x
            opt_ret, opt_std = portfolio_performance(sharpe_w)

            # 2. 蒙地卡羅模擬
            sim_n = 5000  
            results = np.zeros((3, sim_n)) # 修正處：確保括號閉合
            weights_record = [] 
            
            for i in range(sim_n):
                w = np.random.random(num_assets)
                w /= np.sum(w)
                weights_record.append(w)
                pr, ps = portfolio_performance(w)
                results[0,i] = ps
                results[1,i] = pr
                results[2,i] = (pr - risk_free_rate) / ps

            # 3. 甜蜜點 (Utopia Point 距離法 - 鎖定左上角)
            min_std_global = np.min(results[0, :])
            max_ret_global = np.max(results[1, :])
            dists = np.sqrt((results[0,:] - min_std_global)**2 + (results[1,:] - max_ret_global)**2)
            sweet_idx = np.argmin(dists)
            sweet_ret = results[1, sweet_idx]
            sweet_std = results[0, sweet_idx]
            sweet_w = weights_record[sweet_idx]

            # 4. 指定區間最佳點
            mask = (results[0, :] >= req_min_risk) & (results[0, :] <= req_max_risk)
            range_exists = False
            range_ret, range_std, range_w = 0, 0, []

            if np.any(mask):
                range_exists = True
                valid_idx = np.where(mask)[0]
                best_sub_idx = np.argmax(results[1, valid_idx])
                best_global_idx = valid_idx[best_sub_idx]
                range_ret = results[1, best_global_idx]
                range_std = results[0, best_global_idx]
                range_w = weights_record[best_global_idx]

            # === 顯示結果 ===
            st.markdown("---")
            data_dict = {
                "股票": used_tickers,
                "🔴 Max Sharpe": [f"{w:.1%}" for w in sharpe_w],
                "🔶 甜蜜點": [f"{w:.1%}" for w in sweet_w],
                "raw_sharpe": sharpe_w, "raw_sweet": sweet_w
            }
            if range_exists:
                data_dict[f"💜 區間最佳"] = [f"{w:.1%}" for w in range_w]
                data_dict["raw_range"] = range_w

            df = pd.DataFrame(data_dict)
            st.table(df[[c for c in df.columns if not c.startswith('raw_')]])

            # 圓餅圖
            titles = ['🔴 Max Sharpe', '🔶 甜蜜點']
            specs = [[{'type':'domain'}, {'type':'domain'}]]
            if range_exists:
                titles.append(f"💜 區間 ({req_min_risk}-{req_max_risk})")
                specs = [[{'type':'domain'}, {'type':'domain'}, {'type':'domain'}]]
            
            fig_p = make_subplots(rows=1, cols=len(titles), specs=specs, subplot_titles=titles)
            fig_p.add_trace(go.Pie(labels=df["股票"], values=df["raw_sharpe"], name="Max Sharpe"), 1, 1)
            fig_p.add_trace(go.Pie(labels=df["股票"], values=df["raw_sweet"], name="Sweet Spot"), 1, 2)
            if range_exists:
                fig_p.add_trace(go.Pie(labels=df["股票"], values=df["raw_range"], name="Range Best"), 1, 3)
            fig_p.update_layout(height=300, margin=dict(t=30, b=0, l=0, r=0))
            st.plotly_chart(fig_p, use_container_width=True)

            # 散佈圖
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=results[0,:], y=results[1,:], mode='markers',
                marker=dict(size=5, color=results[2,:], colorscale='Viridis', showscale=True),
                name='模擬點', text=[f"R:{r:.2f}" for r in results[0,:]]))
            
            fig.add_trace(go.Scatter(x=[opt_std], y=[opt_ret], mode='markers+text',
                marker=dict(color='red', size=18, symbol='star'), name='Max Sharpe', text=["★ Max Sharpe"], textposition="top center"))
            
            fig.add_trace(go.Scatter(x=[sweet_std], y=[sweet_ret], mode='markers+text',
                marker=dict(color='orange', size=16, symbol='diamond', line=dict(width=2, color='white')),
                name='甜蜜點', text=["🔶 甜蜜點"], textposition="top center"))

            if range_exists:
                fig.add_trace(go.Scatter(x=[range_std], y=[range_ret], mode='markers+text',
                    marker=dict(color='purple', size=15, symbol='square', line=dict(width=2, color='white')),
                    name='區間最佳', text=["💜 區間"], textposition="bottom center"))
                fig.add_vrect(x0=req_min_risk, x1=req_max_risk, fillcolor="purple", opacity=0.1, layer="below", line_width=0)

            fig.update_layout(height=600, xaxis_title="風險 (年化波動)", yaxis_title="報酬 (年化)")
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"發生錯誤: {e}")

if st.button("🚀 開始計算", type="primary"):
    run_optimization()