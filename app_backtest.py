import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, MACD, SMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
import warnings

warnings.filterwarnings('ignore')

# 设置网页宽屏布局
st.set_title = "BTC 量化回测引擎"
st.set_page_config(page_title="BTC 量化回测引擎", layout="wide")

# ==========================================
# 🛡️ 核心防封禁机制：使用缓存装饰器
# 无论参数怎么变，这个函数只会在启动时执行一次！
# ==========================================
@st.cache_data(ttl=3600) # 缓存有效期 1 小时
def download_and_prepare_data():
    btc = yf.Ticker("BTC-USD")
    df = btc.history(period="5y", interval="1d")
    df.index = df.index.tz_localize(None)
    
    df['EMA20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
    df['EMA50'] = EMAIndicator(close=df['Close'], window=50).ema_indicator()
    df['MACD'] = MACD(close=df['Close'], window_fast=12, window_slow=26, window_sign=9).macd_diff()
    
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_H'] = bb.bollinger_hband()
    df['BB_L'] = bb.bollinger_lband()
    df['ATR'] = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()
    
    df['OBV'] = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
    df['OBV_MA'] = SMAIndicator(close=df['OBV'], window=20).sma_indicator()
    df['Volume_MA20'] = SMAIndicator(close=df['Volume'], window=20).sma_indicator()
    
    df.dropna(inplace=True)
    return df

# ==========================================
# 📈 进化版回测引擎 (百分比滚利复投算法)
# ==========================================
def run_backtest(df, initial_capital, max_loss_pct, rr_ratio, use_ema_filter, use_vol_filter):
    trades = []
    in_position = False
    position_type = None  
    entry_price = 0
    sl, tp = 0, 0
    
    wins, losses = 0, 0
    capital = initial_capital
    equity_curve = [{"Date": df.index[0], "Capital": capital}]
    
    for index, row in df.iterrows():
        # --- 持仓监控 (以真实涨跌幅 % 更新复利本金) ---
        if in_position:
            if position_type == 'LONG':
                if row['Low'] <= sl:
                    # 计算真实价格跌幅
                    trade_return = (sl - entry_price) / entry_price
                    capital = capital * (1 + trade_return)
                    losses += 1
                    trades.append({
                        "交易日期": index.date(),
                        "交易类型": "🔴 多单止损",
                        "入场价格": round(entry_price, 2),
                        "出场价格": round(sl, 2),
                        "单笔盈亏": f"{trade_return:.2%}",
                        "当前结余": round(capital, 2)
                    })
                    in_position = False
                elif row['High'] >= tp:
                    # 计算真实价格涨幅
                    trade_return = (tp - entry_price) / entry_price
                    capital = capital * (1 + trade_return)
                    wins += 1
                    trades.append({
                        "交易日期": index.date(),
                        "交易类型": "🟢 多单止盈",
                        "入场价格": round(entry_price, 2),
                        "出场价格": round(tp, 2),
                        "单笔盈亏": f"+{trade_return:.2%}",
                        "当前结余": round(capital, 2)
                    })
                    in_position = False

            elif position_type == 'SHORT':
                if row['High'] >= sl:
                    # 空单止损亏损率
                    trade_return = (entry_price - sl) / entry_price
                    capital = capital * (1 + trade_return)
                    losses += 1
                    trades.append({
                        "交易日期": index.date(),
                        "交易类型": "🔴 空单止损",
                        "入场价格": round(entry_price, 2),
                        "出场价格": round(sl, 2),
                        "单笔盈亏": f"{trade_return:.2%}",
                        "当前结余": round(capital, 2)
                    })
                    in_position = False
                elif row['Low'] <= tp:
                    # 空单止盈利润率
                    trade_return = (entry_price - tp) / entry_price
                    capital = capital * (1 + trade_return)
                    wins += 1
                    trades.append({
                        "交易日期": index.date(),
                        "交易类型": "🟢 空单止盈",
                        "入场价格": round(entry_price, 2),
                        "出场价格": round(tp, 2),
                        "单笔盈亏": f"+{trade_return:.2%}",
                        "当前结余": round(capital, 2)
                    })
                    in_position = False
            
            # 记录持仓期间的每日净值变化
            equity_curve.append({"Date": index, "Capital": capital})
            continue 
            
        # --- 信号扫描 ---
        price, atr = row['Close'], row['ATR']
        score = 0
        if price > row['EMA20']: score += 1
        elif price < row['EMA20']: score -= 1
        if row['EMA20'] > row['EMA50']: score += 1
        elif row['EMA20'] < row['EMA50']: score -= 1
        if row['MACD'] > 0: score += 1
        else: score -= 1
        if row['OBV'] > row['OBV_MA']: score += 1
        else: score -= 1
        
        # 门神过滤条件
        long_allowed = (price > row['EMA50']) if use_ema_filter else True
        short_allowed = (price < row['EMA50']) if use_ema_filter else True
        vol_allowed = (row['Volume'] > row['Volume_MA20']) if use_vol_filter else True

        if score >= 3 and long_allowed and vol_allowed:
            entry_price = price
            sl_math = entry_price - (1.5 * atr)
            sl_hard = entry_price * (1 - max_loss_pct)
            sl = max(sl_math, sl_hard, row['BB_L'])
            risk = entry_price - sl
            if risk > 0:
                tp = entry_price + (rr_ratio * risk)
                in_position, position_type = True, 'LONG'
                
        elif score <= -3 and short_allowed and vol_allowed:
            entry_price = price
            sl_math = entry_price + (1.5 * atr)
            sl_hard = entry_price * (1 + max_loss_pct)
            sl = min(sl_math, sl_hard, row['BB_H'])
            risk = sl - entry_price
            if risk > 0:
                tp = entry_price - (rr_ratio * risk)
                in_position, position_type = True, 'SHORT'
                
        # 记录空仓期间的每日净值变化
        equity_curve.append({"Date": index, "Capital": capital})

    return trades, wins, losses, capital, pd.DataFrame(equity_curve)

# ==========================================
# UI 界面构建
# ==========================================
st.title("📈 BTC 量化门神策略 - 实时热更新回测控制台")

# 获取数据 (带进度提示)
with st.spinner('首次从雅虎拉取全量数据中... (之后修改参数将进入秒级缓存运行)'):
    df_market = download_and_prepare_data()

# 左侧控制面板 (Sidebar)
st.sidebar.header("⚙️ 策略参数设置")
# 💡 核心修复：注入唯一 `key` 解决 StreamlitDuplicateElementId 报错，并实现无按钮热重载
initial_capital = st.sidebar.number_input("💵 初始本金 (USD)", value=10000, step=1000, key="initial_capital_input")
max_loss_pct = st.sidebar.slider("🛡️ 单笔最大亏损比例 (%)", min_value=1.0, max_value=20.0, value=8.0, step=0.5, key="max_loss_slider") / 100.0
rr_ratio = st.sidebar.slider("⚖️ 止盈止损比 (盈亏比)", min_value=1.0, max_value=5.0, value=1.5, step=0.1, key="rr_ratio_slider")

st.sidebar.markdown("---")
st.sidebar.header("🚪 门神过滤器")
use_ema_filter = st.sidebar.toggle("启用 EMA50 趋势过滤", value=True, key="ema_filter_toggle")
use_vol_filter = st.sidebar.toggle("启用 20日均量放量确认", value=True, key="vol_filter_toggle")

# 直接执行计算逻辑，移除了 manual button，实现丝滑热更新
trades, wins, losses, final_capital, df_equity = run_backtest(
    df_market, initial_capital, max_loss_pct, rr_ratio, use_ema_filter, use_vol_filter
)

total_trades = wins + losses
win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
total_return = ((final_capital - initial_capital) / initial_capital) * 100

# 顶部数据卡片
col1, col2, col3, col4 = st.columns(4)
col1.metric("总交易次数", f"{total_trades} 次")
col2.metric("策略胜率", f"{win_rate:.2f}%")
col3.metric("最终资金结余", f"${final_capital:,.2f}", f"{total_return:+.2f}%")
col4.metric("盈亏单数", f"赚 {wins} / 亏 {losses}")

# 资金曲线图表
st.subheader("💰 资金复利增长曲线")
st.line_chart(df_equity.set_index("Date")["Capital"])

# 交易流水明细
st.subheader("📜 详细交易流水")
if trades:
    df_trades = pd.DataFrame(trades)
    # 💡 核心修复：重塑 Index 数组，使其完全从 1 开始
    df_trades.index = np.arange(1, len(df_trades) + 1)
    st.dataframe(df_trades, use_container_width=True)
else:
    st.info("当前参数下无任何交易信号。")