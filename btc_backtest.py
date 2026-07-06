import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, MACD, SMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
import warnings

# 忽略 pandas 的一些警告输出，保持界面整洁
warnings.filterwarnings('ignore')

def download_and_prepare_data():
    print("⏳ 正在从雅虎财经拉取 BTC 过去 5 年历史数据...")
    btc = yf.Ticker("BTC-USD")
    df = btc.history(period="5y", interval="1d")
    df.index = df.index.tz_localize(None)
    
    print("🧮 正在计算全量历史技术指标 (EMA, MACD, OBV, ATR, BB)...")
    df['EMA20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
    df['EMA50'] = EMAIndicator(close=df['Close'], window=50).ema_indicator()
    df['MACD'] = MACD(close=df['Close'], window_fast=12, window_slow=26, window_sign=9).macd_diff()
    
    # 布林带与 ATR
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_H'] = bb.bollinger_hband()
    df['BB_L'] = bb.bollinger_lband()
    df['ATR'] = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()
    
    # OBV 及均线
    df['OBV'] = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
    df['OBV_MA'] = SMAIndicator(close=df['OBV'], window=20).sma_indicator()
    
    # 【修复核心】：补上 20日移动平均成交量的计算，与实盘代码完全对齐
    df['Volume_MA20'] = SMAIndicator(close=df['Volume'], window=20).sma_indicator()
    
    df.dropna(inplace=True)
    return df

def run_backtest(df):
    print(f"🚀 开始执行回测引擎 (总交易天数: {len(df)} 天)...\n")
    
    trades = []
    in_position = False
    position_type = None  
    entry_price = 0
    sl = 0
    tp = 0
    
    wins = 0
    losses = 0
    total_profit_pct = 0.0
    
    # ========================================================
    # 💡 投研参数配置区 (对齐最初赚钱的黄金配比)
    # ========================================================
    MAX_LOSS_PCT = 0.08  # 单笔现货绝对亏损死上限 (8%)
    RR_RATIO = 1.5       # 初始终极盈亏比目标 (1.5 倍)
    # ========================================================
    
    for index, row in df.iterrows():
        # ================= 1. 持仓监控阶段 (回归纯净风控) =================
        if in_position:
            # 悲观原则：先测止损，后测止盈
            if position_type == 'LONG':
                if row['Low'] <= sl:
                    loss_pct = (sl - entry_price) / entry_price
                    total_profit_pct += loss_pct
                    losses += 1
                    trades.append(f"🔴 {index.date()} [多单止损] 入场:{entry_price:.0f} -> 出场:{sl:.0f} ({loss_pct:.2%})")
                    in_position = False
                elif row['High'] >= tp:
                    profit_pct = (tp - entry_price) / entry_price
                    total_profit_pct += profit_pct
                    wins += 1
                    trades.append(f"🟢 {index.date()} [多单止盈] 入场:{entry_price:.0f} -> 出场:{tp:.0f} (+{profit_pct:.2%})")
                    in_position = False

            elif position_type == 'SHORT':
                if row['High'] >= sl:
                    loss_pct = (entry_price - sl) / entry_price
                    total_profit_pct += loss_pct
                    losses += 1
                    trades.append(f"🔴 {index.date()} [空单止损] 入场:{entry_price:.0f} -> 出场:{sl:.0f} ({loss_pct:.2%})")
                    in_position = False
                elif row['Low'] <= tp:
                    profit_pct = (entry_price - tp) / entry_price
                    total_profit_pct += profit_pct
                    wins += 1
                    trades.append(f"🟢 {index.date()} [空单止盈] 入场:{entry_price:.0f} -> 出场:{tp:.0f} (+{profit_pct:.2%})")
                    in_position = False
            continue 
            
        # ================= 2. 信号扫描阶段 (趋势过滤器保留) =================
        price = row['Close']
        atr = row['ATR']
        
        score = 0
        if price > row['EMA20']: score += 1
        elif price < row['EMA20']: score -= 1
        if row['EMA20'] > row['EMA50']: score += 1
        elif row['EMA20'] < row['EMA50']: score -= 1
        if row['MACD'] > 0: score += 1
        else: score -= 1
        if row['OBV'] > row['OBV_MA']: score += 1
        else: score -= 1
        
        # 多头信号
        if score >= 3:
            # 黄金趋势门神：只在 50日均线之上做多 + 20日均量放量确认
            if price > row['EMA50'] and row['Volume'] > row['Volume_MA20']:
                entry_price = price
                sl_math = entry_price - (1.5 * atr)
                sl_hard = entry_price * (1 - MAX_LOSS_PCT)
                sl = max(sl_math, sl_hard, row['BB_L'])
                
                risk = entry_price - sl
                if risk > 0:
                    tp = entry_price + (RR_RATIO * risk)
                    in_position = True
                    position_type = 'LONG'
                
        # 空头信号
        elif score <= -3:
            # 黄金趋势门神：只在 50日均线下方做空 + 20日均量放量确认
            if price < row['EMA50'] and row['Volume'] > row['Volume_MA20']:
                entry_price = price
                sl_math = entry_price + (1.5 * atr)
                sl_hard = entry_price * (1 + MAX_LOSS_PCT)
                sl = min(sl_math, sl_hard, row['BB_H'])
                
                risk = sl - entry_price
                if risk > 0:
                    tp = entry_price - (RR_RATIO * risk)
                    in_position = True
                    position_type = 'SHORT'

    # ================= 3. 打印全量报告 =================
    print(f"📜 历史完整交易流水对账单 (总计 {len(trades)} 条事件记录)：")
    print("-" * 75)
    for t in trades:  
        print(t)
    print("-" * 75)

    total_trades = wins + losses
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    
    print("\n" + "="*50)
    print("📊 优化版：量化风控引擎【门神过滤体】5 年深度回测")
    print("="*50)
    print(f"• 核心配置: 50日线趋势过滤 + 20日放量进场确认 (彻底关闭保本移动)")
    print(f"• 总交易次数 (真实算输赢): {total_trades} 次")
    print(f"• 成功止盈 (赚 {RR_RATIO} 倍): {wins} 次")
    print(f"• 触发真实止损 (亏 1.0 倍): {losses} 次")
    print(f"🏆 策略真实胜率: {win_rate:.2f}%")
    print(f"💰 累计无杠杆净回报: {total_profit_pct:.2%}")
    print("="*50)


if __name__ == "__main__":
    df = download_and_prepare_data()
    run_backtest(df)