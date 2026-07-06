import yfinance as yf
import pandas as pd
import json
import asyncio
import ccxt
import requests
import re
import socket
import time
from datetime import time as dt_time
from zoneinfo import ZoneInfo
from openai import OpenAI
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

import ta
from ta.trend import EMAIndicator, MACD, SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

# ================= 配置区 =================
# 从同目录下的 telegram_config.py 文件中导入变量
from telegram_config import BOT_TOKEN, ALLOWED_USER_ID
# ==========================================

# 锁定东京时区
TOKYO_TZ = ZoneInfo("Asia/Tokyo")

# --- 趋势定性 (区分日线与周线定义) ---
def trend_status(price, ema20, ema50, ema200, is_weekly=False):
    if pd.isna(ema200): return "数据不足"
    short = "多头排列 (偏强)" if price > ema20 else "空头排列 (偏弱)"
    mid = "多头占优" if price > ema50 else "空头压制"
    if is_weekly:
        long = "宏观牛市 (站稳周线牛熊分水岭)" if price > ema200 else "宏观熊市 (跌破周线牛熊分水岭)"
    else:
        long = "阶段性强势 (高于日线年线)" if price > ema200 else "阶段性弱势 (低于日线年线)"
    return f"短线{short}，中线{mid}，长线大趋势为{long}"

def bollinger_position(price, upper, lower):
    if pd.isna(upper) or pd.isna(lower): return "数据不足"
    mid = (upper + lower) / 2
    band_range = upper - lower
    if price < lower + 0.1 * band_range: return "极度贴近下轨 (存在超卖反弹预期)"
    elif price > upper - 0.1 * band_range: return "极度贴近上轨 (存在超买回调风险)"
    elif price > mid: return "位于中轨上方 (偏强震荡)"
    else: return "位于中轨下方 (偏弱震荡)"

# --- 新增：OBV 主力吸筹/派发定性 ---
def obv_status(obv_current, obv_ma):
    if pd.isna(obv_current) or pd.isna(obv_ma): return "数据不足"
    if obv_current > obv_ma: return "OBV 呈上升趋势 (资金暗中吸筹)"
    else: return "OBV 呈下降趋势 (资金持续流出)"

def momentum_status(macd_hist, current_vol, avg_vol, obv_stat):
    macd_label = "正值 (多头动能发散)" if macd_hist > 0 else "负值 (空头动能主导)"
    vol_label = "放量" if current_vol > avg_vol else "缩量"
    return f"MACD {macd_label}，成交量 {vol_label}，{obv_stat}"

def calculate_support_resistance(current_price, levels_dict):
    supports, resistances = [], []
    for name, value in levels_dict.items():
        if pd.isna(value): continue
        pct_diff = (value - current_price) / current_price * 100
        if current_price >= value: supports.append((name, value, pct_diff))
        else: resistances.append((name, value, pct_diff))
            
    supports.sort(key=lambda x: x[1], reverse=True)
    resistances.sort(key=lambda x: x[1])
    supports_formatted = [f"{item[0]}: {item[1]:.2f} (距离 {item[2]:+.2f}%)" for item in supports]
    resistances_formatted = [f"{item[0]}: {item[1]:.2f} (距离 {item[2]:+.2f}%)" for item in resistances]
    return supports_formatted, resistances_formatted

def rsi_status(rsi_value):
    if pd.isna(rsi_value): return "数据不足"
    if rsi_value >= 70: return f"{rsi_value:.2f} (极度超买)"
    elif rsi_value <= 30: return f"{rsi_value:.2f} (极度超卖)"
    elif rsi_value > 50: return f"{rsi_value:.2f} (中性偏强)"
    else: return f"{rsi_value:.2f} (中性偏弱)"

# ================= 独立功能数据接口 =================

def get_live_funding_rate():
    try:
        exchange = ccxt.binance({'timeout': 3000})
        funding = exchange.fetch_funding_rate('BTC/USDT:USDT')
        rate_pct = funding['fundingRate'] * 100
        status = "多头极其强势 (多付空)" if rate_pct > 0.01 else "空头强势 (空付多)" if rate_pct < 0 else "市场情绪中性"
        return f"📈 <b>币安合约实时情绪</b>\n\n• 当前资金费率: <b>{rate_pct:.4f}%</b>\n• 情绪判断: {status}"
    except Exception as e:
        return f"❌ 获取资金费率失败: {e}"

def get_fear_and_greed():
    try:
        resp = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5).json()
        val = resp['data'][0]['value']
        classification = resp['data'][0]['value_classification']
        return f"🧭 <b>全网恐慌与贪婪指数</b>\n\n• 当前指数: <b>{val}</b> (0极度恐慌 - 100极度贪婪)\n• 市场状态: <b>{classification}</b>"
    except Exception as e:
        return f"❌ 获取恐慌贪婪指数失败: {e}"

def get_24h_summary():
    try:
        exchange = ccxt.binance({'timeout': 3000})
        ticker = exchange.fetch_ticker('BTC/USDT')
        return (f"📊 <b>BTC 24小时概况 (Binance)</b>\n\n"
                f"• 最新价: <b>{ticker['last']}</b> USDT\n"
                f"• 24h涨跌: {ticker['percentage']:.2f}%\n"
                f"• 最高价: {ticker['high']}\n"
                f"• 最低价: {ticker['low']}\n"
                f"• 24h现货成交: {ticker['baseVolume']:.2f} BTC")
    except Exception as e: 
        return f"❌ 获取24小时概况失败: {e}"

def fetch_sentiment_data():
    sentiment = {"funding": "获取失败", "fng": "获取失败"}
    try:
        ex = ccxt.binance({'timeout': 3000})
        rate = ex.fetch_funding_rate('BTC/USDT:USDT')['fundingRate'] * 100
        sentiment['funding'] = f"{rate:.4f}% ({'多付空' if rate > 0 else '空付多'})"
    except: pass
    try:
        resp = requests.get("https://api.alternative.me/fng/?limit=1", timeout=3).json()
        sentiment['fng'] = f"{resp['data'][0]['value']} ({resp['data'][0]['value_classification']})"
    except: pass
    return sentiment

# ================= 核心量化模型与分析引擎 =================

def get_market_data():
    btc = yf.Ticker("BTC-USD")
    
    # 1. 下载日线历史底座
    df_daily = btc.history(period="1y", interval="1d")
    df_daily.index = df_daily.index.tz_localize(None)
    
    # 计算基础日线技术指标
    df_daily['EMA_20'] = EMAIndicator(close=df_daily['Close'], window=20).ema_indicator()
    df_daily['EMA_50'] = EMAIndicator(close=df_daily['Close'], window=50).ema_indicator()
    df_daily['EMA_200'] = EMAIndicator(close=df_daily['Close'], window=200).ema_indicator()
    df_daily['RSI_14'] = RSIIndicator(close=df_daily['Close'], window=14).rsi()
    df_daily['MACD_Histogram'] = MACD(close=df_daily['Close'], window_fast=12, window_slow=26, window_sign=9).macd_diff()
    
    indicator_bb = BollingerBands(close=df_daily['Close'], window=20, window_dev=2)
    df_daily['BB_High'] = indicator_bb.bollinger_hband()
    df_daily['BB_Low'] = indicator_bb.bollinger_lband()
    
    # ⚔️ 核心修复点 1：将所有原先错误的 df_d 统一矫正为 df_daily 
    df_daily['Volume_MA20'] = SMAIndicator(close=df_daily['Volume'], window=20).sma_indicator()
    df_daily['ATR'] = AverageTrueRange(high=df_daily['High'], low=df_daily['Low'], close=df_daily['Close'], window=14).average_true_range()
    
    df_daily['OBV'] = OnBalanceVolumeIndicator(close=df_daily['Close'], volume=df_daily['Volume']).on_balance_volume()
    df_daily['OBV_MA'] = SMAIndicator(close=df_daily['OBV'], window=20).sma_indicator()
    
    # 清理由于计算均线产生的历史空行
    df_daily.dropna(inplace=True)
    
    # 异常数据源检查
    pct_change = df_daily['Close'].pct_change().abs().iloc[-1]
    if pct_change > 0.50: raise Exception("检测到单日振幅>50%，Yahoo数据源可能遭遇插针污染，拒绝分析。")
    
    latest_d = df_daily.iloc[-1]

    # 【风控核心】计算昨日 Pivot Points 枢轴点
    prev_d = df_daily.iloc[-2]
    PP = (prev_d['High'] + prev_d['Low'] + prev_d['Close']) / 3
    R1 = 2 * PP - prev_d['Low']
    S1 = 2 * PP - prev_d['High']

    # 2. 下载周线历史底座
    df_weekly = btc.history(period="5y", interval="1wk")
    df_weekly.index = df_weekly.index.tz_localize(None)
    df_weekly['EMA_20'] = EMAIndicator(close=df_weekly['Close'], window=20).ema_indicator()
    df_weekly['EMA_50'] = EMAIndicator(close=df_weekly['Close'], window=50).ema_indicator()
    df_weekly['EMA_200'] = EMAIndicator(close=df_weekly['Close'], window=200).ema_indicator()
    df_weekly['RSI_14'] = RSIIndicator(close=df_weekly['Close'], window=14).rsi()
    df_weekly.dropna(inplace=True)
    latest_w = df_weekly.iloc[-1]

    # 3. 实时盘口深度抓取 (Binance)
    order_book_status = "数据获取失败"
    imbalance = 1.0
    try:
        exchange = ccxt.binance({'timeout': 3000})
        ticker = exchange.fetch_ticker('BTC/USDT')
        current_price = ticker['last']
        data_time = pd.to_datetime(ticker['timestamp'], unit='ms').tz_localize('UTC').tz_convert('Asia/Tokyo').strftime('%Y-%m-%d %H:%M:%S')
        price_source = "Binance (实时)"
        
        orderbook = exchange.fetch_order_book('BTC/USDT', limit=50)
        bids_vol = sum(bid[1] for bid in orderbook['bids']) 
        asks_vol = sum(ask[1] for ask in orderbook['asks']) 
        imbalance = bids_vol / asks_vol if asks_vol > 0 else 1.0
        
        if imbalance > 1.5: order_book_status = "买盘挂单极强 (下方托盘厚实)"
        elif imbalance < 0.66: order_book_status = "卖盘挂单沉重 (上方抛压如山)"
        else: order_book_status = "买卖盘口势均力敌"

    except Exception as e:
        current_price = round(latest_d['Close'], 2)
        data_time = pd.Timestamp.now(tz='Asia/Tokyo').strftime('%Y-%m-%d %H:%M:%S')
        price_source = "Yahoo (延迟)"

    # 4. 获取外部清算与情绪数据
    sentiment = fetch_sentiment_data()

    all_levels = {
        "日线 EMA20": latest_d['EMA_20'], "日线 EMA50": latest_d['EMA_50'], "日线 EMA200": latest_d['EMA_200'],
        "日线布林带上轨": latest_d['BB_High'], "日线布林带下轨": latest_d['BB_Low'],
        "周线 EMA20": latest_w['EMA_20'], "周线 EMA50": latest_w['EMA_50'], "周线 EMA200": latest_w['EMA_200']
    }
    supports, resistances = calculate_support_resistance(current_price, all_levels)

    # 安全提取情绪数值用于打分引擎
    fng_val, funding_val = 50, 0.0
    if '获取失败' not in sentiment['fng']:
        try: fng_val = int(sentiment['fng'].split(' ')[0])
        except: pass
    if '获取失败' not in sentiment['funding']:
        try: funding_val = float(sentiment['funding'].split('%')[0])
        except: pass

    # ⚔️ 核心修复点 2：将完整的、清洗后的全维度上下文打包在函数尾部统一 return，完美对接打分引擎
    context = {
        "Asset": "BTC-USDT",
        "Current_Price": current_price,
        "Data_Timestamp_JST": data_time,
        "Price_Source": price_source,
        "Explicit_Trend_Labels": {
            "Daily_Trend": trend_status(current_price, latest_d['EMA_20'], latest_d['EMA_50'], latest_d['EMA_200'], False),
            "Weekly_Trend": trend_status(current_price, latest_w['EMA_20'], latest_w['EMA_50'], latest_w['EMA_200'], True)
        },
        "Explicit_Momentum_Labels": {
            "Bollinger_Position": bollinger_position(current_price, latest_d['BB_High'], latest_d['BB_Low']),
            "Daily_RSI": rsi_status(latest_d['RSI_14']),
            "Weekly_RSI": rsi_status(latest_w['RSI_14']),
            "Volume_and_MACD": momentum_status(latest_d['MACD_Histogram'], latest_d['Volume'], latest_d['Volume_MA20'], obv_status(latest_d['OBV'], latest_d['OBV_MA']))
        },
        "Volatility_and_Sentiment": {
            "Daily_ATR_Volatility": round(latest_d['ATR'], 2),
            "Order_Book_Imbalance": order_book_status,
            "Funding_Rate": sentiment['funding'],
            "Fear_Greed_Index": sentiment['fng']
        },
        "Explicit_Support_and_Resistance": {
            "Below_Price_Supports": supports,
            "Above_Price_Resistances": resistances
        },
        "Raw_Data_For_Scoring": {
            "EMA20": latest_d['EMA_20'],
            "EMA50": latest_d['EMA_50'],
            "MACD": latest_d['MACD_Histogram'],
            "OBV_Up": bool(latest_d['OBV'] > latest_d['OBV_MA']),
            "Volume": float(latest_d['Volume']),
            "Prev_Volume": float(df_daily['Volume'].iloc[-2]),  # 👈 新增这一行：提取昨日完整成交量          
            "Volume_MA20": float(latest_d['Volume_MA20']), 
            "ATR": latest_d['ATR'],
            "BB_H": latest_d['BB_High'],
            "BB_L": latest_d['BB_Low'],
            "OrderBookRatio": imbalance,
            "FNG_Value": fng_val,
            "Funding_Value": funding_val,
            "Pivot_PP": PP,
            "Pivot_R1": R1,
            "Pivot_S1": S1
        }
    }
    return context

# --- 🔥 全新升级：【门神过滤体】实盘量化执行大脑 (防时差与负风险版) ---
def generate_trading_signal(ctx):
    price = ctx["Current_Price"]
    raw = ctx["Raw_Data_For_Scoring"]
    atr = raw["ATR"]
    
    # 💡 【同步回测圣杯参数】
    MAX_LOSS_PCT = 0.08  # 铁律：最大现货亏损 8%
    RR_RATIO = 1.5       # 铁律：强行 1.5 倍盈亏比
    
    # 【实盘 6 因子打分引擎】
    score = 0
    if price > raw["EMA20"]: score += 1
    elif price < raw["EMA20"]: score -= 1
    if raw["EMA20"] > raw["EMA50"]: score += 1
    elif raw["EMA20"] < raw["EMA50"]: score -= 1
    if raw["MACD"] > 0: score += 1
    else: score -= 1
    if raw["OBV_Up"]: score += 1
    else: score -= 1
    if raw["OrderBookRatio"] > 1.5: score += 1
    elif raw["OrderBookRatio"] < 0.66: score -= 1
    
    if raw["FNG_Value"] < 30 and raw["Funding_Value"] < 0: score += 1
    elif raw["FNG_Value"] > 75 and raw["Funding_Value"] > 0.05: score -= 1

    action, signal = "⏳ 观望 (Neutral)", "指标分歧或未满足门神过滤，空仓等待。"
    entry, sl, tp, risk = 0, 0, 0, 0
    
    # 💡 解决成交量时差陷阱：今日已放量 OR 昨日已放量，均视为有效突破
    volume_confirmed = (raw["Volume"] > raw["Volume_MA20"]) or (raw["Prev_Volume"] > raw["Volume_MA20"])

    # ================= 多头风控逻辑 =================
    # 💡 解决分数漏洞：门槛从 >=1 提高到 >=3，强制要求多头高度共振
    if score >= 3: 
        if price > raw["EMA50"] and volume_confirmed:
            action = "🚀 强力做多 (Strong Long)"
            signal = "多头高度共振且放量确认，站稳生命线，准许做多。"
            entry = price
            
            sl_math = entry - (1.5 * atr)
            sl_hard = entry * (1 - MAX_LOSS_PCT)
            sl = max(sl_math, sl_hard, raw["BB_L"] * 0.998)
            
            risk = entry - sl
            # 💡 解决负风险倒挂漏洞：确保算出来的止损位符合逻辑
            if risk > 0:
                tp = entry + (RR_RATIO * risk)
            else:
                action = "⏳ 放弃多单 (风控计算异常)"
                signal = f"底层偏多，但防插针止损位 ({sl:.0f}) 高于现价，盈亏比计算失效，放弃。"
        else:
            action = "⏳ 放弃多单 (门神拦截)"
            signal = f"底层强势 ({score}/6)，但未满足【价格&gt;EMA50】或【放量】的门神条件，防止假突破，放弃。"

    # ================= 空头风控逻辑 =================
    elif score <= -3:
        if price < raw["EMA50"] and volume_confirmed:
            action = "🩸 强力做空 (Strong Short)"
            signal = "空头高度共振且放量确认，跌破生命线，准许做空。"
            entry = price
            
            sl_math = entry + (1.5 * atr)
            sl_hard = entry * (1 + MAX_LOSS_PCT)
            sl = min(sl_math, sl_hard, raw["BB_H"] * 1.002)
            
            risk = sl - entry
            if risk > 0:
                tp = entry - (RR_RATIO * risk)
            else:
                action = "⏳ 放弃空单 (风控计算异常)"
                signal = f"底层偏空，但防插针止损位 ({sl:.0f}) 低于现价，盈亏比计算失效，放弃。"
        else:
            action = "⏳ 放弃空单 (门神拦截)"
            signal = f"底层弱势 ({score}/6)，但未满足【价格&lt;EMA50】或【放量】的门神条件，谨防反抽，放弃。"

    # ================= 输出排版 =================
    html = f"<b>🎯 终极量化判决 (多空倾向: {score}/6)</b>\n"
    html += f"• <b>执行动作：</b> {action}\n"
    html += f"• <b>逻辑归因：</b> {signal}\n\n"
    
    if "强力" in action:
        html += f"<b>⚔️ 【门神过滤体】风控操盘计划</b>\n"
        html += f"• <b>现价建仓：</b> <code>{entry:.2f}</code> 附近\n"
        html += f"• <b>🚩 铁律止损：</b> <code>{sl:.2f}</code> (防插针/8%兜底)\n"
        html += f"• <b>💎 机械止盈：</b> <code>{tp:.2f}</code> (强吃 1.5倍 盈亏比)\n"
        html += f"<i>(注：实盘请死守止损，切勿受诱惑提前移动保本)</i>"
    elif "放弃" in action:
        html += f"<b>🛡️ 拒绝理由 (管住手是交易的一部分)</b>\n"
        html += f"• 触发风控拦截：未满足回测验证的高胜率环境，强制阻止开仓。\n"
        
    return html

def format_fast_snapshot(context_dict):
    cp = context_dict["Current_Price"]
    dt = context_dict["Data_Timestamp_JST"]
    src = context_dict["Price_Source"]
    trends = context_dict["Explicit_Trend_Labels"]
    moms = context_dict["Explicit_Momentum_Labels"]
    vol = context_dict["Volatility_and_Sentiment"]
    sr = context_dict["Explicit_Support_and_Resistance"]
    
    signal_html = generate_trading_signal(context_dict)
    
    html = f"⚡ <b>毫秒级极速快照 (带严苛风控引擎)</b>\n\n"
    html += f"<b>🕒 行情快照 (JST)</b>\n• 最新报价： {cp} USDT\n• 抓取时间： {dt}\n• 行情来源： {src}\n\n"
    html += f"<b>📊 行情主基调</b>\n• 日线： {trends['Daily_Trend']}\n• 周线： {trends['Weekly_Trend']}\n\n"
    html += f"<b>📈 动能、筹码与情绪</b>\n"
    html += f"• 布林带： {moms['Bollinger_Position']}\n"
    html += f"• RSI： 日线 {moms['Daily_RSI']} | 周线 {moms['Weekly_RSI']}\n"
    html += f"• 量价： {moms['Volume_and_MACD']}\n"
    html += f"• 盘口： {vol['Order_Book_Imbalance']}\n"
    html += f"• 波动率： 日均波动 {vol['Daily_ATR_Volatility']} USDT\n"
    html += f"• 情绪： F&G {vol['Fear_Greed_Index']} | 费率 {vol['Funding_Rate']}\n\n"
    html += f"<b>🎯 核心兵家必争之地</b>\n"
    html += f"• 下方支撑：\n"
    for s in sr['Below_Price_Supports'][:2]: html += f"  • {s}\n"
    html += f"• 上方阻力：\n"
    for r in sr['Above_Price_Resistances'][:2]: html += f"  • {r}\n\n"
    html += f"〰️〰️〰️〰️〰️〰️〰️〰️〰️〰️\n"
    html += signal_html
    return html

def analyze_with_local_llm(context_dict):
    clean_ctx = {k: v for k, v in context_dict.items() if k != "Raw_Data_For_Scoring"}
    data_json = json.dumps(clean_ctx, default=str, ensure_ascii=False)
    signal_html = generate_trading_signal(context_dict)
    
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    system_prompt = f"""
    你是一个极其冷酷、只看数据的华尔街顶级对冲基金量化交易员。
    读取数据后，先用极度专业、精炼的中文写一段 150 字以内的“宏观与微观盘面点评”。
    
    点评结束后，**必须原封不动、一字不差地**将以下这套系统给出的最终操作指令抄写在报告的最下方！
    系统指令文本如下：
    {signal_html}
    
    【排版要求】只使用 <b> 加粗，使用 • 作为列表，严禁使用 ``` 任何代码块！
    """
    try:
        response = client.chat.completions.create(
            model="mistralai/ministral-3-14b-reasoning", 
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": f"当前数据：{data_json}"}],
            temperature=0.1, max_tokens=4096
        )
        content = response.choices[0].message.content.strip()
        return re.sub(r"^```[a-zA-Z]*\n?", "", content).replace("```", "").strip()
    except Exception as e: return f"<b>❌ 连接失败</b>\n{e}"

# ================= Telegram 交互与主循环 =================
def get_standard_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🚀 AI 深度分析", callback_data='run_llm'),
         InlineKeyboardButton("⚡ 极速快照", callback_data='run_fast')],
        [InlineKeyboardButton("🎯 终极判决 (信号)", callback_data='signal'),
         InlineKeyboardButton("📊 24h概况", callback_data='summary')],
        [InlineKeyboardButton("📈 资金费率", callback_data='funding'),
         InlineKeyboardButton("🧭 恐贪指数", callback_data='fng')]
    ])

async def generate_and_send_report(bot, chat_id, edit_message_id=None, use_llm=True, only_signal=False):
    try:
        loop = asyncio.get_event_loop()
        ctx = await loop.run_in_executor(None, get_market_data)
        
        # 处理不同的指令请求
        if only_signal:
            report = f"🤖 <b>系统剥离行情，纯净执行信号：</b>\n\n{generate_trading_signal(ctx)}"
        elif use_llm:
            report = await loop.run_in_executor(None, analyze_with_local_llm, ctx)
        else:
            report = format_fast_snapshot(ctx) 
            
        # ================= 🛡️ 终极 HTML 防火墙 =================
        # 匹配所有孤立的、没有正确闭合的 < 符号（即后面没有紧跟 html 标签如 b, /b, code, /code, a, /a, u, /u, i, /i 等）
        # 将它们强行物理洗白为 &lt; 防止大模型胡乱吐出数学符号导致 Telegram 崩溃
        report = re.sub(r'<(?!/?(b|code|a|u|i|pre)\b)', '&lt;', report)
        # =======================================================
            
        reply_markup = get_standard_keyboard()

        if edit_message_id:
            await bot.edit_message_text(chat_id=chat_id, message_id=edit_message_id, text=report, parse_mode="HTML", reply_markup=reply_markup)
        else:
            await bot.send_message(chat_id=chat_id, text=report, parse_mode="HTML", reply_markup=reply_markup)
    except Exception as e:
        await bot.send_message(chat_id=chat_id, text=f"❌ 运行出错: {e}")

async def cmd_run_llm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ALLOWED_USER_ID: return
    msg = await update.message.reply_text("🔄 深度数据融合中，唤醒大模型...")
    await generate_and_send_report(context.bot, ALLOWED_USER_ID, msg.message_id, use_llm=True)

async def cmd_run_fast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ALLOWED_USER_ID: return
    await generate_and_send_report(context.bot, ALLOWED_USER_ID, use_llm=False)

async def cmd_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ALLOWED_USER_ID: return
    await generate_and_send_report(context.bot, ALLOWED_USER_ID, only_signal=True)

async def cmd_funding(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ALLOWED_USER_ID: return
    await update.message.reply_text(get_live_funding_rate(), parse_mode="HTML", reply_markup=get_standard_keyboard())

async def cmd_summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ALLOWED_USER_ID: return
    await update.message.reply_text(get_24h_summary(), parse_mode="HTML", reply_markup=get_standard_keyboard())

async def cmd_fng(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ALLOWED_USER_ID: return
    await update.message.reply_text(get_fear_and_greed(), parse_mode="HTML", reply_markup=get_standard_keyboard())

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if query.from_user.id != ALLOWED_USER_ID: return
    await query.answer()
    
    if query.data == 'run_llm':
        await query.edit_message_text("🔄 正在唤醒大模型，融合新数据...")
        await generate_and_send_report(context.bot, ALLOWED_USER_ID, query.message.message_id, use_llm=True)
    elif query.data == 'run_fast':
        await generate_and_send_report(context.bot, ALLOWED_USER_ID, query.message.message_id, use_llm=False)
    elif query.data == 'signal':
        await generate_and_send_report(context.bot, ALLOWED_USER_ID, query.message.message_id, only_signal=True)
    elif query.data == 'funding':
        await context.bot.send_message(chat_id=ALLOWED_USER_ID, text=get_live_funding_rate(), parse_mode="HTML", reply_markup=get_standard_keyboard())
    elif query.data == 'summary':
        await context.bot.send_message(chat_id=ALLOWED_USER_ID, text=get_24h_summary(), parse_mode="HTML", reply_markup=get_standard_keyboard())
    elif query.data == 'fng':
        await context.bot.send_message(chat_id=ALLOWED_USER_ID, text=get_fear_and_greed(), parse_mode="HTML", reply_markup=get_standard_keyboard())

async def daily_digest(context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=ALLOWED_USER_ID, text="🌅 <b>时区例行推演：定时全维简报生成中...</b>", parse_mode="HTML")
    await generate_and_send_report(context.bot, ALLOWED_USER_ID, use_llm=True)

async def price_monitor(context: ContextTypes.DEFAULT_TYPE):
    try:
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, get_market_data)
        
        bb = data["Explicit_Momentum_Labels"]["Bollinger_Position"]
        rsi = data["Explicit_Momentum_Labels"]["Daily_RSI"]
        ob_imb = data["Volatility_and_Sentiment"]["Order_Book_Imbalance"]
        price = data["Current_Price"]
        src = data["Price_Source"]
        dt = data["Data_Timestamp_JST"]

        import ccxt
        exchange = ccxt.binance({'timeout': 2000})
        orderbook = exchange.fetch_order_book('BTC/USDT', limit=50)
        bids_vol = sum(bid[1] for bid in orderbook['bids']) 
        asks_vol = sum(ask[1] for ask in orderbook['asks']) 
        ratio = bids_vol / asks_vol if asks_vol > 0 else 1.0

        alert = ""
        if "极度" in bb: alert += f"▪️ <b>空间异动：</b>价格 {bb}\n"
        if "极度" in rsi: alert += f"▪️ <b>动能异动：</b>日线 RSI 处于 {rsi} 状态\n"
        if "极强" in ob_imb or "沉重" in ob_imb: 
            alert += f"▪️ <b>盘口异动：</b>{ob_imb}\n"
            alert += f"  - 50档累计买单: <code>{bids_vol:.2f} BTC</code>\n"
            alert += f"  - 50档累计卖单: <code>{asks_vol:.2f} BTC</code>\n"
            alert += f"  - 实测买卖比例: <code>{ratio:.2f}</code>\n"
        
        if alert:
            msg = (
                f"🚨 <b>【量化盯盘警报】</b> 🚨\n\n"
                f"<b>🕒 监控基本面</b>\n"
                f"• 最新报价：<b>{price} USDT</b>\n"
                f"• 数据来源：{src}\n"
                f"• 抓取时间：{dt}\n\n"
                f"<b>🔍 异动指标详情</b>\n"
                f"{alert}\n"
                f"<i>💡 建议点击下方 /signal 获取最新风控买卖判定。</i>"
            )
            await context.bot.send_message(chat_id=ALLOWED_USER_ID, text=msg, parse_mode="HTML")
    except Exception as e:
        print(f"⚠️ 盯盘监控运行中遭遇偶发异常: {e}")

def wait_for_internet():
    print("🌐 等待网络连接...")
    while True:
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            print("✅ 网络已就绪。")
            return
        except OSError:
            time.sleep(10)

def main():
    print("🤖 量化外挂：终极双轨风控引擎（防插针与盈亏比熔断）已启动。")
    while True:
        try:
            wait_for_internet()
            application = Application.builder().token(BOT_TOKEN).build()

            async def setup_bot(app):
                await app.bot.set_my_commands([
                    ("fast", "⚡ 极速快照 (含交易信号)"),
                    ("run", "🚀 AI 深度分析 (含交易信号)"),
                    ("signal", "🎯 纯净交易指令 (入场/风控)"),
                    ("funding", "📈 实时资金费率"),
                    ("summary", "📊 24小时盘面概况"),
                    ("fng", "🧭 恐慌贪婪指数")
                ])
            
            loop = asyncio.get_event_loop()
            loop.run_until_complete(setup_bot(application))

            job_queue = application.job_queue
            job_queue.run_daily(daily_digest, time=dt_time(hour=8, minute=0, tzinfo=TOKYO_TZ), chat_id=ALLOWED_USER_ID)
            job_queue.run_daily(daily_digest, time=dt_time(hour=19, minute=0, tzinfo=TOKYO_TZ), chat_id=ALLOWED_USER_ID)
            job_queue.run_repeating(price_monitor, interval=900, first=10, chat_id=ALLOWED_USER_ID)

            application.add_handler(CommandHandler("run", cmd_run_llm))
            application.add_handler(CommandHandler("fast", cmd_run_fast))
            application.add_handler(CommandHandler("signal", cmd_signal))
            application.add_handler(CommandHandler("funding", cmd_funding))
            application.add_handler(CommandHandler("summary", cmd_summary))
            application.add_handler(CommandHandler("fng", cmd_fng))
            application.add_handler(CallbackQueryHandler(button_callback))
            
            print("🚀 引擎进入监听状态...")
            application.run_polling()
            
        except Exception as e:
            print(f"⚠️ 发生网络或电报连接错误: {e}, 10秒后尝试重启引擎...")
            time.sleep(10)
            continue

if __name__ == "__main__":
    main()
