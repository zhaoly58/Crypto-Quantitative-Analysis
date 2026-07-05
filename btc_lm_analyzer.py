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
    """供独立按钮和命令使用"""
    try:
        exchange = ccxt.binance({'timeout': 3000})
        funding = exchange.fetch_funding_rate('BTC/USDT:USDT')
        rate_pct = funding['fundingRate'] * 100
        status = "多头极其强势 (多付空)" if rate_pct > 0.01 else "空头强势 (空付多)" if rate_pct < 0 else "市场情绪中性"
        return f"📈 <b>币安合约实时情绪</b>\n\n• 当前资金费率: <b>{rate_pct:.4f}%</b>\n• 情绪判断: {status}"
    except Exception as e:
        return f"❌ 获取资金费率失败: {e}"

def get_fear_and_greed():
    """供独立按钮和命令使用"""
    try:
        resp = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5).json()
        val = resp['data'][0]['value']
        classification = resp['data'][0]['value_classification']
        return f"🧭 <b>全网恐慌与贪婪指数</b>\n\n• 当前指数: <b>{val}</b> (0极度恐慌 - 100极度贪婪)\n• 市场状态: <b>{classification}</b>"
    except Exception as e:
        return f"❌ 获取恐慌贪婪指数失败: {e}"

def get_24h_summary():
    """供独立按钮和命令使用"""
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
    """供核心研报聚合使用"""
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
    
    # --- 日线历史底座与核心指标计算 ---
    df_daily = btc.history(period="1y", interval="1d")
    df_daily.index = df_daily.index.tz_localize(None)
    df_daily['EMA_20'] = EMAIndicator(close=df_daily['Close'], window=20).ema_indicator()
    df_daily['EMA_50'] = EMAIndicator(close=df_daily['Close'], window=50).ema_indicator()
    df_daily['EMA_200'] = EMAIndicator(close=df_daily['Close'], window=200).ema_indicator()
    df_daily['RSI_14'] = RSIIndicator(close=df_daily['Close'], window=14).rsi()
    df_daily['MACD_Histogram'] = MACD(close=df_daily['Close'], window_fast=12, window_slow=26, window_sign=9).macd_diff()
    
    indicator_bb = BollingerBands(close=df_daily['Close'], window=20, window_dev=2)
    df_daily['BB_High'] = indicator_bb.bollinger_hband()
    df_daily['BB_Low'] = indicator_bb.bollinger_lband()
    df_daily['Volume_MA20'] = SMAIndicator(close=df_daily['Volume'], window=20).sma_indicator()
    
    df_daily['ATR'] = AverageTrueRange(high=df_daily['High'], low=df_daily['Low'], close=df_daily['Close'], window=14).average_true_range()
    
    df_daily['OBV'] = OnBalanceVolumeIndicator(close=df_daily['Close'], volume=df_daily['Volume']).on_balance_volume()
    df_daily['OBV_MA'] = SMAIndicator(close=df_daily['OBV'], window=20).sma_indicator()
    
    df_daily.dropna(inplace=True)
    
    pct_change = df_daily['Close'].pct_change().abs().iloc[-1]
    if pct_change > 0.50: raise Exception("检测到单日振幅>50%，Yahoo数据源可能遭遇插针污染，拒绝分析。")
    latest_d = df_daily.iloc[-1]

    # --- 周线历史底座 ---
    df_weekly = btc.history(period="5y", interval="1wk")
    df_weekly.index = df_weekly.index.tz_localize(None)
    df_weekly['EMA_20'] = EMAIndicator(close=df_weekly['Close'], window=20).ema_indicator()
    df_weekly['EMA_50'] = EMAIndicator(close=df_weekly['Close'], window=50).ema_indicator()
    df_weekly['EMA_200'] = EMAIndicator(close=df_weekly['Close'], window=200).ema_indicator()
    df_weekly['RSI_14'] = RSIIndicator(close=df_weekly['Close'], window=14).rsi()
    df_weekly.dropna(inplace=True)
    latest_w = df_weekly.iloc[-1]

    # --- 实时数据与订单簿获取 (Binance) ---
    order_book_status = "数据获取失败"
    try:
        exchange = ccxt.binance({'timeout': 3000})
        ticker = exchange.fetch_ticker('BTC/USDT')
        current_price = ticker['last']
        data_time = pd.to_datetime(ticker['timestamp'], unit='ms').tz_localize('UTC').tz_convert('Asia/Tokyo').strftime('%Y-%m-%d %H:%M:%S')
        price_source = "Binance (实时)"
        
        orderbook = exchange.fetch_order_book('BTC/USDT', limit=50)
        bids_vol = sum(bid[1] for bid in orderbook['bids']) 
        asks_vol = sum(ask[1] for ask in orderbook['asks']) 
        imbalance = bids_vol / asks_vol if asks_vol > 0 else 1
        
        if imbalance > 1.5: order_book_status = "买盘挂单极强 (下方托盘厚实)"
        elif imbalance < 0.66: order_book_status = "卖盘挂单沉重 (上方抛压如山)"
        else: order_book_status = "买卖盘口势均力敌"

    except Exception as e:
        print(f"⚠️ 币安连接失败，降级使用雅虎数据: {e}")
        current_price = round(latest_d['Close'], 2)
        data_time = pd.Timestamp.now(tz='Asia/Tokyo').strftime('%Y-%m-%d %H:%M:%S')
        price_source = "Yahoo (延迟)"

    sentiment = fetch_sentiment_data()

    all_levels = {
        "日线 EMA20": latest_d['EMA_20'], "日线 EMA50": latest_d['EMA_50'], "日线 EMA200": latest_d['EMA_200'],
        "日线布林带上轨": latest_d['BB_High'], "日线布林带下轨": latest_d['BB_Low'],
        "周线 EMA20": latest_w['EMA_20'], "周线 EMA50": latest_w['EMA_50'], "周线 EMA200": latest_w['EMA_200']
    }
    supports, resistances = calculate_support_resistance(current_price, all_levels)

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
        }
    }
    return context

def generate_algo_prediction(context_dict):
    trends = context_dict["Explicit_Trend_Labels"]
    moms = context_dict["Explicit_Momentum_Labels"]
    vol = context_dict["Volatility_and_Sentiment"]
    
    short_pred = ""
    if "多头排列" in trends['Daily_Trend']:
        if "资金暗中吸筹" in moms['Volume_and_MACD'] and "买盘挂单极强" in vol['Order_Book_Imbalance']:
            short_pred = "日线多头排列且 OBV 显示资金吸筹，叠加实时盘口买盘强劲，短线具备极强向上突破动能，有望加速冲击上方阻力位。"
        elif "缩量" in moms['Volume_and_MACD'] or "抛压如山" in vol['Order_Book_Imbalance']:
            short_pred = "日线虽维持多头排列，但盘口抛压较重或量能不足，提防主力诱多。短线或在当前区间剧烈震荡，日均波动幅度预估为 " + str(vol['Daily_ATR_Volatility']) + " USDT。"
        else:
            short_pred = "日线呈多头结构运行，各项指标平稳，短线有望依托 EMA20 支撑稳步向上震荡。"
    else:
        if "存在超卖反弹预期" in moms['Bollinger_Position'] and "资金暗中吸筹" in moms['Volume_and_MACD']:
            short_pred = "价格极度贴近下轨且引发超卖，但 OBV 呈现底背离（资金未跟随价格流出），短线随时可能触发报复性反弹，重点观察盘口买单承接力。"
        else:
            short_pred = "日线空头压制明显，且大资金未见回补迹象。在当前 " + str(vol['Daily_ATR_Volatility']) + " USDT 的日均波动率下，短线仍有向下寻底风险，切忌盲目抄底。"

    long_pred = ""
    if "宏观熊市" in trends['Weekly_Trend']:
        long_pred = "周线跌破牛熊分水岭，宏观格局维持熊市压制。在全网情绪处于 " + vol['Fear_Greed_Index'] + " 的背景下，大周期资金观望情绪浓厚。若无极强宏观利好刺激放量，周线级别将继续寻底。"
    else:
        long_pred = "周线稳居牛熊分水岭之上，长线牛市基座稳固。大周期多头结构未被破坏，建议在日线级别的每一次深度回调中（靠近强支撑位）寻找介入机会。"

    return short_pred, long_pred

def format_fast_snapshot(context_dict):
    cp = context_dict["Current_Price"]
    dt = context_dict["Data_Timestamp_JST"]
    src = context_dict["Price_Source"]
    trends = context_dict["Explicit_Trend_Labels"]
    moms = context_dict["Explicit_Momentum_Labels"]
    vol = context_dict["Volatility_and_Sentiment"]
    sr = context_dict["Explicit_Support_and_Resistance"]
    
    short_pred, long_pred = generate_algo_prediction(context_dict)
    
    html = f"⚡ <b>毫秒级极速快照 (融合 OBV/ATR/盘口)</b>\n\n"
    html += f"<b>🕒 行情快照 (JST)</b>\n"
    html += f"• 最新报价： {cp} USDT\n"
    html += f"• 抓取时间： {dt}\n"
    html += f"• 行情来源： {src}\n\n"
    html += f"<b>📊 行情主基调</b>\n"
    html += f"• 日线级别： {trends['Daily_Trend']}\n"
    html += f"• 周线级别： {trends['Weekly_Trend']}\n\n"
    html += f"<b>📈 动能、筹码与情绪</b>\n"
    html += f"• 布林带位置： {moms['Bollinger_Position']}\n"
    html += f"• RSI指标： 日线 {moms['Daily_RSI']} | 周线 {moms['Weekly_RSI']}\n"
    html += f"• 量价配合： {moms['Volume_and_MACD']}\n"
    html += f"• 实时挂单(深度)： {vol['Order_Book_Imbalance']}\n"
    html += f"• 波动率(ATR)： 日均波动 {vol['Daily_ATR_Volatility']} USDT\n"
    html += f"• 宏观情绪： F&G {vol['Fear_Greed_Index']} | 费率 {vol['Funding_Rate']}\n\n"
    html += f"<b>🎯 核心兵家必争之地</b>\n"
    html += f"• 下方支撑：\n"
    for s in sr['Below_Price_Supports'][:2]: html += f"  • {s}\n"
    html += f"• 上方阻力：\n"
    for r in sr['Above_Price_Resistances'][:2]: html += f"  • {r}\n\n"
    html += f"<b>🔮 周期级推演预判 (算法自动生成)</b>\n"
    html += f"• <b>未来 1-3 天 (短线博弈)：</b>\n  {short_pred}\n\n"
    html += f"• <b>未来 1-4 周 (宏观趋势)：</b>\n  {long_pred}"
    return html

def analyze_with_local_llm(context_dict):
    data_json = json.dumps(context_dict, indent=2, ensure_ascii=False)
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    
    system_prompt = """
    你是一个负责排版和语言润色的顶级量化金融播报员。
    
    【最高指令：绝对服从 Python】
    所有的趋势定性、OBV筹码、盘口深度、ATR波动率等，均已在数据中为你算好。
    用流畅、充满机构交易员专业语气的中文，将这些结论组合成清晰的报告。绝对照抄预设结论，禁止自行修改。
    
    【排版要求】
    1. 必须遵守 Telegram HTML 限制，仅使用 <b> 加粗。
    2. 列表项使用 "• "。直接换行。
    3. 绝对不使用 Markdown 代码块 (```)。
    
    【内容结构】
    <b>🕒 实时行情快照 (JST)</b>
    • 最新报价与数据来源、时间。

    <b>📊 行情主基调</b>
    • 日线与周线结论。

    <b>📈 动能、筹码与情绪</b>
    • 必须包含：布林带位置、日周线RSI、量价与OBV状态、实时盘口深度 (Order Book)、ATR波动幅度、以及当前的宏观情绪 (贪婪指数与费率)。

    <b>🎯 核心兵家必争之地</b>
    • 提取离现价最近的前两项支撑和阻力。

    <b>🔮 周期级趋势预测与演推</b>
    • 结合上述盘口、OBV吸筹状态、波动率及均线，分别写出 1-3 天和 1-4 周的前瞻推演。
    """
    try:
        response = client.chat.completions.create(
            model="mistralai/ministral-3-14b-reasoning", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"这是核心量化数据：\n{data_json}\n请生成 HTML 纯净报告。"}
            ],
            temperature=0.1,
            max_tokens=4096
        )
        
        content = response.choices[0].message.content.strip()
        content = re.sub(r"^```[a-zA-Z]*\n?", "", content) 
        content = re.sub(r"```$", "", content).strip()       
        return content
    except Exception as e:
        return f"<b>❌ 连接失败</b>\n连接 LM Studio 失败。错误信息: {e}"

# ================= Telegram 交互与主循环 =================

def get_standard_keyboard():
    # 恢复五宫格全键盘
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🚀 AI 深度分析", callback_data='run_llm'),
         InlineKeyboardButton("⚡ 极速快照", callback_data='run_fast')],
        [InlineKeyboardButton("📈 资金费率", callback_data='funding'),
         InlineKeyboardButton("📊 24h概况", callback_data='summary')],
        [InlineKeyboardButton("🧭 恐慌贪婪指数", callback_data='fng')]
    ])

async def generate_and_send_report(bot, chat_id, edit_message_id=None, use_llm=True):
    try:
        loop = asyncio.get_event_loop()
        market_data_dict = await loop.run_in_executor(None, get_market_data)
        
        if use_llm:
            report = await loop.run_in_executor(None, analyze_with_local_llm, market_data_dict)
        else:
            report = format_fast_snapshot(market_data_dict) 
            
        reply_markup = get_standard_keyboard()

        if edit_message_id:
            await bot.edit_message_text(chat_id=chat_id, message_id=edit_message_id, text=report, parse_mode="HTML", reply_markup=reply_markup)
        else:
            await bot.send_message(chat_id=chat_id, text=report, parse_mode="HTML", reply_markup=reply_markup)
    except Exception as e:
        await bot.send_message(chat_id=chat_id, text=f"❌ 运行出错: {e}")

# --- 指令处理器 ---
async def cmd_run_llm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_user.id != ALLOWED_USER_ID: return
    status_message = await update.message.reply_text("🔄 深度数据融合中，唤醒大模型...")
    await generate_and_send_report(context.bot, ALLOWED_USER_ID, status_message.message_id, use_llm=True)

async def cmd_run_fast(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_user.id != ALLOWED_USER_ID: return
    await generate_and_send_report(context.bot, ALLOWED_USER_ID, use_llm=False)

async def cmd_funding(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_user.id != ALLOWED_USER_ID: return
    await update.message.reply_text(get_live_funding_rate(), parse_mode="HTML", reply_markup=get_standard_keyboard())

async def cmd_summary(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_user.id != ALLOWED_USER_ID: return
    await update.message.reply_text(get_24h_summary(), parse_mode="HTML", reply_markup=get_standard_keyboard())

async def cmd_fng(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_user.id != ALLOWED_USER_ID: return
    await update.message.reply_text(get_fear_and_greed(), parse_mode="HTML", reply_markup=get_standard_keyboard())

# --- 内联按钮回调 ---
async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query.from_user.id != ALLOWED_USER_ID: return
    await query.answer()
    
    if query.data == 'run_llm':
        await query.edit_message_text("🔄 正在唤醒大模型，融合新数据...")
        await generate_and_send_report(context.bot, ALLOWED_USER_ID, query.message.message_id, use_llm=True)
    elif query.data == 'run_fast':
        await generate_and_send_report(context.bot, ALLOWED_USER_ID, query.message.message_id, use_llm=False)
    elif query.data == 'funding':
        await context.bot.send_message(chat_id=ALLOWED_USER_ID, text=get_live_funding_rate(), parse_mode="HTML", reply_markup=get_standard_keyboard())
    elif query.data == 'summary':
        await context.bot.send_message(chat_id=ALLOWED_USER_ID, text=get_24h_summary(), parse_mode="HTML", reply_markup=get_standard_keyboard())
    elif query.data == 'fng':
        await context.bot.send_message(chat_id=ALLOWED_USER_ID, text=get_fear_and_greed(), parse_mode="HTML", reply_markup=get_standard_keyboard())

# --- 定时监控与推送 ---
async def daily_digest(context: ContextTypes.DEFAULT_TYPE) -> None:
    await context.bot.send_message(chat_id=ALLOWED_USER_ID, text="🌅 <b>时区例行推演：定时全维简报生成中...</b>", parse_mode="HTML")
    await generate_and_send_report(context.bot, ALLOWED_USER_ID, use_llm=True)

async def price_monitor(context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, get_market_data)
        
        # 提取指标状态
        bb = data["Explicit_Momentum_Labels"]["Bollinger_Position"]
        rsi = data["Explicit_Momentum_Labels"]["Daily_RSI"]
        ob_imb = data["Volatility_and_Sentiment"]["Order_Book_Imbalance"]
        price = data["Current_Price"]
        src = data["Price_Source"]
        dt = data["Data_Timestamp_JST"]

        # 重新单独拉取一次精准盘口数据，提取具体挂单量用于展示
        # 这里使用极其安全的公开免密接口
        import ccxt
        exchange = ccxt.binance({'timeout': 2000})
        orderbook = exchange.fetch_order_book('BTC/USDT', limit=50)
        bids_vol = sum(bid[1] for bid in orderbook['bids']) # 累计买单
        asks_vol = sum(ask[1] for ask in orderbook['asks']) # 累计卖单
        ratio = bids_vol / asks_vol if asks_vol > 0 else 1.0

        alert = ""
        # 1. 空间异动判定
        if "极度" in bb: 
            alert += f"▪️ <b>空间异动：</b>价格 {bb}\n"
        # 2. 动能异动判定
        if "极度" in rsi: 
            alert += f"▪️ <b>动能异动：</b>日线 RSI 处于 {rsi} 状态\n"
        # 3. 盘口异动判定
        if "极强" in ob_imb or "沉重" in ob_imb: 
            alert += f"▪️ <b>盘口异动：</b>{ob_imb}\n"
            alert += f"  - 50档累计买单: <code>{bids_vol:.2f} BTC</code>\n"
            alert += f"  - 50档累计卖单: <code>{asks_vol:.2f} BTC</code>\n"
            alert += f"  - 实测买卖比例: <code>{ratio:.2f}</code>\n"
        
        # 如果触发了任何一项异动，则组装满血版警报下发
        if alert:
            msg = (
                f"🚨 <b>【量化盯盘警报】</b> 🚨\n\n"
                f"<b>🕒 监控基本面</b>\n"
                f"• 最新报价：<b>{price} USDT</b>\n"
                f"• 数据来源：{src}\n"
                f"• 抓取时间：{dt}\n\n"
                f"<b>🔍 异动指标详情</b>\n"
                f"{alert}"
            )
            await context.bot.send_message(chat_id=ALLOWED_USER_ID, text=msg, parse_mode="HTML")
    except Exception as e:
        # 静默盯盘，报错不中断主进程
        print(f"⚠️ 盯盘监控运行中遭遇偶发异常: {e}")

# ================= 永续运行与网络守护 =================

def wait_for_internet():
    """持续检查网络，直到 ping 通公共 DNS"""
    print("🌐 等待网络连接...")
    while True:
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            print("✅ 网络已就绪。")
            return
        except OSError:
            time.sleep(30)

def main():
    print("🤖 量化外挂：快慢双规引擎启动。")
    while True:
        try:
            # 1. 确保网络就绪
            wait_for_internet()
            
            # 2. 初始化 App
            application = Application.builder().token(BOT_TOKEN).build()

            # 注册命令
            async def setup_bot(app):
                await app.bot.set_my_commands([
                    ("fast", "⚡ 毫秒级极速数据快照"),
                    ("run", "🚀 AI 深度综合技术分析"),
                    ("funding", "📈 实时资金费率"),
                    ("summary", "📊 24小时盘面概况"),
                    ("fng", "🧭 恐慌贪婪指数")
                ])
            
            loop = asyncio.get_event_loop()
            loop.run_until_complete(setup_bot(application))

            # 设置定时任务
            job_queue = application.job_queue
            job_queue.run_daily(daily_digest, time=dt_time(hour=8, minute=0, tzinfo=TOKYO_TZ), chat_id=ALLOWED_USER_ID)
            job_queue.run_daily(daily_digest, time=dt_time(hour=19, minute=0, tzinfo=TOKYO_TZ), chat_id=ALLOWED_USER_ID)
            job_queue.run_repeating(price_monitor, interval=900, first=10, chat_id=ALLOWED_USER_ID)

            # 绑定处理器
            application.add_handler(CommandHandler("run", cmd_run_llm))
            application.add_handler(CommandHandler("fast", cmd_run_fast))
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
