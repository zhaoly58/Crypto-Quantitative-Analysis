import yfinance as yf
import pandas as pd
import json
import asyncio
import ccxt
import requests
import re
from datetime import time
from zoneinfo import ZoneInfo
from openai import OpenAI
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

import ta
from ta.trend import EMAIndicator, MACD, SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

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
    
    # 宏观与阶段性区分
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

def momentum_status(macd_hist, current_vol, avg_vol):
    macd_label = "正值 (多头动能发散)" if macd_hist > 0 else "负值 (空头动能主导)"
    vol_label = "放量" if current_vol > avg_vol else "缩量"
    return f"MACD {macd_label}，成交量 {vol_label}"

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

# --- 扩展数据接口模块 ---

def get_live_funding_rate():
    try:
        exchange = ccxt.binance({'timeout': 3000})
        funding = exchange.fetch_funding_rate('BTC/USDT:USDT')
        rate_pct = funding['fundingRate'] * 100
        status = "多头极其强势 (多付空)" if rate_pct > 0.01 else "空头强势 (空付多)" if rate_pct < 0 else "市场情绪中性"
        return f"📈 <b>币安合约实时情绪</b>\n\n• 当前资金费率: <b>{rate_pct:.4f}%</b>\n• 情绪判断: {status}"
    except Exception as e:
        return f"❌ 获取资金费率失败: {e}"

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

def get_fear_and_greed():
    try:
        resp = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
        data = resp.json()
        val = data['data'][0]['value']
        classification = data['data'][0]['value_classification']
        return f"🧭 <b>全网恐慌与贪婪指数</b>\n\n• 当前指数: <b>{val}</b> (0极度恐慌 - 100极度贪婪)\n• 市场状态: <b>{classification}</b>"
    except Exception as e:
        return f"❌ 获取恐慌贪婪指数失败: {e}"


def get_market_data():
    btc = yf.Ticker("BTC-USD")
    
    # 日线历史底座
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
    df_daily.dropna(inplace=True)
    
    # 50%单日振幅熔断保护
    pct_change = df_daily['Close'].pct_change().abs().iloc[-1]
    if pct_change > 0.50:
        raise Exception("检测到单日振幅>50%，Yahoo数据源可能遭遇插针污染，拒绝分析。")

    latest_d = df_daily.iloc[-1]

    # 周线历史底座
    df_weekly = btc.history(period="5y", interval="1wk")
    df_weekly.index = df_weekly.index.tz_localize(None)
    df_weekly['EMA_20'] = EMAIndicator(close=df_weekly['Close'], window=20).ema_indicator()
    df_weekly['EMA_50'] = EMAIndicator(close=df_weekly['Close'], window=50).ema_indicator()
    df_weekly['EMA_200'] = EMAIndicator(close=df_weekly['Close'], window=200).ema_indicator()
    df_weekly['RSI_14'] = RSIIndicator(close=df_weekly['Close'], window=14).rsi()
    df_weekly['MACD_Histogram'] = MACD(close=df_weekly['Close'], window_fast=12, window_slow=26, window_sign=9).macd_diff()
    df_weekly.dropna(inplace=True)
    latest_w = df_weekly.iloc[-1]

    # 从交易所获取精准实时一手的价格与时间戳（东京时区对齐）
    try:
        exchange = ccxt.binance({'timeout': 3000})
        ticker = exchange.fetch_ticker('BTC/USDT')
        current_price = ticker['last']
        data_time = pd.to_datetime(ticker['timestamp'], unit='ms').tz_localize('UTC').tz_convert('Asia/Tokyo').strftime('%Y-%m-%d %H:%M:%S')
        price_source = "Binance (实时)"
    except Exception as e:
        print(f"⚠️ 币安连接失败，降级使用雅虎数据: {e}")
        current_price = round(latest_d['Close'], 2)
        data_time = pd.Timestamp.now(tz='Asia/Tokyo').strftime('%Y-%m-%d %H:%M:%S')
        price_source = "Yahoo (延迟)"

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
            "Volume_and_MACD": momentum_status(latest_d['MACD_Histogram'], latest_d['Volume'], latest_d['Volume_MA20'])
        },
        "Explicit_Support_and_Resistance": {
            "Below_Price_Supports": supports,
            "Above_Price_Resistances": resistances
        }
    }
    return context

# --- 新增：纯 Python 毫秒级格式化函数 ---
def format_fast_snapshot(context_dict):
    cp = context_dict["Current_Price"]
    dt = context_dict["Data_Timestamp_JST"]
    src = context_dict["Price_Source"]
    trends = context_dict["Explicit_Trend_Labels"]
    moms = context_dict["Explicit_Momentum_Labels"]
    sr = context_dict["Explicit_Support_and_Resistance"]
    
    html = f"⚡ <b>毫秒级极速快照 (纯数据无延迟)</b>\n\n"
    html += f"<b>🕒 实时行情快照 (JST)</b>\n"
    html += f"• <b>最新报价：</b> {cp} USDT\n"
    html += f"• <b>抓取时间：</b> {dt}\n"
    html += f"• <b>行情来源：</b> {src}\n\n"
    
    html += f"<b>📊 行情主基调</b>\n"
    html += f"• <b>日线：</b> {trends['Daily_Trend']}\n"
    html += f"• <b>周线：</b> {trends['Weekly_Trend']}\n\n"
    
    html += f"<b>📈 动能与空间定量</b>\n"
    html += f"• <b>布林带：</b> {moms['Bollinger_Position']}\n"
    html += f"• <b>日线RSI：</b> {moms['Daily_RSI']} | <b>周线RSI：</b> {moms['Weekly_RSI']}\n"
    html += f"• <b>量能：</b> {moms['Volume_and_MACD']}\n\n"
    
    html += f"<b>🎯 核心兵家必争之地</b>\n"
    html += f"• <b>下方支撑：</b>\n"
    for s in sr['Below_Price_Supports'][:2]: html += f"  • {s}\n"
    html += f"• <b>上方阻力：</b>\n"
    for r in sr['Above_Price_Resistances'][:2]: html += f"  • {r}\n"
        
    return html

def analyze_with_local_llm(context_dict):
    data_json = json.dumps(context_dict, indent=2, ensure_ascii=False)
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    
    system_prompt = """
    你是一个负责排版和语言润色的金融播报员。
    
    【最高指令：绝对服从 Python】
    所有的金融趋势定性、动能强弱、支撑阻力位，全部已经在数据节点中为你计算并写死了！
    你不需要进行任何自主的方向判断。用流畅专业的中文，将这些给定的结论组合成一份清晰的报告。绝对照抄预设结论。
    
    【强制排版要求】
    1. 必须遵守 Telegram HTML 限制，仅使用 <b> 加粗。
    2. 列表项使用 "• "。直接换行，不用 <p> 或 <br>。
    3. 绝对不要使用任何 Markdown 代码块（如 ```html 或 ```），必须直接输出纯文本内容！！！
    
    【内容结构】
    <b>🕒 实时行情快照 (JST)</b>
    • <b>最新报价：</b> Current_Price USDT
    • <b>抓取时间：</b> Data_Timestamp_JST (东京时间)
    • <b>行情来源：</b> Price_Source

    <b>📊 行情主基调</b>
    (严格复述 Explicit_Trend_Labels)

    <b>📈 动能与空间定量</b>
    (严格复述 Explicit_Momentum_Labels)

    <b>🎯 核心兵家必争之地</b>
    • <b>下方支撑：</b>(提取 Below_Price_Supports 前两项)
    • <b>上方阻力：</b>(提取 Above_Price_Resistances 前两项)

    <b>🔮 周期级趋势预测与演推</b>
    • <b>未来 1-3 天走势预测（基于日线级别）：</b>(结合日线动能与支撑阻力，推演短线)
    • <b>未来 1-4 周趋势前瞻（基于周线大势）：</b>(结合周线长周期与空间形态，推演中长线)
    """
    try:
        response = client.chat.completions.create(
            model="mistralai/ministral-3-14b-reasoning", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"这是 Python 计算完毕的结论：\n{data_json}\n请排版输出符合 Telegram 规范的绝对纯净 HTML。"}
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

# 生成报告并绑定按钮（LLM 或 Fast 模式通用底层）
def get_standard_keyboard():
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
            report = format_fast_snapshot(market_data_dict) # 绕过大模型，纯本地渲染
            
        reply_markup = get_standard_keyboard()

        if edit_message_id:
            await bot.edit_message_text(chat_id=chat_id, message_id=edit_message_id, text=report, parse_mode="HTML", reply_markup=reply_markup)
        else:
            await bot.send_message(chat_id=chat_id, text=report, parse_mode="HTML", reply_markup=reply_markup)
    except Exception as e:
        await bot.send_message(chat_id=chat_id, text=f"❌ 运行中出错: {e}")

# --- 各路指令处理器 ---

async def cmd_run_llm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_user.id != ALLOWED_USER_ID: return
    status_message = await update.message.reply_text("🔄 正在生成双周期专业研报...")
    await generate_and_send_report(context.bot, ALLOWED_USER_ID, status_message.message_id, use_llm=True)

async def cmd_run_fast(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_user.id != ALLOWED_USER_ID: return
    # 极速快照直接下发，不需要“正在生成”提示
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

# --- 底部按钮回调处理器 ---
async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query.from_user.id != ALLOWED_USER_ID: return
    await query.answer()
    
    if query.data == 'run_llm':
        await query.edit_message_text("🔄 正在唤醒本地大模型，重构剧本...")
        await generate_and_send_report(context.bot, ALLOWED_USER_ID, query.message.message_id, use_llm=True)
    elif query.data == 'run_fast':
        await generate_and_send_report(context.bot, ALLOWED_USER_ID, query.message.message_id, use_llm=False)
    elif query.data == 'funding':
        await context.bot.send_message(chat_id=ALLOWED_USER_ID, text=get_live_funding_rate(), parse_mode="HTML", reply_markup=get_standard_keyboard())
    elif query.data == 'summary':
        await context.bot.send_message(chat_id=ALLOWED_USER_ID, text=get_24h_summary(), parse_mode="HTML", reply_markup=get_standard_keyboard())
    elif query.data == 'fng':
        await context.bot.send_message(chat_id=ALLOWED_USER_ID, text=get_fear_and_greed(), parse_mode="HTML", reply_markup=get_standard_keyboard())

# --- 定时与监控任务 ---
async def daily_digest(context: ContextTypes.DEFAULT_TYPE) -> None:
    await context.bot.send_message(chat_id=ALLOWED_USER_ID, text="🌅 <b>时区例行推演：定时简报生成中...</b>", parse_mode="HTML")
    await generate_and_send_report(context.bot, ALLOWED_USER_ID, use_llm=True)

async def price_monitor(context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        loop = asyncio.get_event_loop()
        market_data_dict = await loop.run_in_executor(None, get_market_data)
        
        bb_pos = market_data_dict["Explicit_Momentum_Labels"]["Bollinger_Position"]
        rsi_status = market_data_dict["Explicit_Momentum_Labels"]["Daily_RSI"]
        price = market_data_dict["Current_Price"]

        alert_msg = ""
        if "极度" in bb_pos: alert_msg += f"⚠️ <b>空间异动：</b>价格 ({price}) {bb_pos}！\n"
        if "极度" in rsi_status: alert_msg += f"⚠️ <b>动能异动：</b>日线 RSI 处于 {rsi_status} 状态！\n"
        
        if alert_msg:
            await context.bot.send_message(chat_id=ALLOWED_USER_ID, text=f"🚨 <b>【静默盯盘警报】</b> 🚨\n{alert_msg}", parse_mode="HTML")
    except Exception:
        pass

def main():
    print("🤖 终极量化外挂：快慢双规引擎启动。早8点、晚7点例行深度研报。")
    application = Application.builder().token(BOT_TOKEN).build()

    # 注册菜单命令
    async def setup_bot(app):
        commands = [
            ("run", "🚀 AI 深度综合技术分析"),
            ("fast", "⚡ 毫秒级极速数据快照 (无延迟)"),
            ("funding", "📈 实时资金费率"),
            ("summary", "📊 24小时盘面概况"),
            ("fng", "🧭 全网恐慌与贪婪指数")
        ]
        await app.bot.set_my_commands(commands)
        
    loop = asyncio.get_event_loop()
    loop.run_until_complete(setup_bot(application))

    # 调整为东京时间 早上 8:00 和 晚上 19:00 (7:00 PM)
    job_queue = application.job_queue
    job_queue.run_daily(daily_digest, time=time(hour=8, minute=0, tzinfo=TOKYO_TZ), chat_id=ALLOWED_USER_ID)
    job_queue.run_daily(daily_digest, time=time(hour=19, minute=0, tzinfo=TOKYO_TZ), chat_id=ALLOWED_USER_ID)
    job_queue.run_repeating(price_monitor, interval=900, first=10, chat_id=ALLOWED_USER_ID)

    application.add_handler(CommandHandler("run", cmd_run_llm))
    application.add_handler(CommandHandler("fast", cmd_run_fast))
    application.add_handler(CommandHandler("funding", cmd_funding))
    application.add_handler(CommandHandler("summary", cmd_summary))
    application.add_handler(CommandHandler("fng", cmd_fng))
    application.add_handler(CallbackQueryHandler(button_callback))
    
    application.run_polling()

if __name__ == "__main__":
    main()
