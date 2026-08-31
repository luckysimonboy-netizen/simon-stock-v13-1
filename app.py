import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================
# SIMON STOCK V14.0
# AI-NATIVE US STOCK RESEARCH TERMINAL
# ============================================================

st.set_page_config(
    page_title="Simon Stock",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Design system
# -----------------------------
st.markdown("""
<style>
:root {
    --bg:#f4f6fa;
    --card:rgba(255,255,255,.78);
    --text:#10131a;
    --muted:#687181;
    --line:rgba(20,30,50,.08);
    --blue:#1677ff;
    --green:#16834b;
    --red:#d94a4a;
    --orange:#b77900;
}
@media (prefers-color-scheme: dark) {
    :root {
        --bg:#080a0e;
        --card:rgba(25,29,36,.78);
        --text:#f4f6fa;
        --muted:#9aa3b2;
        --line:rgba(255,255,255,.08);
        --blue:#4b9bff;
        --green:#36c985;
        --red:#ff6868;
        --orange:#e3ae43;
    }
}
.stApp {
    background:
      radial-gradient(circle at 15% 0%, rgba(22,119,255,.08), transparent 28%),
      radial-gradient(circle at 90% 15%, rgba(80,160,255,.06), transparent 25%),
      var(--bg);
    color:var(--text);
}
.block-container {
    max-width:1500px;
    padding-top:1.2rem;
    padding-bottom:4rem;
}
.ss-card {
    background:var(--card);
    border:1px solid var(--line);
    border-radius:22px;
    padding:20px;
    box-shadow:0 8px 30px rgba(20,30,50,.06);
    backdrop-filter:blur(22px) saturate(135%);
    -webkit-backdrop-filter:blur(22px) saturate(135%);
    margin-bottom:16px;
}
.ss-hero {
    padding:28px;
    border-radius:28px;
    background:linear-gradient(135deg,rgba(255,255,255,.78),rgba(255,255,255,.48));
    border:1px solid var(--line);
    box-shadow:0 12px 40px rgba(20,30,50,.08);
}
@media (prefers-color-scheme:dark) {
 .ss-hero {background:linear-gradient(135deg,rgba(35,40,50,.8),rgba(22,25,31,.68));}
}
.ss-title {font-size:2rem;font-weight:800;letter-spacing:-.04em}
.ss-sub {color:var(--muted);font-size:.92rem}
.ss-price {font-size:2.4rem;font-weight:800;letter-spacing:-.04em}
.ss-score {font-size:3rem;font-weight:850;letter-spacing:-.05em}
.ss-pill {
 display:inline-block;padding:6px 12px;border-radius:999px;
 background:rgba(22,119,255,.10);color:var(--blue);font-weight:750;
}
.small {font-size:.82rem;color:var(--muted)}
div[data-testid="stMetric"] {
 background:var(--card);border:1px solid var(--line);
 border-radius:18px;padding:14px;box-shadow:0 5px 20px rgba(20,30,50,.04);
}
.stButton>button {border-radius:14px;min-height:42px;font-weight:650;}
input, textarea {border-radius:14px!important;}
section[data-testid="stSidebar"] {
 background:rgba(248,249,251,.72);
 backdrop-filter:blur(22px);
}
@media (prefers-color-scheme:dark) {
 section[data-testid="stSidebar"] {background:rgba(13,15,20,.82);}
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Utilities
# -----------------------------
def safe(v, default=np.nan):
    try:
        if v is None or pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default

def money(v):
    return "—" if pd.isna(v) else f"${v:,.2f}"

def pct(v):
    return "—" if pd.isna(v) else f"{v*100:+.2f}%"

def clamp(x,a=0,b=100):
    return max(a,min(b,float(x)))

@st.cache_data(ttl=300, show_spinner=False)
def load_market(ticker, period):
    return yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

@st.cache_data(ttl=900, show_spinner=False)
def load_info(ticker):
    try:
        return yf.Ticker(ticker).info
    except Exception:
        return {}

def normalize(df):
    if df is None or df.empty:
        return pd.DataFrame()
    x=df.copy()
    if isinstance(x.columns,pd.MultiIndex):
        x.columns=x.columns.get_level_values(0)
    x.columns=[str(c).title() for c in x.columns]
    if "Adj Close" not in x and "Close" in x:
        x["Adj Close"]=x["Close"]
    for c in ["Open","High","Low","Close","Adj Close","Volume"]:
        if c in x:
            x[c]=pd.to_numeric(x[c],errors="coerce")
    return x.dropna(subset=["Close"])

def indicators(df):
    x=normalize(df)
    if x.empty:return x
    c=x["Close"]
    x["SMA20"]=c.rolling(20).mean()
    x["SMA50"]=c.rolling(50).mean()
    x["SMA200"]=c.rolling(200).mean()
    x["EMA12"]=c.ewm(span=12,adjust=False).mean()
    x["EMA26"]=c.ewm(span=26,adjust=False).mean()
    x["MACD"]=x["EMA12"]-x["EMA26"]
    x["MACDSignal"]=x["MACD"].ewm(span=9,adjust=False).mean()
    d=c.diff()
    gain=d.clip(lower=0).rolling(14).mean()
    loss=(-d.clip(upper=0)).rolling(14).mean()
    rs=gain/loss.replace(0,np.nan)
    x["RSI14"]=100-(100/(1+rs))
    x["Return1D"]=c.pct_change()
    x["Return20D"]=c.pct_change(20)
    x["Return60D"]=c.pct_change(60)
    x["Vol20"]=x["Return1D"].rolling(20).std()*np.sqrt(252)
    x["VolumeSMA20"]=x["Volume"].rolling(20).mean()
    x["RelVolume"]=x["Volume"]/x["VolumeSMA20"].replace(0,np.nan)
    x["High52"]=c.rolling(252).max()
    x["Low52"]=c.rolling(252).min()
    return x

def score_stock(df, info):
    x=indicators(df)
    if x.empty:return None
    r=x.iloc[-1]
    price=safe(r["Close"],0)

    trend=50
    for a,b in [("SMA20",8),("SMA50",10),("SMA200",12)]:
        if safe(r[a],price)<price: trend+=b
        elif safe(r[a],price)>price: trend-=b*.7
    if safe(r["SMA20"],0)>safe(r["SMA50"],0):trend+=8
    if safe(r["SMA50"],0)>safe(r["SMA200"],0):trend+=8
    trend=clamp(trend)

    mom=50
    rsi=safe(r["RSI14"],50)
    if 50<=rsi<=70:mom+=12
    elif rsi>75:mom-=8
    elif rsi<30:mom+=4
    mom += 10 if safe(r["MACD"])>safe(r["MACDSignal"]) else -8
    mom += 8 if safe(r["Return20D"])>0 else -6
    mom += 8 if safe(r["Return60D"])>0 else -6
    mom=clamp(mom)

    pe=safe(info.get("trailingPE"))
    fpe=safe(info.get("forwardPE"))
    growth=safe(info.get("earningsGrowth"))
    margin=safe(info.get("profitMargins"))
    roe=safe(info.get("returnOnEquity"))
    revg=safe(info.get("revenueGrowth"))

    valuation=50
    if not pd.isna(pe):
        if pe<15:valuation+=15
        elif pe<25:valuation+=8
        elif pe>45:valuation-=15
        elif pe>30:valuation-=7
    if not pd.isna(fpe) and not pd.isna(pe) and fpe<pe:valuation+=5
    if growth>0.20:valuation+=10
    valuation=clamp(valuation)

    quality=50
    if roe>.20:quality+=15
    elif roe>.10:quality+=8
    if margin>.20:quality+=15
    elif margin>.10:quality+=8
    if revg>.15:quality+=10
    quality=clamp(quality)

    ret=x["Return1D"].dropna()
    vol=safe(ret.std()*np.sqrt(252),.30)
    dd=(x["Close"]/x["Close"].cummax()-1).min()
    risk=100
    if vol>.60:risk-=35
    elif vol>.45:risk-=25
    elif vol>.30:risk-=15
    elif vol>.20:risk-=8
    mdd=abs(safe(dd,0))
    if mdd>.60:risk-=35
    elif mdd>.40:risk-=25
    elif mdd>.25:risk-=15
    elif mdd>.15:risk-=8
    risk=clamp(risk)

    composite=trend*.20+mom*.15+quality*.25+valuation*.25+risk*.15
    if risk<25:signal="HIGH RISK"
    elif composite>=80 and trend>=65:signal="STRONG BUY"
    elif composite>=68:signal="BUY"
    elif composite>=55:signal="WATCH"
    elif composite>=42:signal="HOLD"
    elif composite>=30:signal="REDUCE"
    else:signal="SELL"

    confidence=clamp(100-np.std([composite,trend,mom,risk])*1.5,20,95)
    risk_label="LOW" if risk>=75 else "MODERATE" if risk>=55 else "HIGH" if risk>=35 else "VERY HIGH"

    return dict(
        data=x,price=price,trend=trend,momentum=mom,quality=quality,
        valuation=valuation,risk=risk,composite=composite,
        signal=signal,confidence=confidence,risk_label=risk_label,
        pe=pe,fpe=fpe,growth=growth,margin=margin,roe=roe,revg=revg,
        name=info.get("longName") or info.get("shortName") or "",
        sector=info.get("sector","—"),industry=info.get("industry","—"),
        marketcap=safe(info.get("marketCap")),beta=safe(info.get("beta")),
        dividend=safe(info.get("dividendYield")),
        target=safe(info.get("targetMeanPrice")),
        cash=safe(info.get("totalCash")),debt=safe(info.get("totalDebt")),
    )

# -----------------------------
# AI-ready local research engine
# -----------------------------
def ai_template(result, ticker, question=""):
    s=result
    return f"""
Simon Stock AI Research Brief — {ticker}

Company: {s['name']}
Sector: {s['sector']}
Price: {money(s['price'])}

Quantitative score:
- Composite: {s['composite']:.1f}/100
- Trend: {s['trend']:.1f}
- Momentum: {s['momentum']:.1f}
- Business quality: {s['quality']:.1f}
- Valuation: {s['valuation']:.1f}
- Risk: {s['risk']:.1f}
- Signal: {s['signal']}
- Confidence: {s['confidence']:.1f}%

Value lens:
Assess moat, free cash flow, capital allocation, balance sheet and margin of safety.
Business lens:
Assess business model, pricing power, switching costs, management and opportunity cost.
First-principles lens:
Assess underlying economics, technology, cost structure, market size and scalability.
Event lens:
Assess rates, regulation, tariffs, macro conditions, catalysts and sentiment.

Available fundamentals:
P/E={s['pe']}; Forward P/E={s['fpe']}; Earnings growth={s['growth']};
Margin={s['margin']}; ROE={s['roe']}; Revenue growth={s['revg']};
Beta={s['beta']}; Market cap={s['marketcap']}; Target consensus={s['target']}.

User question: {question or 'Give a balanced investment research view.'}

Important: distinguish facts from assumptions, state missing data, present bull/base/bear cases, and never guarantee returns.
"""

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("## 📈 Simon Stock")
    st.caption("V14 · AI-native US equity research")
    ticker=st.text_input("Ticker", value=st.session_state.get("ticker","AAPL")).upper().strip()
    st.session_state.ticker=ticker
    period=st.selectbox("History",["6mo","1y","2y","5y","10y"],index=1)
    if st.button("🔄 Refresh data",use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    page=st.radio("Workspace",[
        "Overview","AI Research","Fundamentals","Technical","Risk Lab","Backtest"
    ])
    st.divider()
    st.caption("V14 foundation")
    st.caption("Data: Yahoo Finance via yfinance")
    st.caption("Research output is informational, not financial advice.")

# -----------------------------
# Load
# -----------------------------
if not ticker:
    st.warning("Enter a US stock ticker.")
    st.stop()

with st.spinner(f"Loading {ticker}…"):
    raw=load_market(ticker,period)
    info=load_info(ticker)
result=score_stock(raw,info)

if result is None:
    st.error("No valid market data returned. Check the ticker and try again.")
    st.stop()

d=result["data"]
price=result["price"]
prev=d["Close"].iloc[-2] if len(d)>1 else np.nan
day=(price/prev-1) if prev else np.nan

# -----------------------------
# Header
# -----------------------------
st.markdown(f"""
<div class="ss-hero">
 <div class="ss-sub">SIMON STOCK · V14</div>
 <div class="ss-title">{ticker} <span class="ss-pill">{result['signal']}</span></div>
 <div class="ss-sub">{result['name']} · {result['sector']}</div>
 <div style="margin-top:12px" class="ss-price">{money(price)}</div>
 <div class="ss-sub">Today {pct(day)} · Updated {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
</div>
""",unsafe_allow_html=True)

# -----------------------------
# Score strip
# -----------------------------
cols=st.columns(5)
for col,label,key in zip(cols,["Simon Score","Trend","Momentum","Quality","Valuation"],
                         ["composite","trend","momentum","quality","valuation"]):
    with col:
        st.metric(label,f"{result[key]:.0f}/100")

# -----------------------------
# Pages
# -----------------------------
if page=="Overview":
    c1,c2=st.columns([1.35,1])
    with c1:
        st.markdown('<div class="ss-card"><b>Price & Trend</b>',unsafe_allow_html=True)
        chart=d[["Close","SMA20","SMA50","SMA200"]].dropna(how="all")
        st.line_chart(chart,use_container_width=True)
        st.markdown("</div>",unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="ss-card"><b>Simon Decision Engine</b>',unsafe_allow_html=True)
        st.markdown(f"### {result['signal']}")
        st.progress(int(result["composite"]))
        st.write(f"**Confidence:** {result['confidence']:.0f}%")
        st.write(f"**Risk:** {result['risk_label']} ({result['risk']:.0f}/100)")
        st.write(f"**20D:** {pct(safe(d['Return20D'].iloc[-1]))}")
        st.write(f"**60D:** {pct(safe(d['Return60D'].iloc[-1]))}")
        st.write(f"**RSI:** {safe(d['RSI14'].iloc[-1]):.1f}")
        st.markdown("</div>",unsafe_allow_html=True)

    a,b,c=st.columns(3)
    with a:
        st.markdown('<div class="ss-card"><b>Value Lens</b><p>Valuation score reflects available P/E, forward P/E and growth evidence. Missing data is not fabricated.</p></div>',unsafe_allow_html=True)
    with b:
        st.markdown('<div class="ss-card"><b>Business Lens</b><p>Quality score considers profitability, ROE and revenue growth as a foundation for deeper AI research.</p></div>',unsafe_allow_html=True)
    with c:
        st.markdown('<div class="ss-card"><b>First Principles</b><p>Use AI Research to challenge the business model, technology, economics and long-term growth ceiling.</p></div>',unsafe_allow_html=True)

elif page=="AI Research":
    st.subheader("🤖 Simon AI Research")
    st.caption("四大研究框架：Value · Business · First Principles · Event")
    q=st.text_area("Ask Simon",placeholder="例如：现在这个价格最大的风险是什么？什么情况值得加仓？")
    prompt=ai_template(result,ticker,q)
    st.markdown('<div class="ss-card">',unsafe_allow_html=True)
    st.markdown("### Research context")
    st.code(prompt,language="text")
    st.markdown("</div>",unsafe_allow_html=True)
    st.info("AI Provider 接口已预留。下一阶段接入 Gemini / OpenRouter 后，这里可以直接生成完整 Bull / Base / Bear 投资委员会报告。")

elif page=="Fundamentals":
    st.subheader("💰 Fundamentals")
    m1,m2,m3,m4=st.columns(4)
    m1.metric("P/E", "—" if pd.isna(result["pe"]) else f"{result['pe']:.1f}x")
    m2.metric("Forward P/E","—" if pd.isna(result["fpe"]) else f"{result['fpe']:.1f}x")
    m3.metric("ROE","—" if pd.isna(result["roe"]) else pct(result["roe"]))
    m4.metric("Revenue Growth","—" if pd.isna(result["revg"]) else pct(result["revg"]))
    st.markdown('<div class="ss-card"><b>Company profile</b>',unsafe_allow_html=True)
    st.write(f"**Industry:** {result['industry']}")
    st.write(f"**Market cap:** {money(result['marketcap']) if not pd.isna(result['marketcap']) else '—'}")
    st.write(f"**Profit margin:** {pct(result['margin']) if not pd.isna(result['margin']) else '—'}")
    st.write(f"**Earnings growth:** {pct(result['growth']) if not pd.isna(result['growth']) else '—'}")
    st.write(f"**Cash:** {money(result['cash']) if not pd.isna(result['cash']) else '—'}")
    st.write(f"**Debt:** {money(result['debt']) if not pd.isna(result['debt']) else '—'}")
    st.markdown("</div>",unsafe_allow_html=True)

elif page=="Technical":
    st.subheader("📊 Technical Lab")
    st.line_chart(d[["Close","SMA20","SMA50","SMA200"]].dropna(how="all"))
    x,y=st.columns(2)
    with x:
        st.markdown("### RSI 14")
        st.line_chart(d[["RSI14"]])
    with y:
        st.markdown("### MACD")
        st.line_chart(d[["MACD","MACDSignal"]])
    st.dataframe(d.tail(30)[["Close","SMA20","SMA50","SMA200","RSI14","MACD","RelVolume"]],use_container_width=True)

elif page=="Risk Lab":
    st.subheader("🛡️ Risk Lab")
    r1,r2,r3,r4=st.columns(4)
    r1.metric("Risk score",f"{result['risk']:.0f}/100")
    r2.metric("Risk level",result["risk_label"])
    r3.metric("Volatility",pct(safe(d["Return1D"].std()*np.sqrt(252))))
    r4.metric("Beta","—" if pd.isna(result["beta"]) else f"{result['beta']:.2f}")
    dd=d["Close"]/d["Close"].cummax()-1
    st.markdown("### Drawdown")
    st.line_chart(dd.rename("Drawdown"))
    st.markdown("### Risk interpretation")
    st.write("Higher Simon Risk Score means lower observed historical risk. Historical volatility and drawdown do not predict future losses.")

elif page=="Backtest":
    st.subheader("🧪 Simple Strategy Backtest")
    st.caption("基础示例：SMA20 上穿 SMA50 持有，否则空仓。仅用于研究，不代表可交易结果。")
    bt=d[["Close","SMA20","SMA50"]].dropna().copy()
    bt["Signal"]=(bt["SMA20"]>bt["SMA50"]).astype(int)
    bt["MarketReturn"]=bt["Close"].pct_change().fillna(0)
    bt["StrategyReturn"]=bt["MarketReturn"]*bt["Signal"].shift(1).fillna(0)
    equity=(1+bt["StrategyReturn"]).cumprod()
    market=(1+bt["MarketReturn"]).cumprod()
    q1,q2,q3=st.columns(3)
    total=equity.iloc[-1]-1
    market_total=market.iloc[-1]-1
    maxdd=(equity/equity.cummax()-1).min()
    q1.metric("Strategy return",pct(total))
    q2.metric("Buy & hold",pct(market_total))
    q3.metric("Max drawdown",pct(maxdd))
    st.line_chart(pd.DataFrame({"Strategy":equity,"Buy & Hold":market}))
    st.dataframe(bt.tail(50),use_container_width=True)

st.divider()
st.caption("Simon Stock V14.0 · AI-native research terminal · Always verify data and conduct your own due diligence.")