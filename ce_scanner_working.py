"""
Chandelier Exit Scanner — GitHub Actions version
Pine Script v6 exact port | Heikin Ashi 4H | ATR(1)×3 + ZLSMA(50)
Output: docs/ce_signals_<timestamp>.html (GitHub Pages)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import pytz
import os
import sys
from datetime import datetime, time, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

IST = pytz.timezone("Asia/Kolkata")

CE_LENGTH    = 1
CE_MULT      = 3.0
ZLSMA_LENGTH = 50
MAX_WORKERS  = 8
SIGNAL_DAYS  = 2

def load_symbols(csv_path="nse_list.csv"):
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found!")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    symbol_col = None
    for col in df.columns:
        if col.strip().lower() in ["symbol","ticker","scrip","nse symbol","nse_symbol"]:
            symbol_col = col
            break
    if symbol_col is None:
        symbol_col = df.columns[0]
    symbols = df[symbol_col].dropna().astype(str).str.strip().tolist()
    symbols = [s if s.endswith(".NS") else s + ".NS" for s in symbols]
    return list(dict.fromkeys(symbols))

def build_4h_bars(df_1h):
    if df_1h.index.tzinfo is None:
        df_1h.index = df_1h.index.tz_localize("UTC").tz_convert(IST)
    else:
        df_1h.index = df_1h.index.tz_convert(IST)
    records = []
    for date, day_df in df_1h.groupby(df_1h.index.date):
        b1 = day_df.between_time("09:15", "13:14")
        b2 = day_df.between_time("13:15", "15:29")
        if len(b1) >= 2:
            records.append({"datetime": pd.Timestamp(f"{date} 09:15", tz=IST),
                "Open": float(b1["Open"].iloc[0]), "High": float(b1["High"].max()),
                "Low":  float(b1["Low"].min()),    "Close": float(b1["Close"].iloc[-1])})
        if len(b2) >= 1:
            records.append({"datetime": pd.Timestamp(f"{date} 13:15", tz=IST),
                "Open": float(b2["Open"].iloc[0]), "High": float(b2["High"].max()),
                "Low":  float(b2["Low"].min()),    "Close": float(b2["Close"].iloc[-1])})
    if not records:
        return None
    return pd.DataFrame(records).set_index("datetime")

def calc_ha(df):
    n = len(df)
    ho, hh, hl, hc = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    o, h, l, c = df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values
    hc[0] = (o[0]+h[0]+l[0]+c[0])/4
    ho[0] = (o[0]+c[0])/2
    hh[0] = max(h[0], ho[0], hc[0])
    hl[0] = min(l[0], ho[0], hc[0])
    for i in range(1, n):
        hc[i] = (o[i]+h[i]+l[i]+c[i])/4
        ho[i] = (ho[i-1]+hc[i-1])/2
        hh[i] = max(h[i], ho[i], hc[i])
        hl[i] = min(l[i], ho[i], hc[i])
    return pd.DataFrame({"Open":ho,"High":hh,"Low":hl,"Close":hc}, index=df.index)

def wilder_atr(high, low, close, length):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr  = np.full(len(tr), np.nan)
    vals = tr.values
    idx  = int(np.argmax(~np.isnan(vals)))
    if idx + length <= len(vals):
        atr[idx+length-1] = np.mean(vals[idx:idx+length])
        for i in range(idx+length, len(vals)):
            atr[i] = (atr[i-1]*(length-1) + vals[i]) / length
    return pd.Series(atr, index=tr.index)

def calc_ce(ha, length=1, mult=3.0, use_close=True):
    close = ha["Close"]
    high  = ha["High"]
    low   = ha["Low"]
    atr   = mult * wilder_atr(high, low, close, length)
    highest = close.rolling(length).max() if use_close else high.rolling(length).max()
    lowest  = close.rolling(length).min() if use_close else low.rolling(length).min()
    ls = (highest - atr).values.copy()
    ss = (lowest  + atr).values.copy()
    c  = close.values
    for i in range(1, len(c)):
        lp = ls[i-1] if not np.isnan(ls[i-1]) else ls[i]
        sp = ss[i-1] if not np.isnan(ss[i-1]) else ss[i]
        if not np.isnan(ls[i]):
            ls[i] = max(ls[i], lp) if c[i-1] > lp else ls[i]
            ss[i] = min(ss[i], sp) if c[i-1] < sp else ss[i]
    direction = np.ones(len(c), dtype=int)
    for i in range(1, len(c)):
        lp = ls[i-1] if not np.isnan(ls[i-1]) else ls[i]
        sp = ss[i-1] if not np.isnan(ss[i-1]) else ss[i]
        if   c[i] > sp: direction[i] = 1
        elif c[i] < lp: direction[i] = -1
        else:           direction[i] = direction[i-1]
    dir_s   = pd.Series(direction, index=ha.index)
    buy_sig = (dir_s == 1) & (dir_s.shift(1) == -1)
    return buy_sig

def calc_zlsma(series, length=50):
    def linreg(s, n):
        result = np.full(len(s), np.nan)
        arr = s.values if hasattr(s, 'values') else np.array(s)
        for i in range(n-1, len(arr)):
            y = arr[i-n+1:i+1]
            if np.any(np.isnan(y)): continue
            x  = np.arange(n, dtype=float)
            xm, ym = x.mean(), y.mean()
            denom = np.sum((x-xm)**2)
            if denom == 0: continue
            slope     = np.sum((x-xm)*(y-ym)) / denom
            result[i] = (ym - slope*xm) + slope*(n-1)
        return pd.Series(result, index=s.index if hasattr(s,'index') else range(len(result)))
    lsma  = linreg(series, length)
    lsma2 = linreg(lsma,   length)
    return 2*lsma - lsma2

def bar_is_open():
    t = datetime.now(IST).time()
    return time(9, 15) <= t < time(15, 30)

def is_fresh(signal_ts, now_ist, max_days=2):
    sig_date = signal_ts.date()
    now_date = now_ist.date()
    days_back = 0
    check = now_date
    while check > sig_date and days_back <= max_days:
        check -= timedelta(days=1)
        if check.weekday() < 5:
            days_back += 1
    return days_back <= max_days and sig_date <= now_date

def scan(symbol):
    try:
        df_1h = yf.download(symbol, period="60d", interval="1h",
                            progress=False, auto_adjust=True)
        if df_1h is None or len(df_1h) < 30: return None
        if isinstance(df_1h.columns, pd.MultiIndex):
            df_1h.columns = df_1h.columns.get_level_values(0)
        df_1h = df_1h[["Open","High","Low","Close"]].dropna()
        df_4h = build_4h_bars(df_1h)
        if df_4h is None or len(df_4h) < ZLSMA_LENGTH + 5: return None
        if bar_is_open():
            df_4h = df_4h.iloc[:-1]
        now_ist = datetime.now(IST)
        ha      = calc_ha(df_4h)
        buy_sig = calc_ce(ha, length=CE_LENGTH, mult=CE_MULT)
        last_sig   = buy_sig.iloc[-1]
        second_sig = buy_sig.iloc[-2] if len(buy_sig) > 1 else False
        last_fresh   = last_sig   and is_fresh(buy_sig.index[-1], now_ist, SIGNAL_DAYS)
        second_fresh = second_sig and is_fresh(buy_sig.index[-2], now_ist, SIGNAL_DAYS)
        if not (last_fresh or second_fresh): return None
        if last_fresh:
            label    = "Last Bar"
            sig_time = buy_sig.index[-1]
        else:
            label    = "2nd Last Bar"
            sig_time = buy_sig.index[-2]
        zlsma_series = calc_zlsma(df_4h["Close"], length=ZLSMA_LENGTH)
        zlsma_val    = round(float(zlsma_series.iloc[-1]), 2) if not np.isnan(zlsma_series.iloc[-1]) else None
        price    = round(float(df_4h["Close"].iloc[-1]), 2)
        diff     = round(price - zlsma_val, 2)      if zlsma_val else None
        diff_pct = round(diff / zlsma_val * 100, 2) if zlsma_val else None
        return {
            "symbol": symbol.replace(".NS",""),
            "price":  price, "bar": label,
            "sig_ts": sig_time,
            "time":   sig_time.strftime("%d %b %H:%M"),
            "zlsma":  zlsma_val, "diff": diff, "diff_pct": diff_pct,
        }
    except Exception:
        pass
    return None

def make_html(results, scan_time, is_open, total):
    rows = ""
    for i, r in enumerate(results, 1):
        badge_color = "#00c853" if r["bar"] == "Last Bar" else "#ff9100"
        sym_clean   = r['symbol'].replace("&", "%26")
        tv_url      = f"https://www.tradingview.com/chart/?symbol=NSE%3A{sym_clean}&interval=240"
        if r["diff"] is not None:
            diff_color = "#3fb950" if r["diff"] > 0 else "#f85149"
            diff_arrow = "▲" if r["diff"] > 0 else "▼"
            zlsma_str  = f"₹{r['zlsma']:,.2f}"
            diff_str   = f'<span style="color:{diff_color};font-weight:600">{diff_arrow} ₹{abs(r["diff"]):,.2f} ({abs(r["diff_pct"])}%)</span>'
        else:
            zlsma_str = diff_str = "—"
        rows += f"""<tr>
            <td class="num">{i}</td>
            <td class="sym"><a href="{tv_url}" target="_blank" class="tv-link">{r['symbol']}<span class="tv-icon">↗</span></a></td>
            <td class="price">₹{r['price']:,.2f}</td>
            <td><span class="badge" style="background:{badge_color}">{r['bar']}</span></td>
            <td class="time">{r['time']}</td>
            <td class="zlsma">{zlsma_str}</td>
            <td class="diff">{diff_str}</td></tr>"""

    count = len(results)
    warn  = '<div class="note">⚠️ Market open — current bar excluded</div>' if is_open else ""
    empty = '<div class="empty"><div style="font-size:40px;margin-bottom:12px">—</div><p>No fresh CE buy signals found.</p></div>'
    table = f'<table><thead><tr><th>#</th><th>Symbol</th><th>Price</th><th>Signal</th><th>Bar Time (IST)</th><th>ZLSMA (50)</th><th>Price vs ZLSMA</th></tr></thead><tbody>{rows}</tbody></table>' if results else empty

    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="300">
<title>CE Scanner — {scan_time}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Space+Grotesk:wght@400;600;700&display=swap');
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#ffffff;color:#cdd9e5;font-family:'Space Grotesk',sans-serif;padding:32px 16px;min-height:100vh}}
.hdr{{max-width:1100px;margin:0 auto 28px;display:flex;justify-content:space-between;align-items:flex-end;border-bottom:1px solid #21262d;padding-bottom:20px;flex-wrap:wrap;gap:12px}}
.hdr h1{{font-size:28px;font-weight:700;color:#6e7681;letter-spacing:-0.5px}}
.hdr p{{font-size:13px;color:#ffffff;margin-top:5px;font-family:'JetBrains Mono',monospace}}
.meta{{text-align:right;font-family:'JetBrains Mono',monospace;font-size:12px;color:#ffffff}}
.cnt{{font-size:40px;font-weight:700;color:#3fb950;font-family:'Space Grotesk',sans-serif;line-height:1}}
.cnt.z{{color:#f85149}}
.scanned{{font-size:11px;color:#484f58;margin-top:3px}}
.note{{max-width:1100px;margin:0 auto 16px;background:#161b22;border:1px solid #e3b341;border-radius:8px;padding:11px 16px;font-size:13px;color:#e3b341;font-family:'JetBrains Mono',monospace}}
table{{width:100%;max-width:1100px;margin:0 auto;border-collapse:collapse;border-radius:8px;overflow:hidden}}
thead tr{{background:#161b22;border-bottom:2px solid #21262d}}
th{{padding:13px 16px;font-size:11px;letter-spacing:1px;text-transform:uppercase;color:#ffffff;text-align:left;font-family:'JetBrains Mono',monospace;white-space:nowrap}}
tbody tr{{border-bottom:1px solid #161b22;transition:background .12s}}
tbody tr:hover{{background:#161b22}}
td{{padding:14px 16px;font-size:14px;vertical-align:middle}}
td.num{{color:#484f58;font-family:'JetBrains Mono',monospace;font-size:12px;width:36px}}
td.sym{{font-weight:700;font-size:15px}}
td.price{{font-family:'JetBrains Mono',monospace;color:#79c0ff;font-size:14px;font-weight:600}}
td.time{{font-family:'JetBrains Mono',monospace;font-size:13px;color:#8b949e}}
td.zlsma{{font-family:'JetBrains Mono',monospace;color:#d2a8ff;font-size:13px}}
td.diff{{font-family:'JetBrains Mono',monospace;font-size:13px}}
.tv-link{{color:#6e7681;text-decoration:none;display:inline-flex;align-items:center;gap:5px}}
.tv-link:hover{{color:#58a6ff}}
.tv-icon{{font-size:12px;color:#484f58;transition:color .15s}}
.tv-link:hover .tv-icon{{color:#58a6ff}}
.badge{{display:inline-block;padding:4px 11px;border-radius:5px;font-size:11px;font-weight:700;color:#000;font-family:'JetBrains Mono',monospace}}
.empty{{max-width:1100px;margin:60px auto;text-align:center;color:#484f58;font-family:'JetBrains Mono',monospace}}
.foot{{max-width:1100px;margin:28px auto 0;font-size:11px;color:#30363d;font-family:'JetBrains Mono',monospace;text-align:center;padding-top:16px;border-top:1px solid #21262d}}
</style></head><body>
<div class="hdr">
  <div><h1>CE Scanner</h1><p>4H · Heikin Ashi · ATR(1)×3 · ZLSMA(50) · Pine Script v6</p></div>
  <div class="meta">
    <div class="cnt {'z' if count==0 else ''}">{count}</div>
    <div>signals found</div>
    <div class="scanned">{total} stocks scanned</div>
    <div style="margin-top:8px;color:#8b949e">{scan_time}</div>
  </div>
</div>
{warn}{table}
<div class="foot">
  Click symbol → TradingView 4H chart &nbsp;·&nbsp; CE buy = direction flip short→long on HA 4H<br>
  ZLSMA = 2×linreg(close,50) − linreg(linreg(close,50),50) &nbsp;·&nbsp; Fresh signals: today + yesterday only
</div>
</body></html>"""

def main():
    is_open = bar_is_open()
    now     = datetime.now(IST)
    print(f"CE Scanner starting — {now.strftime('%d %b %Y %H:%M IST')}")
    print(f"Bar: {'OPEN' if is_open else 'CLOSED'}")

    symbols = load_symbols("nse_list.csv")
    total   = len(symbols)
    print(f"Scanning {total} stocks...")

    results = []
    done    = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(scan, sym): sym for sym in symbols}
        for f in as_completed(futures):
            done += 1
            print(f"[{done}/{total}]", end="\r")
            r = f.result()
            if r:
                results.append(r)
                print(f"  SIGNAL: {r['symbol']} — {r['bar']} — {r['time']}")

    results.sort(key=lambda x: (
        0 if x["bar"] == "Last Bar" else 1,
        -x["sig_ts"].timestamp(),
        x["symbol"]
    ))

    print(f"\nSignals found: {len(results)} / {total}")

    scan_time = now.strftime("%d %b %Y, %H:%M IST")
    html      = make_html(results, scan_time, is_open, total)
 
    # Save to docs folder (GitHub Pages publish_dir)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
    os.makedirs(out_dir, exist_ok=True)

    fname    = f"ce_signals_{now.strftime('%Y-%m-%d_%H-%M')}.html"
    out_path = os.path.join(out_dir, fname)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"  Report : {out_path}")

    # Auto-generate index.html with all reports listed (latest first)
    all_files = sorted(
        [f for f in os.listdir(out_dir) if f.startswith("ce_signals_") and f.endswith(".html")],
        reverse=True
    )

    def make_index(files, generated_at):
        rows = ""
        for i, f in enumerate(files):
            try:
                ts_part = f.replace("ce_signals_", "").replace(".html", "")
                dt = datetime.strptime(ts_part, "%Y-%m-%d_%H-%M")
                dt_ist = IST.localize(dt)
                label = dt_ist.strftime("%d %b %Y, %H:%M IST")
            except Exception:
                label = f
            badge = '<span style="background:#00c853;color:#000;font-size:10px;font-weight:700;padding:3px 8px;border-radius:4px;font-family:monospace;margin-left:10px">LATEST</span>' if i == 0 else ""
            rows += f"""<tr>
                <td class="num">{i+1}</td>
                <td class="lbl"><a href="{f}" class="link">{label}{badge}</a></td>
                <td class="fn">{f}</td>
            </tr>"""

        return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CE Scanner — Reports</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Space+Grotesk:wght@400;600;700&display=swap');
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#ffffff;color:#e2e8f0;font-family:'Space Grotesk',sans-serif;padding:40px 16px;min-height:100vh}}
.hdr{{max-width:860px;margin:0 auto 32px;border-bottom:1px solid #21262d;padding-bottom:20px}}
.hdr h1{{font-size:26px;font-weight:700;color:#6e7681}}
.hdr p{{font-size:12px;color:#ffffff;margin-top:6px;font-family:'JetBrains Mono',monospace}}
table{{width:100%;max-width:860px;margin:0 auto;border-collapse:collapse;border-radius:8px;overflow:hidden}}
thead tr{{background:#1e293b;border-bottom:2px solid #334155}}
th{{padding:12px 16px;font-size:10px;letter-spacing:1px;text-transform:uppercase;color:#ffffff;text-align:left;font-family:'JetBrains Mono',monospace}}
tbody tr{{border-bottom:1px solid #1e293b;transition:background .12s}}
tbody tr:hover{{background:#172033}}
td{{padding:14px 16px;font-size:14px;vertical-align:middle}}
td.num{{color:#484f58;font-family:'JetBrains Mono',monospace;font-size:12px;width:36px}}
td.fn{{font-family:'JetBrains Mono',monospace;font-size:11px;color:#484f58}}
.link{{color:#58a6ff;text-decoration:none;font-weight:600}}
.link:hover{{text-decoration:underline}}
.foot{{max-width:860px;margin:28px auto 0;font-size:11px;color:#30363d;font-family:'JetBrains Mono',monospace;text-align:center;padding-top:16px;border-top:1px solid #21262d}}
</style></head><body>
<div class="hdr">
  <h1>CE Scanner — All Reports</h1>
  <p>Total {len(files)} report(s) &nbsp;·&nbsp; Generated: {generated_at}</p>
</div>
<table>
  <thead><tr><th>#</th><th>Scan Time</th><th>File</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
<div class="foot">Latest report on top &nbsp;·&nbsp; Click to open report</div>
</body></html>"""

    index_html = make_index(all_files, now.strftime("%d %b %Y, %H:%M IST"))
    index_path = os.path.join(out_dir, "index.html")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(index_html)
    print(f"  Index  : {index_path}")

if __name__ == "__main__":
    main()
