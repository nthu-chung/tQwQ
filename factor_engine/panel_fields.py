# -*- coding: utf-8 -*-
"""
傳統多因子計算模組
==============
輸入單檔股票的 bundle（price / inst / margin），輸出一個對齊到「交易日」的
DataFrame，每一列是某一天、每一欄是一個因子值或標準化分數。

為什麼要算「每一天」？
  回測需要在每個歷史日取得「當時」的因子值。只算最後一天無法回測。

防未來函數（point-in-time）：
  - 法人 / 融資資料用 merge_asof 對齊到價格日，且只會用「<= 當日」的最近一筆。
  - 所有 rolling 計算都是因果的（只看過去），pandas rolling 預設即如此。
  - 訊號在第 T 日收盤後產生，回測在 T+1 開盤進場（見 backtest.py）。

每個因子提供兩種輸出：
  - 原始值欄位（給回測 IC 分析、給人看）
  - *_score 欄位：0~1 標準化分數（給多因子加權評分）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


# ── 小工具：把任意值壓到 0~1 ────────────────────────────────────────────
def _clip01(x):
    return float(min(1.0, max(0.0, x)))


def _scale(value, lo, hi):
    """線性映射 [lo,hi] -> [0,1]，超出範圍夾住。"""
    if hi == lo:
        return 0.0
    return _clip01((value - lo) / (hi - lo))


# ── 對齊 ────────────────────────────────────────────────────────────────
def _align(price: pd.DataFrame, inst: pd.DataFrame, margin: pd.DataFrame) -> pd.DataFrame:
    """以 price 的交易日為主軸，把 inst / margin 以 asof（<=當日）對齊。"""
    df = price.copy().sort_values("date").reset_index(drop=True)

    inst_cols = ["foreign_net", "trust_net", "dealer_net", "inst_net"]
    if inst is not None and not inst.empty:
        # 法人買賣超是「流量(flow)」：沒有申報的交易日代表當日淨額 = 0，不可向後
        # 延用舊值（merge_asof backward 會把前一次的買超灌到無申報日，虛增 inst_1d/
        # 6d/12d，並讓 rotation_research 的 inst_6d>0 群組濾網在無申報日假通過）。
        # 改成以交易日為軸精確 left-merge + 缺漏補 0，與 market_flow_monitor 一致。
        cols = [c for c in inst_cols if c in inst.columns]
        inst_s = inst[["date"] + cols].sort_values("date")
        df = df.merge(inst_s, on="date", how="left")
        for c in inst_cols:
            df[c] = df[c].fillna(0.0) if c in df.columns else 0.0
    else:
        for c in inst_cols:
            df[c] = 0.0

    if margin is not None and not margin.empty:
        margin_s = margin.sort_values("date")
        df = pd.merge_asof(df, margin_s, on="date", direction="backward")
    else:
        for c in ["margin_balance", "short_balance", "margin_limit",
                  "margin_change", "short_change"]:
            df[c] = np.nan

    return df


# ── 相對強勢 / 抗跌：滾動下行統計 ───────────────────────────────────────
def _rolling_downside_stats(stock_ret: np.ndarray, mkt_ret: np.ndarray,
                            window: int, min_down: int):
    """
    對每個時點 t，用過去 `window` 天中「大盤下跌日」計算：
      - 下行 beta：cov(個股, 大盤 | 大盤跌) / var(大盤 | 大盤跌)
                   低/負 = 大盤跌時個股跟跌少，抗跌。
      - 下跌日相對報酬：mean(個股日報酬 − 大盤日報酬 | 大盤跌)
                        >0 = 大盤跌時個股相對抗跌。
    全因果（只看 t 之前含 t 的視窗）。下跌日不足 min_down 回 NaN。
    """
    n = len(stock_ret)
    beta = np.full(n, np.nan)
    dd_excess = np.full(n, np.nan)
    s = stock_ret.astype(float)
    mk = mkt_ret.astype(float)
    for t in range(window - 1, n):
        sw = s[t - window + 1:t + 1]
        mw = mk[t - window + 1:t + 1]
        valid = ~(np.isnan(sw) | np.isnan(mw))
        sw = sw[valid]; mw = mw[valid]
        mask = mw < 0
        k = int(mask.sum())
        if k < min_down:
            continue
        sd = sw[mask]; md = mw[mask]
        var = md.var()
        if var > 0:
            beta[t] = float(np.cov(sd, md, ddof=0)[0, 1] / var)
        dd_excess[t] = float((sd - md).mean())
    return beta, dd_excess


def _attach_relative_strength(df: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """
    以 df（個股，已對齊交易日）與 market（大盤 TAIEX）算相對強勢 / 抗跌因子。
    把大盤收盤用 merge_asof(backward) 對齊到個股交易日（防未來函數：只用 ≤當日）。
    market 缺失時，相關欄位全部給 NaN（分數階段會轉成 0，不影響既有因子）。
    """
    n = len(df)
    if market is None or market.empty:
        df["mkt_close"] = np.nan
        df["rs_excess"] = np.nan
        df["downside_beta"] = np.nan
        df["down_day_excess"] = np.nan
        return df

    mkt = market[["date", "close"]].rename(columns={"close": "mkt_close"}).copy()
    # 統一 datetime 精度，避免 merge_asof 因 ns/us dtype 不一致而報錯
    mkt["date"] = mkt["date"].astype("datetime64[ns]")
    mkt = mkt.sort_values("date")
    df = df.copy()
    df["date"] = df["date"].astype("datetime64[ns]")
    df = pd.merge_asof(df.sort_values("date"), mkt, on="date", direction="backward")

    close = df["close"]
    mkt_close = df["mkt_close"]

    # (1) 相對強勢：60日「相對大盤」超額報酬 = 個股報酬 − 大盤報酬（比值型，scale-free）
    stock_lb = close / close.shift(config.RS_LOOKBACK)
    mkt_lb = mkt_close / mkt_close.shift(config.RS_LOOKBACK)
    df["rs_excess"] = stock_lb / mkt_lb - 1.0

    # (2)/(3) 下行 beta + 下跌日相對報酬（滾動視窗，只看大盤下跌日）
    stock_ret = close.pct_change().values
    mkt_ret = mkt_close.pct_change().values
    beta, dd_excess = _rolling_downside_stats(
        stock_ret, mkt_ret, config.DOWNSIDE_WINDOW, config.DOWNSIDE_MIN_DOWN_DAYS)
    df["downside_beta"] = beta
    df["down_day_excess"] = dd_excess
    return df


# ── 主函式 ──────────────────────────────────────────────────────────────
def compute_factors(bundle: dict) -> pd.DataFrame:
    """
    回傳含因子值與分數的 DataFrame（每日一列）。
    若 price 不足以計算 MA60，回傳空。
    """
    price = bundle.get("price")
    if price is None or price.empty or len(price) < config.MA_LONG + 5:
        return pd.DataFrame()

    df = _align(price, bundle.get("inst"), bundle.get("margin"))
    df = _attach_relative_strength(df, bundle.get("market"))

    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"]  # 股

    # ── 技術指標基礎 ────────────────────────────────────────────────
    ma_s = close.rolling(config.MA_SHORT).mean()
    ma_l = close.rolling(config.MA_LONG).mean()
    std_s = close.rolling(config.BBANDS_WIN).std(ddof=0)
    bb_mid = close.rolling(config.BBANDS_WIN).mean()
    bb_upper = bb_mid + config.BBANDS_K * std_s
    bb_lower = bb_mid - config.BBANDS_K * std_s

    df["ma_short"] = ma_s
    df["ma_long"] = ma_l
    df["ma_long_slope"] = ma_l.diff(5)  # MA60 5日斜率

    # 布林位階：(close - 中軌) / (K*std)，0=中軌(月線), +1=上軌, -1=下軌
    bb_pos = (close - bb_mid) / (config.BBANDS_K * std_s.replace(0, np.nan))
    df["bb_pos"] = bb_pos

    # 均線糾結：短期 BIAS = |close-MA20|/MA20、中期 = |close-MA60|/MA60
    df["bias_short"] = (close - ma_s).abs() / ma_s
    df["bias_mid"] = (close - ma_l).abs() / ma_l

    # N 日新高（含今日），與距離新高的位置
    roll_high = high.rolling(config.HIGH_LOOKBACK).max()
    df["roll_high"] = roll_high
    df["near_high"] = close / roll_high  # 越接近 1 = 越靠近新高

    # 動能：MOM_LOOKBACK 日報酬（找「下一波成長」的核心：強者恆強）
    df["mom_ret"] = close / close.shift(config.MOM_LOOKBACK) - 1.0

    # 量能：近5日均量 / 前5日均量（窒息量 < 0.5）
    v5 = vol.rolling(5).mean()
    v5_prev = vol.shift(5).rolling(5).mean()
    df["vol_ratio"] = v5 / v5_prev
    df["avg_vol_lots"] = (vol.rolling(20).mean() / 1000.0)  # 近20日均量(張)

    # ── 籌碼指標 ────────────────────────────────────────────────────
    # 正規化分母：近20日均量(股)。法人淨買累積 / 均量 = 佔量比(天數當量)
    norm = vol.rolling(config.INST_NORM_WINDOW).mean().replace(0, np.nan)
    inst_net = df["inst_net"].fillna(0)

    df["inst_1d"] = inst_net.rolling(config.INST_WIN_SHORT).sum() / norm
    df["inst_6d"] = inst_net.rolling(config.INST_WIN_MID).sum() / norm
    df["inst_12d"] = inst_net.rolling(config.INST_WIN_LONG).sum() / norm

    # 下跌日法人仍買：近5日中，收黑(close<前一日)但 inst_net>0 的天數
    down_day = (close < close.shift(1))
    inst_buy = (inst_net > 0)
    dip_buy = (down_day & inst_buy).rolling(5).sum()
    df["inst_dip_buy_days"] = dip_buy  # 0~5

    # 資券比 = 融資餘額 / 融券餘額
    mb = df.get("margin_balance")
    sb = df.get("short_balance")
    if mb is not None and sb is not None:
        df["margin_short_ratio"] = mb / sb.replace(0, np.nan)
    else:
        df["margin_short_ratio"] = np.nan

    # ── 趨勢保護硬門檻 ──────────────────────────────────────────────
    df["trend_ok"] = (
        (df["ma_short"] > df["ma_long"])
        & (df["ma_long_slope"] > 0)
        & (close > df["ma_long"])
    )

    # Factor *scores* deliberately do not live here.
    #
    # This module computes **panel fields** --- price-derived series, moving
    # averages, trailing institutional flow, margin ratios. They are inputs: a
    # field is something the market did, and two strategies that disagree about
    # everything else still agree on what "20-day average turnover" means.
    #
    # A *score* is an opinion (which field matters, how to scale it, how to
    # weight it against another). Opinions belong to a strategy, declared in its
    # own `score()` and carried in its rules hash. Putting them here made every
    # strategy silently inherit a scoring scheme nobody declared --- including a
    # hard moving-average gate that turned out to be strictly dominated.
    #
    # See `strategy_kit/signal_builder.py` for where opinions are supposed to go.
    return df


SMOKE_SCORE_COLUMN = "composite"


def composite_score(row) -> float:
    """Row-wise smoke score. **Not a strategy.** See `smoke_composite`.

    Kept as a row-wise entry point because the engine's legacy driver applies it
    per row, and infrastructure tests patch this name to inject a deterministic
    ranking. There is deliberately no weighting scheme here to configure ---
    weights are an opinion, and opinions belong to a strategy's rules hash.
    """
    import hashlib
    import pandas as _pd

    d = row["date"]
    day = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10]
    key = f"{row['stock_id']}|{day}".encode()
    return int.from_bytes(hashlib.blake2s(key, digest_size=2).digest(), "big") % 10000 / 100.0


def smoke_composite(panel):
    """A deliberately arbitrary 0--100 score, so the legacy end-to-end driver has
    *something* to rank.

    **This is not a strategy and its output must never be treated as candidates.**

    It exists because the engine's oldest entry point builds its own picks, and a
    lot of infrastructure tests use that entry point as a convenient end-to-end
    driver (does provenance get recorded? does the split reach every phase? does
    the dense-panel flag survive?). Those tests need a ranking to exist; they do
    not care what it means.

    Why arbitrary on purpose: the previous occupant of this slot was a real
    nine-factor weighted composite, which meant the *engine contained a
    strategy*. Both of its knobs were global config, so two runs with the same
    strategy rule hash could trade different stocks --- the exact failure the
    two-layer identity is built to prevent, and it fails silently. Replacing it
    with a scorer that is obviously not alpha keeps the driver working and makes
    it impossible to mistake the output for research.

    The score is a stable function of the symbol and the date only --- no price,
    no volume, no flow. Nothing here can accidentally look predictive.
    """
    import hashlib

    def _h(row):
        key = f"{row['stock_id']}|{row['date']:%Y-%m-%d}".encode()
        return int.from_bytes(hashlib.blake2s(key, digest_size=2).digest(), "big") % 10000 / 100.0

    return panel.apply(_h, axis=1)


if __name__ == "__main__":
    import data
    b = data.fetch_bundle("2330")
    f = compute_factors(b)
    print(f"panel field rows: {len(f)}")
    cols = ["date", "close", "trend_ok", "inst_6d", "inst_12d", "bb_pos",
            "margin_short_ratio"]
    print(f[[c for c in cols if c in f.columns]].tail(5).to_string())
