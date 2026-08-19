# -*- coding: utf-8 -*-
"""Research panels: how a point-in-time panel gets built, and nothing else.

These helpers used to live inside a strategy module, which was a layering
mistake worth naming: *building a panel is not an opinion*. Two strategies that
disagree about every factor still need the same answer to "which stocks existed
on this date, and what were their trailing fields". Leaving that code inside one
strategy meant every other strategy imported that strategy to get a panel.

Three things happen here:

``compute_excluded`` / ``load_excluded``
    Price-integrity blacklist. Symbols whose adjusted series still shows an
    unexplained break are excluded **for the symbol set being asked about**,
    computed on the spot rather than read from a global file --- a stale global
    blacklist silently changes the universe of every later run.

``build_pit_panel``
    The formal path: monthly point-in-time candidate pool, membership applied
    per day, built over the light data path (price + institutional flow only)
    because the pool union runs to several hundred symbols and a full bundle
    exhausts the vendor's hourly quota. Equivalence with the full path is a
    tested property, not an assumption --- see ``live_signal.verify_equivalence``.

``build_panel``
    The legacy single-day static pool, kept **only** as a bias control arm. It
    must be requested explicitly (``static_universe_comparator=True``) so the
    engine marks the result ``formal_evidence_eligible=False``.

``attach_chip_fields``
    Attaches trailing institutional-flow and margin fields. Fields, not scores:
    what the market did, not what we think it means.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
import data
from universes import legacy_static
from backtest import event_backtest


def compute_excluded(symbols: List[str], write: bool = True) -> set:
    """對**傳入的股票集**算價格完整性排除名單(還原後仍有殘留斷點者)。

    ⚠ 不要改回「讀單一全域檔案」。排除名單是**隨 universe 而變**的:
    當期池 300 檔算出 18 檔、PIT 聯集 758 檔算出 45 檔。共用一個檔案時,
    後跑的流程會覆蓋前一個,下次另一條路徑就會被 fail-closed 閘門擋下
    (2026-08-06 實際發生:出上線名單後,研究報告重生就掛了)。
    當場算才不會有這種跨流程污染;`fetch_price` 有快取,成本很低。

    仍會寫出檔案,但那是**產出物供人工檢視**,不是真理來源。
    """
    import data
    from data import price_integrity as pi

    frames = {}
    for sid in symbols:
        p = data.fetch_price(sid)
        if p is not None and not p.empty:
            frames[sid] = p
    if not frames:
        return set()
    audit = pi.audit_price_frames(
        frames, threshold=config.PRICE_INTEGRITY_RETURN_THRESHOLD)
    bad = sorted(audit["stock_id"].unique()) if not audit.empty else []
    if write:
        try:
            json.dump(bad, open(config.OUTPUT_DIR / EXCLUDED_PATH, "w"), indent=1)
        except Exception:
            pass
    return set(bad)


def load_excluded() -> set:
    """讀上一次寫出的排除名單。**僅供檢視**,建構 panel 請用 compute_excluded。"""
    p = config.OUTPUT_DIR / EXCLUDED_PATH
    if not p.exists():
        return set()
    try:
        return set(json.load(open(p, encoding="utf-8")))
    except Exception:
        return set()


def build_panel(symbols: Optional[List[str]] = None,
                use_pit_pool: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """回傳 (稠密 panel, 乾淨 symbols)。

    三件事非做不可:
      1. 走 `event_backtest.build_research_panel()`(預設稠密)—— ts_ 算子需要連續個股序列。
      2. 排除價格完整性名單 —— 否則引擎的未還原價 fail-closed 閘門會擋下。
      3. `use_pit_pool=True`(預設)—— M 月候選池只用完整 M-1 曆月重建。

    為什麼預設走 PIT:靜態池是**單一日期**的成交值 top-N 套用整段歷史,等於用
    「今天知道誰熱門」決定兩年前能選誰。下列舊對照使用「生效日前 20 交易日」
    月池，只保留為偏誤案例；不代表目前完整上個曆月規則的績效:

        IS 中位 1.922 → 1.607(最小 1.352 → 0.762);IS 基準 1.42 → 1.13
        OS 中位 1.772 → 1.938;OS 基準 1.59 → 1.52

    偏誤把策略與基準**同時**灌水,所以超額(策略−基準)在 IS 幾乎不變
    (+0.50 → +0.48)、OS 反而變好(+0.18 → +0.42)。但絕對水準必須用 PIT 的。
    """
    from universes import legacy_static as uni

    if use_pit_pool:
        return build_pit_panel()

    if symbols is None:
        symbols = legacy_static.get_universe(top_n=config.DYNAMIC_UNIVERSE_CANDIDATE_POOL)
    excluded = compute_excluded(symbols)      # 對這組 symbols 當場算,不讀全域檔
    symbols = [s for s in symbols if s not in excluded]

    # static_universe_comparator=True:這條是 legacy 單日靜態池對照組(偏誤案例),
    # 必須顯式宣告,引擎才會放行並在 summary 標 formal_evidence_eligible=False。
    # 一律走公開入口 build_research_panel(預設稠密);策略模組不碰引擎私有的
    # _prepare_panel,免得哪天又拿到「只留成員日」的稀疏 panel 去算 ts_。
    panel = event_backtest.build_research_panel(
        symbols,
        dynamic_enabled=True,
        universe_top_n=config.DYNAMIC_UNIVERSE_TOP_N,
        static_universe_comparator=True,
    )
    return attach_chip_fields(panel), symbols


def build_pit_panel() -> Tuple[pd.DataFrame, List[str]]:
    """用完整上個曆月的逐月池建 panel(候選池成員資格逐日套用)。

    走精簡資料路徑(price + inst),因為 PIT 池聯集有 700+ 檔,完整 bundle 會
    撞爆 FinMind 的 600 次/小時。`live_signal.verify_equivalence` 已證明精簡路徑
    的 trend_ok / in_dynamic_universe 與 `_prepare_panel` 完全一致(136,841 列零差異)。
    """
    from universes import dynamic as du
    import live_signal
    from universes import historical_pit_universe

    # 正式歷史候選池的入口(月頻 PIT)。價格完整性黑名單要對「PIT 聯集」當場算,
    # 所以先取聯集再扣;扣完仍是聯集的子集,引擎的 PIT 一致性檢查會通過。
    pit = historical_pit_universe()
    provider = pit.provider
    union = sorted(set(pit.symbols) - compute_excluded(pit.symbols))

    panel = live_signal.build_light_panel(union, apply_membership=False)
    if panel.empty:
        return panel, []

    candidate_mask = provider.candidate_mask(panel)
    panel = du.add_membership(
        panel, top_n=config.DYNAMIC_UNIVERSE_TOP_N,
        lookback=config.DYNAMIC_UNIVERSE_LOOKBACK,
        min_obs=config.DYNAMIC_UNIVERSE_MIN_OBS,
        min_avg_volume_lots=config.DYNAMIC_UNIVERSE_MIN_AVG_VOLUME_LOTS,
        min_avg_turnover=config.DYNAMIC_UNIVERSE_MIN_AVG_TURNOVER,
        candidate_mask=candidate_mask,
    )
    # 不可先刪掉非候選月份的列：同一股票離開再回到候選池時，先刪列會讓 ts_ 的
    # 「20 列」跨過數月，製造錯誤視窗。候選資格只透過旗標在選股時套用。
    panel = panel.sort_values(["date", "stock_id"]).reset_index(drop=True)
    panel.attrs["universe"] = provider.metadata()
    # provider 本身也帶著走:run_once 要把它傳進引擎,summary 才留得住真實的
    # 候選池 metadata(external picks 路徑沒有 panel,引擎讀不到 attrs)。
    panel.attrs["universe_provider"] = provider
    return panel, union


def attach_chip_fields(panel: pd.DataFrame) -> pd.DataFrame:
    """併入法人分項(外資/投信/自營)淨買。base panel 只有合併後的 inst_6d。

    無申報日補 0(不向後延用舊值),與 factors._align 的慣例一致 —— 把「當日沒
    申報」延用成「當日有買」是典型的籌碼面前視。
    """
    import data

    frames = []
    for sid in sorted(panel["stock_id"].unique()):
        inst = data.fetch_institutional(sid)
        if inst is None or inst.empty:
            continue
        d = inst[["date", "foreign_net", "trust_net", "dealer_net"]].copy()
        d["date"] = pd.to_datetime(d["date"])
        d["stock_id"] = sid
        frames.append(d)
    if not frames:
        for c in ["foreign_net", "trust_net", "dealer_net"]:
            panel[c] = 0.0
        return panel
    chip = pd.concat(frames, ignore_index=True)
    density = panel_density.density_of(panel)
    # merge 會丟 attrs;用包裝把「記得補標」從人的紀律變成呼叫一個函式。
    out = panel_density.preserving_merge(
        panel, chip, on=["date", "stock_id"], how="left")
    for c in ["foreign_net", "trust_net", "dealer_net"]:
        out[c] = out[c].fillna(0.0)
    # merge 會把 attrs 丟掉,稠密度標籤要接回去 —— 標籤掉了 ts_ 的閘門就形同不存在。
    if density is not None:
        panel_density.tag(out, density)
    return out


# ── 報告 ────────────────────────────────────────────────────────────────