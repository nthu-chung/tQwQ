# -*- coding: utf-8 -*-
"""allowlisted 策略註冊表:`strategy_id` → factory。

為什麼是 allowlist 而不是「manifest 裡寫 import path」:研究規格 §5.5 明訂
「plugin 必須從 repo 內的策略 registry 解析,不接受 JSON 直接傳任意 Python
path」。一份 JSON 若能決定正式驗證流程要 import 什麼,凍結的 manifest 就不再
描述一套固定規則 —— 換一個 import path 就換了策略,而 rules hash 完全看不出來。

新增策略 = 在這裡註冊一行。註冊本身不代表任何證據等級;`evidence_status`
只是把策略自己的宣告帶出來,讓 runner 能把 fixture 與正式策略分開對待。
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List

_REGISTRY: Dict[str, Callable[[], Any]] = {}


def register(strategy_id: str, factory: Callable[[], Any]) -> None:
    if not strategy_id or not isinstance(strategy_id, str):
        raise ValueError("strategy_id 必須是非空字串")
    if strategy_id in _REGISTRY:
        raise ValueError(f"strategy_id 重複註冊:{strategy_id}")
    _REGISTRY[strategy_id] = factory


def available() -> List[str]:
    return sorted(_REGISTRY)


def resolve(strategy_id: str):
    """取得策略實例;未註冊一律 fail-closed(不猜、不動態 import)。"""
    if strategy_id not in _REGISTRY:
        raise KeyError(
            f"[fail-closed] 未註冊的 strategy_id={strategy_id!r}。"
            f"可用:{available()}。正式入口只從 registry 解析,"
            "不接受任意 import path(研究規格 §5.5)")
    strategy = _REGISTRY[strategy_id]()
    for attr in ("name", "version", "make_signals", "data_requirements",
                 "default_parameters"):
        if not hasattr(strategy, attr):
            raise TypeError(
                f"[fail-closed] {strategy_id} 缺少策略介面成員 {attr!r}")
    if str(getattr(strategy, "name", "")) != strategy_id:
        raise ValueError(
            f"[fail-closed] {strategy_id} 的 strategy.name="
            f"{getattr(strategy, 'name', None)!r} 與註冊 id 不一致;"
            "兩者必須相同,否則 manifest 記的策略與實際跑的不是同一個")
    return strategy


def evidence_status(strategy_id: str) -> str:
    """策略自己宣告的證據狀態(fixture 與正式策略要分得開)。"""
    return str(getattr(resolve(strategy_id), "evidence_status", "unspecified"))


def _register_builtin() -> None:
    """One explicit line per strategy. Deliberately **not** a directory scan.

    An allowlist means "somebody decided this may be run by the formal path".
    Auto-scanning would turn "drop a file in a folder" into a registration, which
    is the same problem as letting a JSON manifest name an arbitrary import path.

    What is registered here is a **control arm**, not a candidate. See the
    strategy's own docstring for why a hypothesis that is expected to lose is the
    most useful thing to keep in a public engine repo.
    """
    from strategies.h3_short_reversal import H3ShortReversal
    register("h3_short_reversal", H3ShortReversal)

_register_builtin()
