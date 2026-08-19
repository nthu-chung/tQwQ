# -*- coding: utf-8 -*-
"""holdout 使用紀錄的離線回歸測試(P1-3)。

原本的 bug(這支測試逐條釘住)
------------------------------
系統**完全沒有**「這段 OS 已經被看過」的紀錄,而 IS/OS 切點是由凍結資料自身的
首尾日決定的(`evaluation/splits.py` 錨在 `dts[-1]`),資料視窗兩端又隨
`SNAPSHOT_END_DATE` 滑動(`data.py` 的 `start = end - HISTORY_DAYS`)。實測三個
真實快照:

    快照 2026-06-22 → OS = 2025-11-19 ~ 2026-06-18
    快照 2026-08-06 → OS 起點變成 2026-01-05

亦即 2025-11-19 ~ 2026-01-04 這段**從 OS 變成 IS**。推進快照後重跑,同一段資料
會被第二次當成 holdout 報成 fresh OOS,而系統沒有任何欄位擋得住 —— forward-only
又已經是唯一剩下的證據升級路徑。

另外兩個原本不成立的性質:
  - manifest 只凍切割**參數**,沒有釘住解出來的 IS/embargo/OS **日期**。
  - append-only 只靠「用 'a' 模式開檔」,既有列被事後改寫沒有任何人看得出來。

⚠ 這裡的假 summary / 合成 panel 只驗證揭露紀錄與旗標的行為,不代表任何策略績效;
the legacy strategy line 的證據等級仍是 blocked,其既有 OS 仍是 consumed / pseudo-OOS。
"""
from __future__ import annotations

import json
import tempfile
import threading
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import config
from evaluation import holdout

T0 = datetime(2026, 8, 15, 9, 0, 0)


class _LedgerCase(unittest.TestCase):
    """每個測試都用自己的 outputs/,絕不碰真實揭露紀錄。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.ledger = self.out / holdout.LEDGER_NAME
        p = mock.patch.object(config, "OUTPUT_DIR", self.out)
        p.start()
        self.addCleanup(p.stop)

    def lines(self):
        return [ln for ln in self.ledger.read_text(encoding="utf-8").splitlines()
                if ln.strip()]

    def reveal(self, *, os_start, os_end, strategy_hash="hash_a",
               strategy_name=None, now=T0, **kw):
        return holdout.record_reveal(
            strategy_hash=strategy_hash, strategy_name=strategy_name,
            os_start=os_start, os_end=os_end,
            source="tests", now=now, **kw)


# ── 1. 第一次揭露入帳 / 第二次標 previously_seen ──────────────────────────
class LedgerImmutabilityTest(_LedgerCase):
    def test_existing_rows_are_never_rewritten(self):
        first = self.reveal(os_start="2025-11-19", os_end="2026-06-18")
        body_after_first = self.lines()[0]
        for i in range(3):
            self.reveal(os_start="2026-06-19", os_end="2026-08-04",
                        now=datetime(2026, 8, 16 + i, 9, 0, 0))
        rows = self.lines()
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0], body_after_first, "既有列不可被改寫")
        self.assertEqual([json.loads(r)["seq"] for r in rows], [1, 2, 3, 4])
        self.assertEqual(json.loads(rows[0])["reveal_at"], first["reveal_at"])

    def test_tampered_row_is_detected(self):
        """靜默重寫是這份揭露紀錄唯一的致命傷 —— 改過必須讀不出來,而不是照常回傳。"""
        self.reveal(os_start="2025-11-19", os_end="2026-06-18")
        self.reveal(os_start="2026-06-19", os_end="2026-08-04",
                    now=datetime(2026, 8, 16, 9, 0, 0))
        rows = self.lines()
        doctored = json.loads(rows[0])
        doctored["os_start"] = "2026-05-01"       # 把「看過的範圍」改小
        self.ledger.write_text(
            json.dumps(doctored, ensure_ascii=False) + "\n" + rows[1] + "\n",
            encoding="utf-8")
        with self.assertRaisesRegex(holdout.HoldoutLedgerError, "被事後改過"):
            holdout.read_ledger()

    def test_deleted_row_breaks_the_chain(self):
        self.reveal(os_start="2025-11-19", os_end="2026-06-18")
        self.reveal(os_start="2026-06-19", os_end="2026-08-04",
                    now=datetime(2026, 8, 16, 9, 0, 0))
        rows = self.lines()
        self.ledger.write_text(rows[1] + "\n", encoding="utf-8")   # 抽掉第一列
        with self.assertRaisesRegex(holdout.HoldoutLedgerError, "prev_sha256"):
            holdout.read_ledger()

    def test_broken_ledger_blocks_new_reveals(self):
        """揭露紀錄壞掉時不可以「當成空的」繼續寫 —— 那等於用一次寫入洗掉歷史。"""
        self.reveal(os_start="2025-11-19", os_end="2026-06-18")
        self.ledger.write_text("{ not json\n", encoding="utf-8")
        with self.assertRaises(holdout.HoldoutLedgerError):
            self.reveal(os_start="2026-06-19", os_end="2026-08-04")

    def test_deleting_the_whole_ledger_is_not_a_clean_slate(self):
        """**整檔刪除**必須看得見 —— 這是雜湊鏈唯一擋不到的形狀。

        原 bug(2026-08-15 審查實測):鏈只擋改列/刪列/插列,`read_ledger` 對不
        存在的檔直接 `return []`,所以 `os.remove(outputs/holdout_ledger.jsonl)`
        之後,同 hash 同窗立刻回報 `fresh`、零警告 —— append-only 的紀錄被一個
        `rm` 洗掉。而這兩份 ledger 又被 `.gitignore` 的 `outputs/*` 排除,連
        「檔案不見了」都不會出現在 git status。
        """
        first = self.reveal(os_start="2026-09-01", os_end="2026-10-31")
        self.assertTrue(holdout.checkpoint_path().exists())
        self.ledger.unlink()

        with self.assertRaisesRegex(holdout.HoldoutLedgerError, "刪除或截斷"):
            holdout.read_ledger()
        # 也不得靠 reveal_status「查一下」就繞過(那正是回報 fresh 的入口)
        with self.assertRaises(holdout.HoldoutLedgerError):
            holdout.reveal_status(strategy_hash=first["strategy_hash"],
                                  strategy_name=None,
                                  os_start="2026-09-01", os_end="2026-10-31")
        # 更不得靠「再寫一列」重新開始(那會把揭露紀錄洗成 seq=1 的新鏈)
        with self.assertRaises(holdout.HoldoutLedgerError):
            self.reveal(os_start="2026-09-01", os_end="2026-10-31",
                        now=datetime(2026, 8, 16, 9, 0, 0))

    def test_truncating_the_ledger_is_detected(self):
        """留著檔案但砍掉尾巴 = 鏈仍然自洽,只有列數指紋看得出來。"""
        self.reveal(os_start="2026-09-01", os_end="2026-10-31")
        self.reveal(os_start="2026-11-01", os_end="2026-12-31",
                    now=datetime(2026, 8, 16, 9, 0, 0))
        rows = self.lines()
        self.ledger.write_text(rows[0] + "\n", encoding="utf-8")
        with self.assertRaisesRegex(holdout.HoldoutLedgerError, "刪除或截斷"):
            holdout.read_ledger()

    def test_checkpoint_tracks_row_count_and_last_hash(self):
        """指紋內容就是「幾列 + 末列 record_sha256」,每次 append 同步更新。"""
        self.assertIsNone(holdout.read_checkpoint())
        for i in range(3):
            rec = self.reveal(os_start="2026-09-01", os_end="2026-10-31",
                              strategy_hash=f"hash_{i}",
                              now=datetime(2026, 8, 15, 9, i, 0))
            cp = holdout.read_checkpoint()
            self.assertEqual(cp["rows"], i + 1)
            self.assertEqual(cp["last_record_sha256"], rec["record_sha256"])
        self.assertEqual(len(self.lines()), 3)

    def test_corrupt_checkpoint_fails_closed(self):
        """指紋壞掉不得當成「沒有指紋」放行 —— 否則刪揭露紀錄只要順手弄壞指紋。"""
        self.reveal(os_start="2026-09-01", os_end="2026-10-31")
        holdout.checkpoint_path().write_text("{ not json", encoding="utf-8")
        with self.assertRaises(holdout.HoldoutLedgerError):
            holdout.read_ledger()

    def test_fresh_clone_without_ledger_or_checkpoint_starts_empty(self):
        """兩份都不存在 = 從來沒揭露過(乾淨 clone),照常回空 list。"""
        self.assertEqual(holdout.read_ledger(), [])
        self.assertEqual(holdout.verify_ledger(), 0)

    def test_concurrent_reveals_do_not_overwrite_each_other(self):
        """併發揭露:每一次都要留下自己的一列,而且整條鏈仍然接得起來。

        沒有檔案鎖時,兩個 process 會同時讀到「揭露紀錄是空的」→ 兩列都寫
        `prev_sha256=genesis`、`seq=1`,鏈斷掉、而且兩邊都宣稱 fresh。
        """
        errors: list = []

        def worker(i: int):
            try:
                self.reveal(os_start="2025-11-19", os_end="2026-06-18",
                            strategy_hash=f"hash_{i}",
                            now=datetime(2026, 8, 15, 9, i, 0))
            except Exception as exc:            # pragma: no cover - 失敗才會進來
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        self.assertEqual(errors, [])
        rows = holdout.read_ledger()             # 會順便驗鏈
        self.assertEqual(len(rows), 6)
        self.assertEqual(sorted(r["seq"] for r in rows), list(range(1, 7)))
        self.assertEqual(len({r["strategy_hash"] for r in rows}), 6)


# ── 3. the legacy strategy line 既有 OS 維持 consumed / pseudo-OOS ─────────────────────────────
class _FakeProvider:
    all_symbols = ["A", "B", "C", "D"]

    def metadata(self):
        return {"candidate_rule": "month_M_uses_only_calendar_month_M_minus_1"}


def _fake_summary(n: int) -> dict:
    return {"sharpe": 0.5, "ann_ret": 0.1, "ann_vol": 0.2, "max_drawdown": -0.1,
            "n_trades": 40, "win_rate": 0.5, "payoff_ratio": 2.0,
            "eval_audit": {"days_beyond_last_pick": 0},
            "universe": {"candidate_pool_pit": True}}


class RunFullHoldoutTest(_LedgerCase):
    def _run_full(self):
        from backtest import event_backtest
        import evaluation.splits as evaluation_split

        dates = pd.bdate_range("2024-01-01", periods=120)

        def fake_portfolio(*_a, **kwargs):
            start = pd.Timestamp(kwargs.get("start_date") or dates[0])
            end = pd.Timestamp(kwargs.get("end_date") or dates[-1])
            eq = dates[(dates >= start) & (dates <= end)]
            return {
                "summary": {
                    "n_trades": 5, "ann_ret": 0.1, "sharpe": 1.0,
                    "max_drawdown": -0.1, "cum_ret": 0.1,
                    "data": {"integrity_bypassed": False},
                    "universe": {"survivorship_free": True},
                    "eval_audit": {"eval_window": [str(eq[0].date()),
                                                   str(eq[-1].date())]},
                },
                "equity_curve": pd.DataFrame({"date": eq, "equity": 1.0}),
                "trades": pd.DataFrame(),
            }

        with (
            mock.patch.object(event_backtest.uni, "get_universe", return_value=["A"]),
            mock.patch.object(event_backtest, "backtest_portfolio",
                              side_effect=fake_portfolio),
            mock.patch.object(event_backtest, "factor_ic", return_value=pd.DataFrame()),
        ):
            result, _ = event_backtest.run_full(sample=True, top_n=1, rebalance_every=3,
                                          dynamic_enabled=False)
        split = evaluation_split.build_evaluation_split(
            dates, minimum_embargo_days=config.BT_IC_HORIZON)
        return result, split

    def test_run_full_records_os_reveal_with_split_boundaries(self):
        result, split = self._run_full()
        rec = result["holdout"]
        self.assertIsNotNone(rec)
        self.assertEqual([rec["os_start"], rec["os_end"]], list(split.os_window))
        self.assertEqual(rec["is_window"], list(split.is_window))
        self.assertEqual(rec["embargo_trading_days"], split.n_embargo)
        self.assertEqual(rec["source"], "event_backtest.run_full")
        self.assertFalse(rec["holdout_previously_seen"])
        # smoke sample 仍然看過那段資料 → 照樣入帳,只是標成非正式證據
        self.assertFalse(rec["context"]["formal_evidence_eligible"])

    def test_rerunning_run_full_flags_previously_seen(self):
        self._run_full()
        result, _ = self._run_full()
        self.assertEqual(len(self.lines()), 2)
        self.assertTrue(result["holdout"]["holdout_previously_seen"])
        self.assertEqual(result["holdout"]["holdout_status"], "consumed")


if __name__ == "__main__":
    unittest.main()
