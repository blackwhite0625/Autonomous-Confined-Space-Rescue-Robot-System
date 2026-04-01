"""
搜救機器人 — 路徑回溯模組
============================
記錄巡邏時的每個移動動作，回溯時反轉回放原路返回。

使用方式：
  巡邏中：bt.record("forward", 0.22, 1.8)
  回溯時：bt.execute_backtrack()  → 自動 pop + 反轉每一步
"""

import time
import logging
from dataclasses import dataclass

logger = logging.getLogger("rescue.backtrack")

MAX_STACK = 2000


@dataclass
class MoveRecord:
    """單筆移動記錄"""
    action: str      # "forward" | "reverse" | "turn" | "strafe"
    speed: float     # 實際使用的速度值
    duration: float  # 實際耗時（秒，量測值）
    direction: int   # turn/strafe: +1=右, -1=左; forward/reverse: 0


class BacktrackEngine:
    """路徑回溯引擎"""

    def __init__(self, motor, get_distance_fn, check_cancel_fn, add_log_fn):
        self.motor = motor
        self._get_distance = get_distance_fn
        self._check_cancel = check_cancel_fn
        self._log = add_log_fn
        self._stack = []
        self._is_backtracking = False

    # ── 屬性 ──

    @property
    def stack_size(self) -> int:
        return len(self._stack)

    @property
    def is_backtracking(self) -> bool:
        return self._is_backtracking

    # ── 記錄 ──

    def record(self, action: str, speed: float, duration: float, direction: int = 0):
        """記錄一筆移動動作（巡邏中呼叫）"""
        if self._is_backtracking:
            return
        if duration < 0.05:
            return
        self._stack.append(MoveRecord(action, speed, duration, direction))
        if len(self._stack) > MAX_STACK:
            self._stack = self._stack[-MAX_STACK:]

    def clear(self):
        """清空記錄（新任務開始時呼叫）"""
        self._stack.clear()
        logger.info("回溯記錄已清空")

    # ── 回溯執行 ──

    def execute_backtrack(self):
        """反轉回放所有記錄的動作，原路返回。"""
        self._is_backtracking = True
        total = len(self._stack)
        self._log("info", f"開始回溯，共 {total} 步")
        logger.info(f"開始路徑回溯 | {total} 步")

        completed = 0
        try:
            while self._stack:
                if self._check_cancel():
                    self._log("warn", "回溯已取消")
                    break

                # 安全檢查
                if not self._safety_check():
                    break

                cmd = self._stack.pop()
                self._execute_inverse(cmd)
                completed += 1

                # 每 10 步報告進度
                if completed % 10 == 0:
                    remaining = len(self._stack)
                    self._log("info", f"回溯進度: {completed}/{total} (剩 {remaining})")

        finally:
            self.motor.stop()
            self._is_backtracking = False

        if not self._stack:
            self._log("info", f"回溯完成，共 {completed} 步")
        else:
            self._log("warn", f"回溯中斷於第 {completed} 步 (剩 {len(self._stack)})")

    def _safety_check(self) -> bool:
        """每步前檢查超聲波。回傳 True=安全可繼續。"""
        dist = self._get_distance()
        if 0 < dist < 20:
            self._log("warn", f"回溯中偵測障礙 {dist:.0f}cm，暫停...")
            self.motor.stop()
            time.sleep(1.0)
            dist = self._get_distance()
            if 0 < dist < 20:
                self._log("warn", "障礙仍在，停止回溯")
                return False
        return True

    def _execute_inverse(self, cmd: MoveRecord):
        """執行單筆動作的反轉"""
        if cmd.action == "forward":
            # 前進 → 後退
            self._safe_move(0, -cmd.speed, 0, cmd.duration)

        elif cmd.action == "reverse":
            # 後退 → 前進
            self._safe_move(0, cmd.speed, 0, cmd.duration)

        elif cmd.action == "turn":
            # 右轉 → 左轉（反轉 direction）
            omega = cmd.speed * (-cmd.direction)
            self.motor.move(0, 0, omega)
            time.sleep(cmd.duration)
            self.motor.stop()
            time.sleep(0.05)

        elif cmd.action == "strafe":
            # 右移 → 左移（反轉 direction）
            vx = cmd.speed * (-cmd.direction)
            self._safe_move(vx, 0, 0, cmd.duration)

    def _safe_move(self, vx: float, vy: float, omega: float, duration: float):
        """帶超聲波監測的移動（前進/後退/側移用）"""
        self.motor.move(vx, vy, omega)
        t0 = time.time()
        while time.time() - t0 < duration:
            if self._check_cancel():
                break
            time.sleep(0.1)
            # 前進方向才需要超聲波檢查（vy > 0 = 往前）
            if vy > 0:
                d = self._get_distance()
                if 0 < d < 20:
                    break
        self.motor.stop()
        time.sleep(0.05)
