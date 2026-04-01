"""
CarBot 雲台舵機模組 (V2 — 平滑插值)
=====================================
攝影機 Pan/Tilt 雲台控制。

核心改進：
  - 背景執行緒以 30ms 間隔平滑插值到目標角度
  - 軟體 PWM 下大幅降低抖動（減少 PWM 寫入次數）
  - 到位後自動脫離 PWM 信號，徹底消除靜態抖動
  - 搖桿手動操控與自動掃描都經過同一平滑管線
"""

import time
import logging
import threading
import config

logger = logging.getLogger("rescue.servo")

# GPIO 已在 config.py 統一初始化（含 PiGPIOFactory）
try:
    from gpiozero import AngularServo
    GPIO_OK = config.GPIO_AVAILABLE
except ImportError:
    GPIO_OK = False


class ServoController:
    """攝影機雲台舵機控制器（平滑插值版）"""

    # ── 平滑參數 ──
    SMOOTH_STEP = 3          # 每 tick 最大移動度數
    SMOOTH_INTERVAL = 0.03   # tick 間隔（30ms ≈ 33Hz）
    DETACH_AFTER_TICKS = 15  # 穩定 ~0.45s 後脫離 PWM
    DEADZONE = 2             # 目標與現值差距 ≤ 此值 → 視為到位

    def __init__(self):
        self.gpio_ok = GPIO_OK
        self._target_pan = int(config.SERVO_DEFAULT_PAN)
        self._target_tilt = int(config.SERVO_DEFAULT_TILT)
        self.current_pan = self._target_pan
        self.current_tilt = self._target_tilt
        self._lock = threading.Lock()
        self._smooth_running = False

        if self.gpio_ok:
            try:
                self._pan = AngularServo(
                    config.SERVO_PAN_PIN,
                    min_angle=-90, max_angle=90,
                    min_pulse_width=config.SERVO_MIN_PULSE,
                    max_pulse_width=config.SERVO_MAX_PULSE,
                )
                self._tilt = AngularServo(
                    config.SERVO_TILT_PIN,
                    min_angle=-90, max_angle=90,
                    min_pulse_width=config.SERVO_MIN_PULSE,
                    max_pulse_width=config.SERVO_MAX_PULSE,
                )
                # 先歸位
                self._apply_angle(self.current_pan, self.current_tilt)
                time.sleep(0.4)
                self._detach()

                # 啟動平滑插值執行緒
                self._smooth_running = True
                self._smooth_thread = threading.Thread(
                    target=self._smooth_loop, daemon=True
                )
                self._smooth_thread.start()

                logger.info("✅ 舵機控制器初始化完成（平滑插值模式）")
            except Exception as e:
                logger.error(f"❌ 舵機初始化失敗: {e}")
                self.gpio_ok = False
        else:
            logger.warning("⚠️ 舵機控制器：模擬模式（GPIO 不可用）")

    # ──────────────────────────────────────────────
    # 公開 API
    # ──────────────────────────────────────────────

    def set_angle(self, pan, tilt):
        """
        設定目標角度（非阻塞）。
        背景執行緒會以平滑方式移動到目標位置。
        """
        pan = max(-90, min(90, int(pan)))
        tilt = max(-90, min(90, int(tilt)))
        with self._lock:
            self._target_pan = pan
            self._target_tilt = tilt

    def home(self):
        self.set_angle(config.SERVO_DEFAULT_PAN, config.SERVO_DEFAULT_TILT)

    def get_angles(self):
        return {"pan": self.current_pan, "tilt": self.current_tilt}

    # ──────────────────────────────────────────────
    # 平滑插值核心
    # ──────────────────────────────────────────────

    def _smooth_loop(self):
        """背景執行緒：以固定頻率將舵機平滑移動到目標位置"""
        stable_ticks = 0
        detached = False

        while self._smooth_running:
            with self._lock:
                tp = self._target_pan
                tt = self._target_tilt

            dp = tp - self.current_pan
            dt = tt - self.current_tilt

            # 已到位 → 計數穩定 tick，超過閾值就脫離 PWM
            if abs(dp) <= self.DEADZONE and abs(dt) <= self.DEADZONE:
                # 如果差距很小但還沒完全對齊，先跳到目標
                if self.current_pan != tp or self.current_tilt != tt:
                    self.current_pan = tp
                    self.current_tilt = tt

                stable_ticks += 1
                if stable_ticks >= self.DETACH_AFTER_TICKS and not detached:
                    self._detach()
                    detached = True
                time.sleep(self.SMOOTH_INTERVAL)
                continue

            # 有新目標 → 重置穩定計數
            stable_ticks = 0
            if detached:
                detached = False

            # 平滑步進
            step = self.SMOOTH_STEP
            new_pan = self.current_pan + max(-step, min(step, dp))
            new_tilt = self.current_tilt + max(-step, min(step, dt))
            self.current_pan = int(new_pan)
            self.current_tilt = int(new_tilt)

            # 寫入硬體
            self._apply_angle(self.current_pan, self.current_tilt)
            time.sleep(self.SMOOTH_INTERVAL)

    def _apply_angle(self, pan, tilt):
        """直接寫入 PWM 角度（僅在 smooth_loop 中呼叫）"""
        if not self.gpio_ok:
            return
        try:
            self._pan.angle = pan
            self._tilt.angle = tilt
        except Exception as e:
            logger.warning(f"舵機角度寫入失敗: {e}")

    def _detach(self):
        """停止 PWM 信號輸出，消除靜態抖動"""
        if not self.gpio_ok:
            return
        try:
            self._pan.value = None
            self._tilt.value = None
        except Exception:
            pass

    # ──────────────────────────────────────────────
    # 清理
    # ──────────────────────────────────────────────

    def cleanup(self):
        self._smooth_running = False
        if self.gpio_ok:
            self._detach()
            self._pan.close()
            self._tilt.close()
            logger.info("✅ 舵機 GPIO 已釋放")
