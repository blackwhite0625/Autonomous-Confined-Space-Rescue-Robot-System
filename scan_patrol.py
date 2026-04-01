"""
搜救機器人 — 掃描式巡邏模組 (模式 F) V4
=========================================
核心改進：360° 完整超聲波掃描，保證在任何地形找到出口。

策略：
  前方通暢 → 直走 + 攝影機掃描（~5 秒）
  前方堵住 → 原地 360° 旋轉掃描 6 方向 → 選最遠方向 → 轉向 → 前進（~12 秒）
  連續卡死 → 麥克納姆側移 + 360° 重掃

V3 的問題：只掃 ±45°（3 方向），在 90° 牆角看不到背後的出口。
V4 掃 360°（6 方向：0°, 60°, 120°, 180°, -120°, -60°），保證找到出口。
"""

import time
import math
import logging
import threading

import config

logger = logging.getLogger("rescue.scan_patrol")

# ── 速度參數 ──
DRIVE_SPEED = 0.30
DRIVE_SEC = 2.5
REVERSE_SPEED = 0.25
REVERSE_SEC = 0.5
SCAN_TURN_SPEED = 0.28       # 掃描旋轉速度（scanning_lock 保護，可快一點）
STRAFE_SPEED = 0.22
STRAFE_SEC = 0.4

# ── 掃描參數 ──
SCAN_SETTLE_SEC = 0.35       # 等超聲波更新
MIN_CLEAR_CM = 35
OBSTACLE_STOP_CM = 20

# ── 記憶參數 ──
EXPLORE_BONUS = 150

# 旋轉速率
_RAD_PER_SEC = SCAN_TURN_SPEED * getattr(config, 'HEAT_MAP_RAD_PER_SPEED_SEC', 2.0)
DEG_PER_SEC = max(_RAD_PER_SEC * 180 / math.pi, 10)

# ── 掃描鎖（讓 ultrasonic_loop 跳過安全煞車）──
scanning_lock = threading.Event()


class ScanPatrol:
    """掃描式巡邏 V4 — 360° 完整掃描，保證脫困"""

    def __init__(self, motor, servo, get_distance_fn,
                 get_victim_score_fn, check_active_fn,
                 add_log_fn, update_pan_fn, heat_map=None,
                 backtrack_engine=None):
        self.motor = motor
        self.servo = servo
        self._get_distance = get_distance_fn
        self._get_score = get_victim_score_fn
        self._active = check_active_fn
        self._log = add_log_fn
        self._update_pan = update_pan_fn
        self._heat_map = heat_map
        self._bt = backtrack_engine
        self._cycle = 0
        self._stuck_count = 0

    # ══════════════════════════════════════════════
    # 主入口
    # ══════════════════════════════════════════════

    def run_cycle(self):
        if not self._active():
            self.motor.stop()
            return

        self._cycle += 1
        dist = self._get_distance()
        logger.info(f"[F] #{self._cycle} dist={dist:.0f}cm stuck={self._stuck_count}")

        if dist > MIN_CLEAR_CM or dist < 0:
            # ════ 前方通暢：直走 + 攝影機掃描 ════
            self._stuck_count = 0

            # 熱區記憶：已探索方向 → 主動轉向
            if self._heat_map and self._heat_map.is_ahead_fully_scanned(look_cells=3):
                angle, unexplored = self._heat_map.get_best_exploration_angle()
                if unexplored > 2 and abs(angle) > 20:
                    self._log("info", f"已探索，轉向 {angle}")
                    self._safe_rotate(angle)
                    if not self._active():
                        return

            hit = self._drive_forward()
            if not self._active():
                self.motor.stop()
                return
            if not hit:
                self._camera_sweep()
        else:
            # ════ 前方堵住：360° 掃描找出口 ════
            self._stuck_count += 1

            if self._stuck_count >= 4:
                # 極端卡死 → 側移脫困
                self._strafe_escape()
                self._stuck_count = 0
                return

            self._log("info", f"堵住 {dist:.0f}cm (第{self._stuck_count}次)，360° 掃描")

            # 360° 超聲波掃描（回傳掃描數據 + 目前角度偏移）
            scan, scan_offset = self._full_circle_scan()
            if not self._active():
                self.motor.stop()
                return

            # 記錄到地圖
            if self._heat_map:
                try:
                    self._heat_map.mark_ultrasonic_scan(scan)
                except Exception:
                    pass

            # 找可通行方向
            clear = {a: d for a, d in scan.items() if d > MIN_CLEAR_CM}

            if clear:
                best = self._pick_best(scan, clear)

                # ═══ 關鍵改進：從當前位置計算最短旋轉 ═══
                # 掃描後車子在 scan_offset 度（順時針累計，通常 300° = -60°）
                # 目標是 best 度（相對於掃描起始方向）
                # 需要旋轉的角度 = best - (scan_offset 換算成相對角度)
                current_relative = scan_offset % 360
                if current_relative > 180:
                    current_relative -= 360  # 300° → -60°
                net_rotation = best - current_relative
                # 正規化到 -180 ~ +180
                while net_rotation > 180:
                    net_rotation -= 360
                while net_rotation < -180:
                    net_rotation += 360

                self._log("info", f"選 {best}° ({clear[best]:.0f}cm) 淨轉 {net_rotation:.0f}°")
                self._safe_rotate(net_rotation)
                if not self._active():
                    self.motor.stop()
                    return

                hit = self._drive_forward()
                if not hit:
                    self._stuck_count = 0
                    if self._active():
                        self._camera_sweep()
            else:
                # 6 方向全堵 → 從當前位置回到 0° 再後退
                net_back = -scan_offset % 360
                if net_back > 180:
                    net_back -= 360
                self._log("warn", f"360° 全堵！回正 {net_back:.0f}° + 後退 + 掉頭")
                self._safe_rotate(net_back)
                self.motor.move(0, -REVERSE_SPEED, 0)
                time.sleep(0.8)
                self.motor.stop()
                if self._bt:
                    self._bt.record("reverse", REVERSE_SPEED, 0.8)
                self._safe_rotate(180)

    # ══════════════════════════════════════════════
    # 360° 超聲波掃描（核心改進）
    # ══════════════════════════════════════════════

    def _full_circle_scan(self) -> tuple:
        """
        原地旋轉 360° 掃描 6 方向（每 60°）。
        掃描完不回 0°！回傳 (scan_data, current_offset)。
        呼叫者從 current_offset 計算最短旋轉到目標，避免累積 480° 旋轉誤差。
        """
        self.motor.stop()
        scan = {}
        step_deg = 60
        current_offset = 0  # 目前相對於起始的角度偏移

        scanning_lock.set()
        try:
            # 讀 0°（不轉）
            time.sleep(SCAN_SETTLE_SEC)
            scan[0] = self._get_distance()
            logger.info(f"[scan] 0°: {scan[0]:.0f}cm")

            # 順時針旋轉 5 步
            angles = [60, 120, 180, -120, -60]
            for angle_label in angles:
                if not self._active():
                    break
                self._rotate_raw(step_deg)
                current_offset += step_deg  # 累計已轉角度
                time.sleep(SCAN_SETTLE_SEC)
                d = self._get_distance()
                scan[angle_label] = d
                logger.info(f"[scan] {angle_label}°: {d:.0f}cm")

            # 不回 0°！保持在 current_offset（= 300° = -60° 位置）

        finally:
            scanning_lock.clear()
            self.motor.stop()

        summary = " | ".join(f"{a}:{d:.0f}" for a, d in sorted(scan.items()))
        self._log("info", f"掃描: {summary}")
        return scan, current_offset

    # ══════════════════════════════════════════════
    # 方向選擇
    # ══════════════════════════════════════════════

    def _pick_best(self, scan: dict, clear: dict) -> int:
        """從可通行方向中選最佳（距離 + 探索加成 + 前方偏好）。"""
        best_score = -1
        best_angle = 0

        for angle, dist in clear.items():
            score = dist

            # 探索加成
            if self._heat_map:
                try:
                    exp_angle, _ = self._heat_map.get_best_exploration_angle()
                    diff = abs(angle - exp_angle)
                    if diff > 180:
                        diff = 360 - diff
                    if diff < 40:
                        score += EXPLORE_BONUS
                    elif diff < 80:
                        score += EXPLORE_BONUS // 2
                except Exception:
                    pass

            # 前方偏好（少轉 = 省時間）
            if abs(angle) <= 60:
                score += 20

            if score > best_score:
                best_score = score
                best_angle = angle

        return best_angle

    # ══════════════════════════════════════════════
    # 前進
    # ══════════════════════════════════════════════

    def _drive_forward(self) -> bool:
        """慢速前進，途中監測超聲波。回傳 True=被障礙打斷。"""
        logger.info(f"[F] 前進 {DRIVE_SPEED} × {DRIVE_SEC}s")
        self.motor.move(0, DRIVE_SPEED, 0)
        t0 = time.time()
        hit = False
        no_echo = 0

        for _ in range(int(DRIVE_SEC / 0.1)):
            if not self._active():
                break
            time.sleep(0.1)
            d = self._get_distance()
            if d < 0:
                no_echo += 1
                if no_echo >= 3:
                    hit = True
                    break
            elif d < OBSTACLE_STOP_CM:
                hit = True
                break
            else:
                no_echo = 0

        self.motor.stop()
        dt = time.time() - t0

        if self._bt:
            self._bt.record("forward", DRIVE_SPEED, dt)
        if self._heat_map:
            self._heat_map.update_forward(DRIVE_SPEED, dt)
            self._heat_map.mark_scanned()

        if hit:
            logger.info(f"[F] 障礙打斷，後退")
            self.motor.move(0, -REVERSE_SPEED, 0)
            time.sleep(REVERSE_SEC)
            self.motor.stop()
            if self._bt:
                self._bt.record("reverse", REVERSE_SPEED, REVERSE_SEC)
            if self._heat_map:
                self._heat_map.update_reverse(REVERSE_SPEED, REVERSE_SEC)
                self._heat_map.mark_obstacle()

        return hit

    # ══════════════════════════════════════════════
    # 攝影機掃描（含超時保護）
    # ══════════════════════════════════════════════

    def _camera_sweep(self):
        """雲台掃描找人，最長 8 秒，凝視最長 2 秒。"""
        logger.info("[F] 攝影機掃描")
        t0 = time.time()
        pan = config.PATROL_PAN_MIN
        step = abs(config.PATROL_PAN_STEP)
        gaze = 0

        while pan <= config.PATROL_PAN_MAX:
            if not self._active() or time.time() - t0 > 8.0:
                break
            if self._get_score() >= config.VICTIM_SUSPECT_THRESHOLD:
                gaze += 1
                if gaze > 20:
                    gaze = 0
                else:
                    time.sleep(0.1)
                    continue
            gaze = 0
            self.servo.set_angle(pan, config.SERVO_DEFAULT_TILT)
            self._update_pan(pan)
            time.sleep(config.PATROL_SWEEP_DELAY)
            pan += step

        self.servo.set_angle(0, config.SERVO_DEFAULT_TILT)
        self._update_pan(0)
        logger.info(f"[F] 攝影機掃描完成 ({time.time()-t0:.1f}s)")
        if self._heat_map:
            self._heat_map.mark_scan_fan(config.PATROL_PAN_MIN, config.PATROL_PAN_MAX)

    # ══════════════════════════════════════════════
    # 側移脫困（4+ 次卡死）
    # ══════════════════════════════════════════════

    def _strafe_escape(self):
        self._log("warn", "4 次卡死，側移脫困")

        # 1. 後退
        for _ in range(3):
            if not self._active():
                self.motor.stop()
                return
            self.motor.move(0, -REVERSE_SPEED, 0)
            time.sleep(0.3)
            self.motor.stop()
            d = self._get_distance()
            if d > 0 and d > MIN_CLEAR_CM:
                break
        if self._bt:
            self._bt.record("reverse", REVERSE_SPEED, 0.9)

        if not self._active():
            self.motor.stop()
            return

        # 2. 側移
        sdir = 1 if self._cycle % 2 == 0 else -1
        for _ in range(3):
            if not self._active():
                self.motor.stop()
                return
            self.motor.move(STRAFE_SPEED * sdir, 0, 0)
            time.sleep(STRAFE_SEC)
            self.motor.stop()
            time.sleep(0.1)
            d = self._get_distance()
            if d > 0 and d > MIN_CLEAR_CM:
                break
        if self._bt:
            self._bt.record("strafe", STRAFE_SPEED, STRAFE_SEC * 3, sdir)

        # 3. 360° 掃描找出路
        if self._active():
            scan, offset = self._full_circle_scan()
            clear = {a: d for a, d in scan.items() if d > MIN_CLEAR_CM}
            if clear:
                best = max(clear, key=clear.get)
                # 從當前位置計算最短旋轉
                cur_rel = offset % 360
                if cur_rel > 180:
                    cur_rel -= 360
                net = best - cur_rel
                while net > 180: net -= 360
                while net < -180: net += 360
                self._safe_rotate(net)

    # ══════════════════════════════════════════════
    # 旋轉
    # ══════════════════════════════════════════════

    def _rotate_raw(self, degrees: float):
        """原始旋轉（掃描用，不記錄回溯，scanning_lock 已由呼叫者設定）。"""
        if abs(degrees) < 3:
            return
        direction = 1 if degrees > 0 else -1
        dur = abs(degrees) / DEG_PER_SEC
        remaining = dur

        while remaining > 0 and self._active():
            pulse = min(remaining, 0.4)
            self.motor.move(0, 0, SCAN_TURN_SPEED * direction)
            time.sleep(pulse)
            self.motor.stop()
            time.sleep(0.03)
            remaining -= pulse

        if self._heat_map:
            self._heat_map.update_turn(SCAN_TURN_SPEED, dur, direction)

    def _safe_rotate(self, degrees: float):
        """安全旋轉（巡邏用，記錄回溯，設 scanning_lock）。"""
        if abs(degrees) < 5:
            return
        direction = 1 if degrees > 0 else -1
        dur = abs(degrees) / DEG_PER_SEC

        scanning_lock.set()
        try:
            remaining = dur
            while remaining > 0 and self._active():
                pulse = min(remaining, 0.4)
                self.motor.move(0, 0, SCAN_TURN_SPEED * direction)
                time.sleep(pulse)
                self.motor.stop()
                time.sleep(0.03)
                remaining -= pulse
        finally:
            scanning_lock.clear()

        if self._bt:
            self._bt.record("turn", SCAN_TURN_SPEED, dur, direction)
        if self._heat_map:
            self._heat_map.update_turn(SCAN_TURN_SPEED, dur, direction)
