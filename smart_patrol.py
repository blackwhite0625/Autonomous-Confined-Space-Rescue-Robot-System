"""
搜救機器人 — 智慧巡邏模組 (V3 探索導航版)
==========================================
走停掃描 + 熱區地圖導航：
  前進 → 停車掃描 → 查熱區地圖選方向 → 遇障礙轉彎。
  每次掃描後主動轉向未探索最多的方向，不再盲目直走。
  卡住偵測：同一格位停留超過 4 個週期 → 強制大角度脫困。
"""

import time
import logging

import config

logger = logging.getLogger("rescue.smart_patrol")


class SmartPatrol:
    """
    災後搜救智慧巡邏。
    呼叫 run_cycle() 執行一個完整週期（前進→掃描→導航/避障）。
    """

    def __init__(self, motor, servo, get_distance_fn,
                 get_victim_score_fn, check_active_fn,
                 add_log_fn, update_pan_fn, heat_map=None):
        self.motor = motor
        self.servo = servo
        self._get_distance = get_distance_fn
        self._get_score = get_victim_score_fn
        self._active = check_active_fn
        self._log = add_log_fn
        self._update_pan = update_pan_fn
        self._heat_map = heat_map

        self._turn_dir = 1        # 交替左右轉（避障用）
        self._consec_blocked = 0  # 連續被堵次數

        # 卡住偵測
        self._last_grid_pos = None
        self._same_pos_count = 0

    # ──────────────────────────────────────────────
    # 主入口：一個完整的巡邏週期
    # ──────────────────────────────────────────────

    def run_cycle(self):
        """前進 → 停車掃描 → 導航/避障。一個週期約 5~8 秒。"""

        if not self._active():
            self.motor.stop()
            return

        # ── 步驟 1：前進 ──
        hit_obstacle = self._drive_forward()

        if not self._active():
            self.motor.stop()
            return

        # ── 步驟 2：停車掃描（沒撞到障礙時才掃）──
        if not hit_obstacle:
            self._scan_sweep()

        if not self._active():
            self.motor.stop()
            return

        # ── 步驟 3：決定下一步方向 ──
        dist = self._get_distance()
        if dist < 0 or (0 < dist < config.SMART_PATROL_OBSTACLE):
            # 前方有障礙 → 避障
            self._avoid()
        else:
            self._consec_blocked = 0
            # 無障礙 → 用熱區地圖決定最佳探索方向
            self._navigate_by_map()

        # ── 步驟 4：卡住偵測 ──
        self._check_stuck()

    # ──────────────────────────────────────────────
    # 步驟 1：前進
    # ──────────────────────────────────────────────

    def _drive_forward(self) -> bool:
        """
        前進 MOVE_SEC 秒，途中每 0.1s 檢查超聲波。
        回傳 True = 被障礙提前打斷。
        """
        logger.info("[巡邏] 前進中...")
        self.motor.move(0, config.SMART_PATROL_SPEED, 0)

        drive_start = time.time()
        hit_obstacle = False
        no_echo_count = 0
        steps = int(config.SMART_PATROL_MOVE_SEC / 0.1)
        for _ in range(steps):
            if not self._active():
                break
            time.sleep(0.1)
            dist = self._get_distance()

            if dist < 0:
                no_echo_count += 1
                if no_echo_count >= 3:
                    logger.info("[巡邏] 連續無回波，可能已貼牆，緊急停車")
                    self.motor.stop()
                    hit_obstacle = True
                    break
            elif dist < config.SMART_PATROL_OBSTACLE:
                logger.info(f"[巡邏] 前方障礙 {dist:.0f}cm，提前停車")
                self.motor.stop()
                hit_obstacle = True
                break
            else:
                no_echo_count = 0

        if not hit_obstacle:
            self.motor.stop()

        # 更新熱區地圖位置
        if self._heat_map:
            actual_duration = time.time() - drive_start
            self._heat_map.update_forward(config.SMART_PATROL_SPEED, actual_duration)

        return hit_obstacle

    # ──────────────────────────────────────────────
    # 步驟 2：停車掃描
    # ──────────────────────────────────────────────

    def _scan_sweep(self):
        """舵機左→右→歸中，一個完整來回。掃描間隔內持續檢查分數以快速鎖定。"""
        logger.info("[巡邏] 掃描中...")
        pan = config.PATROL_PAN_MIN
        step = abs(config.PATROL_PAN_STEP)

        def _wait_and_check(duration):
            """等待期間每 50ms 檢查一次分數，偵測到目標立即暫停"""
            checks = max(1, int(duration / 0.05))
            for _ in range(checks):
                if self._get_score() >= config.VICTIM_SUSPECT_THRESHOLD:
                    return True
                if not self._active():
                    return True
                time.sleep(0.05)
            return False

        # 往右
        while pan <= config.PATROL_PAN_MAX:
            if not self._active():
                return
            if self._get_score() >= config.VICTIM_SUSPECT_THRESHOLD:
                time.sleep(0.1)
                continue
            self.servo.set_angle(pan, config.SERVO_DEFAULT_TILT)
            self._update_pan(pan)
            if _wait_and_check(config.PATROL_SWEEP_DELAY):
                continue
            pan += step

        # 往左
        pan = config.PATROL_PAN_MAX
        while pan >= config.PATROL_PAN_MIN:
            if not self._active():
                return
            if self._get_score() >= config.VICTIM_SUSPECT_THRESHOLD:
                time.sleep(0.1)
                continue
            self.servo.set_angle(pan, config.SERVO_DEFAULT_TILT)
            self._update_pan(pan)
            if _wait_and_check(config.PATROL_SWEEP_DELAY):
                continue
            pan -= step

        # 歸中
        self.servo.set_angle(0, config.SERVO_DEFAULT_TILT)
        self._update_pan(0)

        # 標記掃描區域到熱區地圖
        if self._heat_map:
            self._heat_map.mark_scan_fan(config.PATROL_PAN_MIN, config.PATROL_PAN_MAX)

    # ──────────────────────────────────────────────
    # 步驟 3a：熱區導航（無障礙時）
    # ──────────────────────────────────────────────

    def _navigate_by_map(self):
        """
        用熱區地圖找到未探索最多的方向，主動轉過去。
        每次掃描後都會執行（不再只等 is_ahead_fully_scanned）。
        """
        if not self._heat_map:
            return

        best_angle, unexplored_count = self._heat_map.get_best_exploration_angle()

        # 正前方（±20°）已經是最佳方向 → 不用轉
        if abs(best_angle) <= 20:
            return

        # 未探索格數太少 → 附近都探索過了，不需要刻意轉
        if unexplored_count < 3:
            return

        # 根據偏離角度決定轉彎時間（角度越大轉越久）
        turn_dir = 1 if best_angle > 0 else -1
        # 將角度映射到轉彎時間：45° ≈ 基礎轉彎時間，90° ≈ 2倍，180° ≈ 3倍
        angle_ratio = min(abs(best_angle) / 60.0, 3.0)
        turn_duration = config.SMART_PATROL_TURN_SEC * angle_ratio

        dir_name = "右" if turn_dir > 0 else "左"
        logger.info(f"[巡邏] 熱區導航：轉{dir_name} {abs(best_angle)}° 探索未知區域")
        self._log("info", f"探索導航：轉{dir_name} {abs(best_angle)}°")

        self.motor.move(0, 0, config.PATROL_TURN_SPEED * turn_dir)
        time.sleep(turn_duration)
        self.motor.stop()

        if self._heat_map:
            self._heat_map.update_turn(
                config.PATROL_TURN_SPEED, turn_duration, turn_dir)

    # ──────────────────────────────────────────────
    # 步驟 3b：避障
    # ──────────────────────────────────────────────

    def _avoid(self):
        """後退 + 轉彎。連續被堵就升級策略。避障永遠交替方向防止卡角落。"""
        self._consec_blocked += 1
        self.motor.stop()

        # 在地圖上標記障礙物位置
        if self._heat_map:
            self._heat_map.mark_obstacle()

        turn_dir = self._turn_dir

        if self._consec_blocked >= 5:
            # 連續 5 次堵死 → 麥克納姆側移脫困
            logger.info("[巡邏] 連續被堵 5 次 → 側移脫困")
            self._log("warn", "側移脫困")
            self.motor.move(config.STRAFE_SPEED * turn_dir, 0, 0)
            time.sleep(config.STRAFE_DURATION)
            self.motor.stop()
            if self._heat_map:
                self._heat_map.update_strafe(
                    config.STRAFE_SPEED, config.STRAFE_DURATION, turn_dir)
            self._turn_dir *= -1
            self._consec_blocked = 0

        elif self._consec_blocked >= 3:
            # 連續 3 次堵 → 大角度後退+轉彎
            logger.info("[巡邏] 連續被堵 3 次 → 大角度後退轉彎")
            self._log("warn", "大角度轉彎脫困")
            self.motor.move(0, -config.PATROL_REVERSE_SPEED, 0)
            time.sleep(config.SMART_PATROL_REVERSE_SEC * 1.5)
            self.motor.stop()
            if self._heat_map:
                self._heat_map.update_reverse(
                    config.PATROL_REVERSE_SPEED, config.SMART_PATROL_REVERSE_SEC * 1.5)
            self.motor.move(0, 0, config.PATROL_TURN_SPEED * turn_dir)
            time.sleep(config.SMART_PATROL_TURN_SEC * 2)
            self.motor.stop()
            if self._heat_map:
                self._heat_map.update_turn(
                    config.PATROL_TURN_SPEED, config.SMART_PATROL_TURN_SEC * 2, turn_dir)
            self._turn_dir *= -1

        else:
            # 一般避障 → 後退 + 轉彎
            logger.info(f"[巡邏] 避障：後退+轉{'左' if turn_dir < 0 else '右'}")
            self._log("info", f"避障轉{'左' if turn_dir < 0 else '右'}")
            self.motor.move(0, -config.PATROL_REVERSE_SPEED, 0)
            time.sleep(config.SMART_PATROL_REVERSE_SEC)
            self.motor.stop()
            if self._heat_map:
                self._heat_map.update_reverse(
                    config.PATROL_REVERSE_SPEED, config.SMART_PATROL_REVERSE_SEC)
            self.motor.move(0, 0, config.PATROL_TURN_SPEED * turn_dir)
            time.sleep(config.SMART_PATROL_TURN_SEC)
            self.motor.stop()
            if self._heat_map:
                self._heat_map.update_turn(
                    config.PATROL_TURN_SPEED, config.SMART_PATROL_TURN_SEC, turn_dir)
            self._turn_dir *= -1

    # ──────────────────────────────────────────────
    # 步驟 4：卡住偵測
    # ──────────────────────────────────────────────

    def _check_stuck(self):
        """
        如果連續 4 個週期都在同一個格位 → 機器人卡住了。
        強制大角度轉彎（用熱區地圖找最佳方向）脫困。
        """
        if not self._heat_map:
            return

        current_pos = self._heat_map.get_grid_position()

        if current_pos == self._last_grid_pos:
            self._same_pos_count += 1
        else:
            self._same_pos_count = 0
            self._last_grid_pos = current_pos

        if self._same_pos_count >= 4:
            logger.warning(f"[巡邏] 卡住偵測：同一位置 {self._same_pos_count} 個週期！強制脫困")
            self._log("warn", "卡住偵測：強制轉向脫困")

            # 用熱區地圖找最佳探索方向
            best_angle, _ = self._heat_map.get_best_exploration_angle()
            if abs(best_angle) < 45:
                best_angle = 120 * self._turn_dir  # 至少轉 120°

            turn_dir = 1 if best_angle > 0 else -1
            turn_duration = config.SMART_PATROL_TURN_SEC * 3  # 大角度轉

            # 先後退
            self.motor.move(0, -config.PATROL_REVERSE_SPEED, 0)
            time.sleep(config.SMART_PATROL_REVERSE_SEC * 2)
            self.motor.stop()
            if self._heat_map:
                self._heat_map.update_reverse(
                    config.PATROL_REVERSE_SPEED, config.SMART_PATROL_REVERSE_SEC * 2)

            # 再大角度轉彎
            self.motor.move(0, 0, config.PATROL_TURN_SPEED * turn_dir)
            time.sleep(turn_duration)
            self.motor.stop()
            if self._heat_map:
                self._heat_map.update_turn(
                    config.PATROL_TURN_SPEED, turn_duration, turn_dir)

            self._same_pos_count = 0
            self._consec_blocked = 0
            self._turn_dir *= -1
