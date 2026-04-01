"""
搜救機器人 — 任務狀態機控制器
===============================
7 階段完整狀態機：
  STANDBY → SEARCH → ANOMALY → LOCK_ON → INQUIRY → CONFIRM → REPORT
  任何階段皆可切回 MANUAL（手動）
"""

import time
import logging
import threading
from typing import Optional

import config
from fusion import VictimFusion, FusionInput, FusionResult
from event_logger import EventLogger

logger = logging.getLogger("rescue.mission")


class MissionController:
    """搜救任務狀態機"""

    STAGES = [
        "STANDBY", "SEARCH", "ANOMALY", "LOCK_ON",
        "INQUIRY", "CONFIRM", "REPORT", "BACKTRACK", "MANUAL"
    ]

    def __init__(self, motor, servo, speaker, fusion: VictimFusion,
                 event_logger: EventLogger, hri_module=None):
        self.motor = motor
        self.servo = servo
        self.speaker = speaker
        self.fusion = fusion
        self.event_logger = event_logger
        self.hri = hri_module  # 可選，None 時跳過 INQUIRY

        self.stage = "STANDBY"
        self._lock = threading.Lock()
        self._latest_fusion: Optional[FusionResult] = None
        self._anomaly_start_time = 0.0
        self._anomaly_confirm_sec = 2.0  # 異常持續 2 秒才升級
        self._report_cooldown_until = 0.0
        self._inquiry_result = None
        self._inquiry_thread = None
        self._backtrack_requested = False
        self._lock_on_start_time = 0.0

        logger.info("✅ 任務狀態機初始化 | 階段: STANDBY")

    @property
    def current_stage(self) -> str:
        with self._lock:
            return self.stage

    @property
    def latest_fusion(self) -> Optional[FusionResult]:
        with self._lock:
            return self._latest_fusion

    def transition_to(self, new_stage: str):
        """安全地切換狀態"""
        if new_stage not in self.STAGES:
            return
        with self._lock:
            old = self.stage
            if old == new_stage:
                return
            self.stage = new_stage
        logger.info(f"🔄 狀態轉換: {old} → {new_stage}")

    def force_manual(self):
        """強制切回手動"""
        self.motor.stop()
        self.transition_to("MANUAL")

    def request_backtrack(self):
        """請求回溯（REPORT 完成後自動啟動）"""
        self._backtrack_requested = True
        logger.info("已排程路徑回溯")

    def start_mission(self):
        """啟動任務（從 STANDBY → SEARCH）"""
        self.transition_to("STANDBY")
        time.sleep(0.5)
        self.transition_to("SEARCH")

    # ------------------------------------------------------------------
    def update(self, person_count, pose_anomaly_score, audio_help_score,
               audio_knock, distance_cm, frame=None, wave_detected=False):
        """每幀更新狀態機"""
        if self.current_stage == "MANUAL":
            return

        # 若處於 Panic Mode 全局搜尋階段，只要有看到人就視為極高機率是受困者 (強制拉高 AI 姿勢權重以觸發 LOCK_ON)
        if time.time() < getattr(self, 'panic_mode_until', 0.0):
            if person_count > 0:
                pose_anomaly_score = max(pose_anomaly_score, 0.9)

        # 計算融合分數
        fusion_input = FusionInput(
            person_detected=(person_count > 0),
            person_count=person_count,
            pose_anomaly_score=pose_anomaly_score,
            audio_help_score=audio_help_score,
            audio_knock=audio_knock,
            motion_response=getattr(self._inquiry_result, 'motion_response_score', 0.0)
                           if self._inquiry_result else 0.0,
            distance_cm=distance_cm,
        )
        fusion_result = self.fusion.compute(fusion_input)

        with self._lock:
            self._latest_fusion = fusion_result
            stage = self.stage

        score = fusion_result.victim_score
        now = time.time()

        # --- 狀態機轉換邏輯 ---

        if stage == "STANDBY":
            # 等待 start_mission() 被呼叫
            pass

        elif stage == "SEARCH":
            if score >= config.VICTIM_SUSPECT_THRESHOLD:
                if self._anomaly_start_time == 0:
                    self._anomaly_start_time = now
                elif now - self._anomaly_start_time >= self._anomaly_confirm_sec:
                    self.transition_to("ANOMALY")
                    self.motor.stop()
            else:
                self._anomaly_start_time = 0

        elif stage == "ANOMALY":
            if score >= config.VICTIM_HIGH_RISK_THRESHOLD:
                self.transition_to("LOCK_ON")
                self._lock_on_start_time = now
                # 不在這裡馬上 stop motor，交由 app.py 讓他前進靠近
            elif score < config.VICTIM_SUSPECT_THRESHOLD:
                # 誤報回退
                self._anomaly_start_time = 0
                self.transition_to("SEARCH")

        elif stage == "LOCK_ON":
            # 等待靠近：距離 20~25cm 區間（安全距離）或超過 15 秒自動詢問
            time_in_lock_on = now - self._lock_on_start_time
            close_enough = (0 < distance_cm <= 30) or distance_cm < 0
            if close_enough or time_in_lock_on > 15.0:
                self.motor.stop()
                if self.hri and not self.hri.is_running:
                    self.transition_to("INQUIRY")
                    self._start_inquiry_async(frame)
                elif not self.hri:
                    self.transition_to("CONFIRM")

        elif stage == "INQUIRY":
            self.motor.stop()
            # 揮手偵測快捷通道：偵測到揮手 → 視為明確求助 → 直接確認通報
            if wave_detected:
                logger.warning("👋 [MISSION] 偵測到揮手求助！跳過語音詢問，立即確認通報")
                from hri_module import InquiryResult
                self._inquiry_result = InquiryResult(
                    completed=True,
                    critical_help_requested=True,
                    recognized_text="[揮手求助]",
                )
                self.transition_to("CONFIRM")
            # 等待 HRI 完成（非阻塞）
            elif self._inquiry_result is not None and self._inquiry_result.completed:
                self.transition_to("CONFIRM")
            elif self.hri and self._inquiry_thread and not self._inquiry_thread.is_alive() and self._inquiry_result is None:
                # HRI 線程已經死掉卻沒有生成 result，判定為啟動失敗，跳過
                self.transition_to("CONFIRM")

        elif stage == "CONFIRM":
            critical_help = getattr(self._inquiry_result, 'critical_help_requested', False)
            inquiry_done = self._inquiry_result is not None and getattr(self._inquiry_result, 'completed', False)

            if critical_help:
                # 明確求救（語音/揮手）
                logger.warning("🚨 [MISSION] 收到明確求救，強制啟動緊急通報！")
                self.transition_to("REPORT")
                self._do_report(frame, fusion_result, critical=True)
            elif inquiry_done and not critical_help and pose_anomaly_score >= 0.5:
                # 完全無回應 + 仍然倒地 → 失去意識，強制通報
                logger.warning("🚨 [MISSION] 傷患無回應且持續倒地 → 判定失去意識，強制通報！")
                self.transition_to("REPORT")
                self._do_report(frame, fusion_result, critical=True)
            elif score >= config.VICTIM_HIGH_RISK_THRESHOLD:
                self.transition_to("REPORT")
                self._do_report(frame, fusion_result, critical=False)
            elif score >= config.VICTIM_SUSPECT_THRESHOLD:
                pass
            else:
                self._inquiry_result = None
                self.transition_to("SEARCH")

        elif stage == "REPORT":
            self.motor.stop()
            self.servo.home()
            if now > self._report_cooldown_until:
                self._inquiry_result = None
                if self._backtrack_requested:
                    self._backtrack_requested = False
                    logger.info("通報完成，啟動路徑回溯")
                    self.transition_to("BACKTRACK")
                else:
                    logger.info("通報完成，回到待命")
                    self.transition_to("STANDBY")

        return fusion_result

    # ------------------------------------------------------------------
    # 內部方法
    # ------------------------------------------------------------------

    def _start_inquiry_async(self, frame):
        """在背景執行緒中執行 HRI 主動詢問"""
        self._inquiry_result = None

        def _run():
            try:
                result = self.hri.run_inquiry_sequence()
                self._inquiry_result = result
            except Exception as e:
                logger.error(f"HRI 互動錯誤: {e}")
                from hri_module import InquiryResult
                self._inquiry_result = InquiryResult(completed=True)

        self._inquiry_thread = threading.Thread(target=_run, daemon=True)
        self._inquiry_thread.start()

    def _do_report(self, frame, fusion_result: FusionResult, critical: bool = False):
        """執行事件回報"""
        self._report_cooldown_until = time.time() + config.ALERT_COOLDOWN

        # 記錄事件
        audio_desc = ""
        if self._inquiry_result:
            parts = []
            if self._inquiry_result.critical_help_requested:
                parts.append("✅辨識出求助關鍵字")
            if self._inquiry_result.voice_detected:
                parts.append("有語音")
            if self._inquiry_result.knock_detected:
                parts.append("有敲擊")
            audio_desc = "+".join(parts) if parts else "無回應"
            
            if hasattr(self._inquiry_result, 'recognized_text') and self._inquiry_result.recognized_text:
                audio_desc += f" (內容: {self._inquiry_result.recognized_text})"

        self.event_logger.log_event(
            frame=frame,
            victim_score=fusion_result.victim_score,
            risk_level=fusion_result.risk_level,
            mission_stage="REPORT",
            person_count=0,  # 由外部在 app_state 中補充
            fallen_count=0,
            audio_event=audio_desc,
            components=fusion_result.components,
        )
