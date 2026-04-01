"""
搜救機器人主程式 (V2)
====================
Flask 伺服器入口，整合所有硬體與 AI 模組。
7 階段任務狀態機 + 多模態融合 + 音訊偵測 + 主動語音互動
"""

import threading
import time
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
import cv2

import config
from camera import Camera
from motor import MotorController
from servo import ServoController
from ultrasonic import UltrasonicSensor
from detector import RescueDetector
from notifier import AlertSystem
from speaker import Speaker
from audio_reader import AudioReader
from audio_detector import AudioDetector, AudioResult
from fusion import VictimFusion, FusionInput
from event_logger import EventLogger
from hri_module import HRIModule
from mission_controller import MissionController

# 新功能模組
try:
    from heat_map import HeatMap
    HEAT_MAP_AVAILABLE = True
except ImportError:
    HEAT_MAP_AVAILABLE = False

try:
    from intercom import Intercom
    INTERCOM_AVAILABLE = True
except ImportError:
    INTERCOM_AVAILABLE = False

try:
    from backtrack import BacktrackEngine
    BACKTRACK_AVAILABLE = True
except ImportError:
    BACKTRACK_AVAILABLE = False

# ============================================================
# 日誌設定
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("rescue")

# ============================================================
# Flask
# ============================================================
app = Flask(__name__)

# 背景工作佇列（確保 Flask 請求立即返回）
_executor = ThreadPoolExecutor(max_workers=3)

def _bg(fn):
    """將耗時操作推入背景執行緒池"""
    _executor.submit(fn)

# ============================================================
# 初始化所有模組
# ============================================================
logger.info("=" * 60)
logger.info("  搜救機器人系統 V2 啟動中...")
logger.info("=" * 60)

# 硬體
camera = Camera()
motor = MotorController()
servo = ServoController()
ultrasonic = UltrasonicSensor()
speaker = Speaker()

# AI
detector = RescueDetector()
audio_reader = AudioReader()
audio_detector = AudioDetector()

# 融合與決策
fusion = VictimFusion()
event_logger = EventLogger()
hri = HRIModule(speaker, audio_reader, audio_detector)
mission = MissionController(motor, servo, speaker, fusion, event_logger, hri)

# 通知
alert_manager = AlertSystem()

# 熱區記憶地圖
heat_map = HeatMap() if (HEAT_MAP_AVAILABLE and config.HEAT_MAP_ENABLED) else None

# 雙向語音對講
intercom = Intercom(audio_reader) if INTERCOM_AVAILABLE else None

# 路徑回溯（在 search_loop 中完成初始化）
bt_engine = None

# ============================================================
# 全域狀態
# ============================================================
state_lock = threading.Lock()
app_state = {
    "mode": "manual",
    "mission_stage": "STANDBY",
    "person_count": 0,
    "fallen_count": 0,
    "pose_anomaly_score": 0.0,
    "victim_score": 0.0,
    "risk_level": "LOW",
    "distance_cm": -1,
    "audio_event": {
        "has_voice": False,
        "help_score": 0.0,
        "knock_detected": False,
        "rms_level": 0.0,
    },
    "fusion_components": {},
    "search_mode": config.SEARCH_MODE,
    "alert_count": 0,
    "event_count": 0,
    "fps": 0,
    "detect_ms": 0,
    "brightness": 0, # Added brightness state
    "wave_detected": False,
    "night_vision": config.CLAHE_MODE,      # "auto" / "on" / "off"（預設 off）
    "brightness_avg": 128,                  # 幀平均亮度
    "unique_person_count": 0,               # 歷史不重複人數
    "unreported_count": 0,                  # 未通報可見人數
    "heat_map_coverage": 0.0,               # 熱區覆蓋率 %
    "objects": [],                          # 偵測到的災區物件
    "ai_loaded": False,
    "mic_ok": audio_reader.is_available,
    "camera_ok": camera.is_opened(),
    "gpio_ok": config.GPIO_AVAILABLE,
    "patrol_pan": 0,
    "logs": [],
    "events": [],
}

_latest_frame = None
_frame_lock = threading.Lock()
MAX_LOGS = 50


def add_log(level, msg):
    entry = {"time": time.strftime("%H:%M:%S"), "level": level, "msg": msg}
    with state_lock:
        app_state["logs"].insert(0, entry)
        if len(app_state["logs"]) > MAX_LOGS:
            app_state["logs"] = app_state["logs"][:MAX_LOGS]


def get_state():
    with state_lock:
        s = dict(app_state)
        s["logs"] = list(app_state["logs"])
        s["events"] = list(app_state["events"])
        s["audio_event"] = dict(app_state["audio_event"])
        s["fusion_components"] = dict(app_state.get("fusion_components", {}))
        return s


# ============================================================
# 背景執行緒：AI 偵測
# ============================================================
def detection_loop():
    global _latest_frame
    logger.info("AI 偵測迴圈啟動")
    add_log("info", "AI 偵測迴圈啟動")

    frame_count = 0
    fps_timer = time.time()
    _report_alert_sent = False   # 防止 REPORT 期間重複觸發警報
    _fallen_alert_sent = False   # 倒地即時警報
    _fallen_first_seen = 0.0    # 首次偵測到倒地的時間（持續 3 秒才發警報）
    FALLEN_ALERT_DELAY = 3.0    # 倒地持續秒數門檻

    # CLAHE（僅在使用者手動開啟時才建立）
    _clahe = None
    _cached_apply_clahe = False
    _cached_avg_br = 128
    _hm_cov = 0.0

    while True:
        try:
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.3)
                continue

            # 一次性讀取所有需要的狀態（減少鎖競爭）
            with state_lock:
                audio_ev = dict(app_state["audio_event"])
                mode = app_state["mode"]
                distance = app_state["distance_cm"]
                brightness = app_state["brightness"]
                night_vision = app_state.get("night_vision", "auto")

            # ── REPORT / STANDBY 期間跳過 AI 推論，釋放 CPU 給警報 TTS ──
            _cur_mission = mission.current_stage
            if _cur_mission in ("REPORT", "STANDBY", "BACKTRACK") and mode == "auto":
                camera.set_display_frame(frame)
                with state_lock:
                    app_state["mission_stage"] = _cur_mission
                time.sleep(0.033)   # ~30fps 原始畫面直通
                continue

            # ── 低光增強（僅在使用者開啟時執行，預設完全跳過）──
            if night_vision != "off":
                if _clahe is None:
                    _clahe = cv2.createCLAHE(
                        clipLimit=config.CLAHE_CLIP_LIMIT,
                        tileGridSize=config.CLAHE_TILE_SIZE)
                if frame_count % 30 == 0:
                    _small = cv2.resize(frame, (80, 60))
                    _cached_avg_br = int(cv2.cvtColor(_small, cv2.COLOR_BGR2GRAY).mean())
                    _cached_apply_clahe = (night_vision == "on") or \
                        (night_vision == "auto" and _cached_avg_br < config.CLAHE_AUTO_THRESHOLD)
                if _cached_apply_clahe:
                    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                    lab[:, :, 0] = _clahe.apply(lab[:, :, 0])
                    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            # 套用亮度調整
            if brightness != 0:
                frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=brightness)

            frame_count += 1

            t0 = time.time()
            result = detector.detect(frame)
            detect_ms = int((time.time() - t0) * 1000)

            with _frame_lock:
                _latest_frame = result.annotated_frame
            camera.set_display_frame(result.annotated_frame)
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                fps = round(frame_count / elapsed, 1)
                frame_count = 0
                fps_timer = time.time()
            else:
                fps = app_state.get("fps", 0)

            # 融合計算（包裹在獨立 try 中，不影響影像串流）
            score = 0.0
            risk = "LOW"
            components = {}
            try:
                if mode == "auto":
                    with _frame_lock:
                        current_frame = _latest_frame
                    fusion_result = mission.update(
                        person_count=result.person_count,
                        pose_anomaly_score=result.pose_anomaly_score,
                        audio_help_score=audio_ev.get("help_score", 0),
                        audio_knock=audio_ev.get("knock_detected", False),
                        distance_cm=distance,
                        frame=current_frame,
                        wave_detected=result.wave_detected,
                    )
                    score = fusion_result.victim_score
                    risk = fusion_result.risk_level
                    components = fusion_result.components
                else:
                    from fusion import FusionInput as _FI
                    _inp = _FI(
                        person_detected=(result.person_count > 0),
                        person_count=result.person_count,
                        pose_anomaly_score=result.pose_anomaly_score,
                        audio_help_score=audio_ev.get("help_score", 0),
                        audio_knock=audio_ev.get("knock_detected", False),
                        distance_cm=distance,
                        heart_rate_bpm=result.heart_rate_bpm,
                        rppg_confidence=result.rppg_confidence,
                    )
                    _fr = fusion.compute(_inp)
                    score = _fr.victim_score
                    risk = _fr.risk_level
                    components = _fr.components
            except Exception as e:
                logger.debug(f"融合計算錯誤: {e}")

            # 在鎖外預先計算（降頻避免鎖競爭）
            if heat_map and frame_count % 30 == 0:
                _hm_cov = heat_map.get_coverage_percent()

            # 更新全域狀態（快進快出）
            with state_lock:
                app_state["person_count"] = result.person_count
                app_state["fallen_count"] = result.fallen_count
                app_state["pose_anomaly_score"] = result.pose_anomaly_score
                app_state["wave_detected"] = result.wave_detected
                app_state["victim_score"] = score
                app_state["risk_level"] = risk
                app_state["fusion_components"] = components
                app_state["mission_stage"] = mission.current_stage
                app_state["alert_count"] = alert_manager.alert_count
                app_state["event_count"] = event_logger.event_count
                app_state["fps"] = fps
                app_state["detect_ms"] = detect_ms
                app_state["brightness_avg"] = _cached_avg_br
                # 新功能狀態
                app_state["unique_person_count"] = result.unique_person_count
                app_state["unreported_count"] = result.unreported_count
                app_state["objects"] = result.objects
                app_state["eye_state"] = result.eye_state
                app_state["heart_rate_bpm"] = result.heart_rate_bpm
                app_state["rppg_confidence"] = result.rppg_confidence
                app_state["rppg_signal_quality"] = result.rppg_signal_quality
                app_state["heat_map_coverage"] = _hm_cov

            # 事件列表更新（降頻：每 10 幀一次，減少鎖競爭）
            if frame_count % 10 == 0 or frame_count == 0:
                with state_lock:
                    app_state["events"] = event_logger.get_events()[:10]

            # ── 倒地警報（持續 3 秒才觸發，避免瞬間誤判）──
            try:
                if result.fallen_count > 0:
                    now_t = time.time()
                    if _fallen_first_seen == 0.0:
                        _fallen_first_seen = now_t  # 記錄首次偵測時間

                    fallen_duration = now_t - _fallen_first_seen

                    if fallen_duration >= FALLEN_ALERT_DELAY and not _fallen_alert_sent:
                        # 持續 3 秒確認 → 發送警報
                        with _frame_lock:
                            alert_frame = _latest_frame
                        eye_info = f" | 眼睛:{result.eye_state}" if result.eye_state != "UNKNOWN" else ""
                        hr_info = f" | 心率:{result.heart_rate_bpm:.0f}bpm" if result.heart_rate_bpm > 0 else ""
                        msg = (f"搜救系統偵測到人員倒地！(持續{fallen_duration:.0f}秒)\n"
                               f"人數:{result.person_count} 倒地:{result.fallen_count}\n"
                               f"VictimScore: {score:.2f} ({risk}){eye_info}{hr_info}")
                        if alert_manager.trigger_alert(
                            frame=alert_frame,
                            message=msg,
                            critical=True,
                        ):
                            _fallen_alert_sent = True
                            add_log("danger", f"倒地持續 {fallen_duration:.0f}s，已發送警報 (Score:{score:.2f})")
                            event_logger.log_event(
                                frame=alert_frame, victim_score=score, risk_level=risk,
                                mission_stage=mission.current_stage,
                                person_count=result.person_count,
                                fallen_count=result.fallen_count,
                                components=components)
                            with state_lock:
                                app_state["events"] = event_logger.get_events()[:10]
                                app_state["event_count"] = event_logger.event_count
                            if detector.tracker:
                                detector.tracker.mark_all_visible_reported()
                            if heat_map:
                                try:
                                    heat_map.mark_person("rescued")
                                except Exception:
                                    pass
                else:
                    # 倒地消失（站起來了）→ 重置計時器 + 中斷正在播放的警報
                    if _fallen_alert_sent and alert_manager.is_broadcasting:
                        alert_manager.cancel_alert()
                        add_log("info", "人員恢復正常，警報已中斷")
                    _fallen_first_seen = 0.0
                    _fallen_alert_sent = False
            except Exception:
                pass

            # Telegram 通知（REPORT 階段補充通報，與上方倒地即時通報互補）
            try:
                _cur_stage = mission.current_stage
                if _cur_stage == "REPORT" and not _report_alert_sent:
                    with _frame_lock:
                        alert_frame = _latest_frame
                    if alert_manager.trigger_alert(
                        frame=alert_frame,
                        message=f"搜救系統通報：完成搜救流程\nVictimScore: {score:.2f} ({risk})",
                        critical=True,
                    ):
                        _report_alert_sent = True
                        if detector.tracker:
                            detector.tracker.mark_all_visible_reported()
                elif _cur_stage != "REPORT":
                    _report_alert_sent = False
            except Exception:
                pass

            time.sleep(0.003)

        except Exception as e:
            logger.error(f"偵測迴圈錯誤: {e}")
            # 極短等待後繼續，避免畫面長時間凍結
            try:
                f = camera.get_frame()
                if f is not None:
                    camera.set_display_frame(f)
            except Exception:
                pass
            time.sleep(0.1)


# ============================================================
# 背景執行緒：音訊偵測
# ============================================================
def audio_loop():
    logger.info("音訊偵測迴圈啟動")
    add_log("info", "音訊偵測迴圈啟動")

    import time
    try:
        import speech_recognition as sr
        import numpy as np
        STT_AVAILABLE = True
    except ImportError:
        logger.warning("speech_recognition 套件未找到，停用全局 STT")
        STT_AVAILABLE = False

    last_stt_time = 0.0  # 限流計時器（取代脆弱的函數屬性）

    while True:
        try:
            if not audio_reader.is_available:
                time.sleep(2)
                continue

            # 防自我觸發：警報播放中或剛結束時暫停收音
            # （避免麥克風聽到自己的 TTS/警報聲 → STT 辨識出「救援」→ 無限迴圈崩潰）
            _audio_stage = mission.current_stage
            if _audio_stage in ("REPORT", "STANDBY"):
                time.sleep(2)
                continue
            # 警報系統正在播放（即使已切離 REPORT 階段）
            if alert_manager.is_broadcasting:
                time.sleep(2)
                continue
            # REPORT 結束後冷卻 15 秒，等警報音完全播完+餘音消散
            if hasattr(alert_manager, 'last_alert_time') and \
               time.time() - alert_manager.last_alert_time < 15:
                time.sleep(1)
                continue

            audio_buffer = audio_reader.get_audio_buffer(config.MIC_BUFFER_SEC)
            audio_result = audio_detector.detect_buffer(
                audio_buffer,
                window_size=config.MIC_CHUNK_SIZE
            )

            critical_help_detected = False

            # 全局收音防護網：只要測到語音，就送交 STT 找求救關鍵字 (限流每 3 秒一次避免 API 撐爆)
            if STT_AVAILABLE and audio_result.has_voice:
                current_time = time.time()
                if current_time - last_stt_time > 3.0:
                    last_stt_time = current_time
                    try:
                        audio_16bit = (audio_buffer * 32767).astype(np.int16)
                        audio_data = sr.AudioData(audio_16bit.tobytes(), sample_rate=config.MIC_SAMPLE_RATE, sample_width=2)
                        
                        recognizer = sr.Recognizer()
                        text = recognizer.recognize_google(audio_data, language="zh-TW")
                        recognized_text = text.lower()
                        logger.info(f"[全局收音] 背後監聽文字: '{recognized_text}'")
                        
                        keywords = ["救", "幫", "需要", "help", "sos", "please", "痛", "受傷",
                                    "救命", "幫忙", "有人嗎", "來人", "快來", "危險",
                                    "hurt", "emergency", "danger", "嗚", "啊", "哎"]
                        if any(kw in recognized_text for kw in keywords):
                            logger.warning("🚨 [全局收音] 偵測到明確求救關鍵字，強行介入通報系統！")
                            critical_help_detected = True
                    except sr.UnknownValueError:
                        pass
                    except Exception as e:
                        logger.error(f"STT 錯誤: {e}") # Added error logging for STT

            # 將結果更新到 global 事件狀態
            with state_lock:
                app_state["audio_event"] = {
                    "has_voice": bool(audio_result.has_voice),
                    "help_score": float(audio_result.help_score),
                    "knock_detected": bool(audio_result.knock_detected),
                    "rms_level": float(audio_result.rms_level)
                }
                
                # 如果背景全局聽到了救命，記錄 Panic Mode，並進行 360 度搜尋
                if critical_help_detected:
                    mission.panic_mode_until = time.time() + 60.0  # 持續警戒 60 秒
                    if app_state["mode"] != "auto":
                        app_state["mode"] = "auto"
                    
                    if mission.current_stage in ["STANDBY", "SEARCH"]:
                        mission.transition_to("SEARCH")
                        add_log("warn", "🚨 [全局求救] 啟動 360 度全景尋人視角與強制鎖定！")

            # Original logic for logging help/knock detection (kept for consistency)
            if audio_result.help_score > config.HELP_THRESHOLD:
                add_log("warn", f"🗣️ 偵測到呼救聲 (score: {audio_result.help_score:.2f})")
            if audio_result.knock_detected:
                add_log("warn", "🔨 偵測到敲擊聲！")

            time.sleep(0.5) # Changed from 0.05 to 0.5 as per new code

        except Exception as e:
            logger.error(f"音訊迴圈錯誤: {e}")
            time.sleep(1)


# ============================================================
# 背景執行緒：超聲波
# ============================================================
def ultrasonic_loop():
    """
    超聲波測距 + 即時安全煞車。
    自動巡邏中任何可能移動的階段（SEARCH / ANOMALY / LOCK_ON），
    ≤ 20cm 立刻停車 + 後退一小步 + 轉彎避開。
    """
    _escape_dir = 1  # 交替左右轉避免卡死

    while True:
        try:
            dist = ultrasonic.get_distance_cm()
            with state_lock:
                app_state["distance_cm"] = dist
                cur_mode = app_state["mode"]

            cur_stage = mission.current_stage
            moving_stages = ("SEARCH", "ANOMALY", "LOCK_ON", "BACKTRACK")

            # 安全煞車：≤20cm 或 連續無回波（-1 = 可能已貼牆）
            too_close = (0 < dist <= 20)
            if dist < 0:
                _no_echo_count = getattr(ultrasonic_loop, '_no_echo_count', 0) + 1
                ultrasonic_loop._no_echo_count = _no_echo_count
                if _no_echo_count >= 3:
                    too_close = True
            else:
                ultrasonic_loop._no_echo_count = 0

            # 掃描旋轉期間跳過安全煞車（避免搶馬達控制權導致快速旋轉）
            from scan_patrol import scanning_lock
            if scanning_lock.is_set():
                time.sleep(0.25)
                continue

            if (cur_mode == "auto"
                    and cur_stage in moving_stages
                    and too_close):
                motor.stop()
                add_log("warn", f"⚠️ 安全煞車 {dist}cm → 後退+轉彎")
                # 後退
                motor.move(0, -config.PATROL_REVERSE_SPEED, 0)
                time.sleep(0.4)
                motor.stop()
                # 轉彎避開
                motor.move(0, 0, config.PATROL_TURN_SPEED * _escape_dir)
                time.sleep(0.4)
                motor.stop()
                _escape_dir *= -1

            time.sleep(0.25)
        except Exception:
            time.sleep(1)


# ============================================================
# 背景執行緒：搜索巡檢 (V4 — D 靜止 + E 智慧巡邏)
# ============================================================
def search_loop():
    global bt_engine
    add_log("info", "搜索巡檢執行緒啟動")
    from smart_patrol import SmartPatrol
    from scan_patrol import ScanPatrol

    _expected_sm = [None]  # 用 list 讓閉包可修改

    def _check_active():
        with state_lock:
            m = app_state["mode"]
            sm = app_state.get("search_mode", "E")
        # 模式或搜索策略改變 → 立即中斷當前週期
        if _expected_sm[0] is not None and sm != _expected_sm[0]:
            return False
        return m == "auto" and mission.current_stage == "SEARCH"

    def _get_search_mode():
        with state_lock:
            return app_state.get("search_mode", "E")

    def _get_distance():
        with state_lock:
            return app_state["distance_cm"]

    def _get_victim_score():
        with state_lock:
            return app_state.get("victim_score", 0.0)

    def _update_pan(pan):
        with state_lock:
            app_state["patrol_pan"] = pan

    # ── D 模式用的阻塞式掃描（保留） ──
    def _scan_sweep():
        pan = config.PATROL_PAN_MIN
        step = abs(config.PATROL_PAN_STEP)
        while pan <= config.PATROL_PAN_MAX:
            if not _check_active():
                return
            with state_lock:
                v_score = app_state.get("victim_score", 0.0)
            if v_score >= config.VICTIM_SUSPECT_THRESHOLD:
                time.sleep(0.1)
                continue
            servo.set_angle(pan, config.SERVO_DEFAULT_TILT)
            _update_pan(pan)
            time.sleep(config.PATROL_SWEEP_DELAY)
            pan += step
        pan = config.PATROL_PAN_MAX
        while pan >= config.PATROL_PAN_MIN:
            if not _check_active():
                return
            with state_lock:
                v_score = app_state.get("victim_score", 0.0)
            if v_score >= config.VICTIM_SUSPECT_THRESHOLD:
                time.sleep(0.1)
                continue
            servo.set_angle(pan, config.SERVO_DEFAULT_TILT)
            _update_pan(pan)
            time.sleep(config.PATROL_SWEEP_DELAY)
            pan -= step
        if _check_active():
            servo.set_angle(0, config.SERVO_DEFAULT_TILT)
            _update_pan(0)

    # ── 智慧巡邏實例（含熱區地圖）──
    smart = SmartPatrol(
        motor=motor, servo=servo,
        get_distance_fn=_get_distance,
        get_victim_score_fn=_get_victim_score,
        check_active_fn=_check_active,
        add_log_fn=add_log,
        update_pan_fn=_update_pan,
        heat_map=heat_map,
    )

    # ── 路徑回溯引擎 ──
    def _check_backtrack_cancel():
        with state_lock:
            m = app_state["mode"]
        return m != "auto" or mission.current_stage not in ("BACKTRACK",)

    if BACKTRACK_AVAILABLE:
        bt_engine = BacktrackEngine(
            motor=motor,
            get_distance_fn=_get_distance,
            check_cancel_fn=_check_backtrack_cancel,
            add_log_fn=add_log,
        )
    else:
        bt_engine = None

    # ── 掃描式巡邏實例（模式 F：超聲波旋轉掃描 + 回溯記錄）──
    scan = ScanPatrol(
        motor=motor, servo=servo,
        get_distance_fn=_get_distance,
        get_victim_score_fn=_get_victim_score,
        check_active_fn=_check_active,
        add_log_fn=add_log,
        update_pan_fn=_update_pan,
        heat_map=heat_map,
        backtrack_engine=bt_engine,
    )

    _was_auto = False
    _prev_mode = None

    while True:
        try:
            with state_lock:
                mode = app_state["mode"]
                distance = app_state["distance_cm"]

            if mode != "auto":
                if _was_auto:
                    motor.stop()
                    _was_auto = False
                time.sleep(0.5)
                continue
            _was_auto = True

            current_stage = mission.current_stage

            # ── BACKTRACK：路徑回溯 ──
            if current_stage == "BACKTRACK":
                if bt_engine:
                    bt_engine.execute_backtrack()
                mission.transition_to("STANDBY")
                with state_lock:
                    app_state["mode"] = "manual"
                motor.stop()
                add_log("info", "回溯完成，切回手動待命")
                continue

            # ── LOCK_ON：漸進靠近目標 ──
            if current_stage == "LOCK_ON":
                with state_lock:
                    pan = app_state.get("patrol_pan", 0)

                if abs(pan) > 15:
                    turn_speed = config.PATROL_TURN_SPEED if pan > 0 else -config.PATROL_TURN_SPEED
                    motor.move(0, 0, turn_speed)
                    new_pan = pan - (15 if pan > 0 else -15)
                    if abs(new_pan) < 15:
                        new_pan = 0
                    servo.set_angle(new_pan, config.SERVO_DEFAULT_TILT)
                    _update_pan(new_pan)
                    time.sleep(0.2)
                    continue

                if 0 < distance < 20:
                    motor.move(0, -config.PATROL_REVERSE_SPEED * 0.6, 0)
                elif distance <= 25:
                    motor.stop()
                elif distance <= 50:
                    motor.move(0, config.SMART_PATROL_SPEED * 0.25, 0)
                elif distance <= 100:
                    motor.move(0, config.SMART_PATROL_SPEED * 0.5, 0)
                else:
                    motor.move(0, config.SMART_PATROL_SPEED * 0.7, 0)

                time.sleep(0.2)
                continue

            # ── 非 SEARCH 階段 → 停車 ──
            if current_stage != "SEARCH":
                if current_stage in ("INQUIRY", "CONFIRM", "REPORT", "STANDBY"):
                    motor.stop()
                time.sleep(0.5)
                continue

            # ── SEARCH 階段 ──
            sm = _get_search_mode()

            # 模式切換時停車
            if sm != _prev_mode:
                motor.stop()
                _prev_mode = sm

            _expected_sm[0] = sm  # 記錄當前模式，切換時 _check_active 會立即回 False

            if sm == "D":
                # 靜止掃描
                motor.stop()
                _scan_sweep()
                time.sleep(0.2)
            elif sm == "F":
                # 掃描式巡邏：超聲波旋轉掃描 + 選方向 + 前進 + 攝影機掃描
                scan.run_cycle()
            else:
                # 智慧巡邏 (E)：一個週期 ≈ 前進3秒 + 掃描5秒 + 避障
                smart.run_cycle()

        except Exception as e:
            logger.warning(f"搜索迴圈錯誤: {e}")
            motor.stop()
            time.sleep(1)


# ============================================================
# Flask 路由
# ============================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/demo")
def demo_page():
    return render_template("demo.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        camera.generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/events/<path:filename>")
def serve_event_image(filename):
    return send_from_directory(config.EVENT_DIR, filename)


@app.route("/control", methods=["POST"])
def control():
    """所有操作均為非阻塞（耗時工作推入背景）"""
    data = request.json
    if not data:
        return jsonify({"status": "error"}), 400

    action = data.get("action", "")

    try:
        if action == "move":
            motor.move(float(data.get("vx", 0)), float(data.get("vy", 0)), float(data.get("omega", 0)))
        elif action == "servo":
            servo.set_angle(int(data.get("pan", 0)), int(data.get("tilt", 0)))
        elif action == "mode_auto":
            if bt_engine:
                bt_engine.clear()
            with state_lock:
                app_state["mode"] = "auto"
            _bg(lambda: mission.start_mission())
            add_log("info", "切換至自動搜索模式")
        elif action == "mode_manual":
            with state_lock:
                app_state["mode"] = "manual"
            mission.force_manual()
            add_log("info", "切換至手動模式")
        elif action == "stop":
            motor.stop()
            with state_lock:
                app_state["mode"] = "manual"
            mission.force_manual()
            add_log("warn", "緊急停止")
        elif action == "home":
            servo.home()
            add_log("info", "雲台歸位")
        elif action == "search_mode":
            m = data.get("search_mode", "E")
            if m in ("D", "E", "F"):
                with state_lock:
                    app_state["search_mode"] = m
                add_log("info", f"搜索模式: {'靜止掃描' if m == 'D' else '智慧巡邏'}")
        elif action == "set_brightness":
            val = int(data.get("value", 0))
            with state_lock:
                app_state["brightness"] = val
            add_log("info", f"設定亮度偏移: {val}")
        elif action == "toggle_night_vision":
            with state_lock:
                cur = app_state.get("night_vision", "auto")
                nxt = {"auto": "on", "on": "off", "off": "auto"}.get(cur, "auto")
                app_state["night_vision"] = nxt
            mode_name = {"auto": "自動", "on": "強制開啟", "off": "關閉"}
            add_log("info", f"夜視模式: {mode_name.get(nxt, nxt)}")
        elif action == "reset_heat_map":
            if heat_map:
                heat_map.reset()
            add_log("info", "熱區地圖已重置")
        elif action == "toggle_intercom":
            if intercom:
                if intercom.is_active:
                    intercom.stop()
                    add_log("info", "對講模式已關閉")
                else:
                    intercom.start()
                    add_log("info", "對講模式已開啟")
        elif action == "request_backtrack":
            stage = mission.current_stage
            if stage == "REPORT":
                mission.request_backtrack()
                add_log("info", "已排程路徑回溯（通報完成後自動啟動）")
            elif stage in ("STANDBY", "MANUAL") and bt_engine and bt_engine.stack_size > 0:
                with state_lock:
                    app_state["mode"] = "auto"
                mission.transition_to("BACKTRACK")
                add_log("info", f"啟動路徑回溯（{bt_engine.stack_size} 步）")
            else:
                add_log("warn", "目前無法回溯")
        elif action == "intercom_speak":
            text = data.get("text", "")
            if intercom and text:
                intercom.speak(text)
                add_log("info", f"[對講] {text[:30]}")

        # --- 手動測試（全部背景執行） ---
        elif action == "set_stage":
            stage = data.get("stage", "STANDBY")
            if stage in mission.STAGES:
                mission.transition_to(stage)
                with state_lock:
                    app_state["mission_stage"] = stage
                add_log("info", f"[手動] 狀態: {stage}")
        elif action == "test_inquiry":
            with state_lock:
                app_state["mode"] = "auto"
                mission.transition_to("INQUIRY")
            mission._start_inquiry_async(None)
            add_log("info", "[手動] 強制進入 INQUIRY 測試真實收音與通報流程")
        elif action == "test_report":
            add_log("info", "[手動] 事件回報中...")
            def _do_report():
                with state_lock:
                    sc = app_state["victim_score"]
                    rk = app_state["risk_level"]
                    pc = app_state["person_count"]
                    fc = app_state["fallen_count"]
                    cp = dict(app_state.get("fusion_components", {}))
                with _frame_lock:
                    report_frame = _latest_frame
                event_logger.log_event(
                    frame=report_frame, victim_score=sc, risk_level=rk,
                    mission_stage=mission.current_stage,
                    person_count=pc, fallen_count=fc, components=cp)
                with state_lock:
                    app_state["events"] = event_logger.get_events()[:10]
                    app_state["event_count"] = event_logger.event_count
                add_log("warn", f"[手動] 回報完成 | Score: {sc:.2f}")
                speaker.play_alert()
            _bg(_do_report)
        elif action == "simulate_audio":
            t = data.get("event_type", "voice")
            with state_lock:
                if t == "voice":
                    app_state["audio_event"] = {"has_voice": True, "help_score": 0.8, "knock_detected": False, "rms_level": 0.1}
                elif t == "knock":
                    app_state["audio_event"] = {"has_voice": False, "help_score": 0.0, "knock_detected": True, "rms_level": 0.15}
                elif t == "clear":
                    app_state["audio_event"] = {"has_voice": False, "help_score": 0.0, "knock_detected": False, "rms_level": 0.0}
            add_log("info", f"[模擬] 音訊: {t}")
        elif action == "trigger_critical_help":
            # 手動模擬收到關鍵字求救，並強制推入 CONFIRM 階段觸發事件
            from hri_module import InquiryResult
            mission._inquiry_result = InquiryResult(completed=True, critical_help_requested=True, recognized_text="救命 (Demo)")
            with state_lock:
                app_state["mode"] = "auto"
                mission.transition_to("CONFIRM")
            add_log("warn", "[模擬] 收到明確求救語音！強制進入確認並發報")
        elif action == "simulate_score":
            sc = float(data.get("score", 0))
            with state_lock:
                app_state["victim_score"] = sc
                app_state["risk_level"] = "HIGH" if sc >= 0.6 else ("SUSPECT" if sc >= 0.3 else "LOW")
                app_state["fusion_components"] = {
                    "person": min(sc * 1.5, 1.0), "pose": min(sc * 1.2, 1.0),
                    "audio": min(sc * 0.8, 1.0), "motion": min(sc * 0.5, 1.0), "distance": 0.5}
            add_log("info", f"[模擬] Score = {sc:.2f}")
    except Exception as e:
        logger.error(f"control 錯誤: {e}")

    return jsonify({"status": "ok"})


# ── 對講 API ──
@app.route("/intercom/presets")
def intercom_presets():
    """取得預設對講訊息"""
    if not intercom:
        return jsonify([])
    from intercom import PRESET_MESSAGES
    return jsonify(PRESET_MESSAGES)


@app.route("/intercom/listen")
def intercom_listen():
    """取得最近 0.5 秒麥克風音訊（WAV 格式，供瀏覽器輪詢播放）"""
    if not intercom or not intercom.is_active:
        return b'', 204
    if not audio_reader.is_available:
        return b'', 204
    try:
        import numpy as np
        buf = audio_reader.get_audio_buffer(0.5)
        if buf is None or len(buf) == 0:
            return b'', 204
        pcm = (buf * 32767).astype(np.int16).tobytes()
        wav = intercom._pcm_to_wav(pcm, 48000)
        return Response(wav, mimetype="audio/wav")
    except Exception:
        return b'', 204


@app.route("/status")
def status():
    s = get_state()
    return jsonify({
        "mission_stage": s["mission_stage"],
        "mode": s["mode"],
        "person_count": s["person_count"],
        "fallen_count": s["fallen_count"],
        "pose_anomaly_score": s["pose_anomaly_score"],
        "victim_score": s["victim_score"],
        "risk_level": s["risk_level"],
        "distance_cm": s["distance_cm"],
        "audio_event": s["audio_event"],
        "fusion_components": s["fusion_components"],
        "search_mode": s["search_mode"],
        "alert_count": s["alert_count"],
        "event_count": s["event_count"],
        "fps": s["fps"],
        "detect_ms": s["detect_ms"],
        "ai_loaded": s["ai_loaded"],
        "ai_backend": detector.backend_name,
        "mic_ok": s["mic_ok"],
        "camera_ok": s["camera_ok"],
        "gpio_ok": s["gpio_ok"],
        "patrol_pan": s["patrol_pan"],
        "wave_detected": s["wave_detected"],
        "eye_state": s.get("eye_state", "UNKNOWN"),
        "heart_rate_bpm": s.get("heart_rate_bpm", -1.0),
        "rppg_confidence": s.get("rppg_confidence", 0.0),
        "rppg_signal_quality": s.get("rppg_signal_quality", "UNKNOWN"),
        "night_vision": s.get("night_vision", "auto"),
        "brightness_avg": s.get("brightness_avg", 128),
        "unique_person_count": s.get("unique_person_count", 0),
        "unreported_count": s.get("unreported_count", 0),
        "heat_map_coverage": s.get("heat_map_coverage", 0.0),
        "heat_map": heat_map.get_grid_data() if heat_map else None,
        "objects": s.get("objects", []),
        "tracks": detector.tracker.get_tracks_info() if detector.tracker else [],
        "intercom_active": intercom.is_active if intercom else False,
        "backtrack_steps": bt_engine.stack_size if bt_engine else 0,
        "is_backtracking": bt_engine.is_backtracking if bt_engine else False,
        "logs": s["logs"][:20],
        "events": s["events"][:10],
    })


# ============================================================
# 啟動
# ============================================================
if __name__ == "__main__":
    with state_lock:
        app_state["ai_loaded"] = detector.is_loaded

    threading.Thread(target=detection_loop, daemon=True).start()
    threading.Thread(target=audio_loop, daemon=True).start()
    threading.Thread(target=ultrasonic_loop, daemon=True).start()
    threading.Thread(target=search_loop, daemon=True).start()

    logger.info("所有模組初始化完成")
    add_log("info", "搜救機器人系統 V2 啟動")

    print()
    print("=" * 52)
    print("  搜救機器人系統 V2 已啟動！")
    print(f"  控制台: http://0.0.0.0:{config.FLASK_PORT}")
    print(f"  Demo:   http://0.0.0.0:{config.FLASK_PORT}/demo")
    print("=" * 52)
    print()

    try:
        app.run(
            host=config.FLASK_HOST,
            port=config.FLASK_PORT,
            debug=False,
            threaded=True,
        )
    except KeyboardInterrupt:
        logger.info("收到中斷訊號...")
    finally:
        motor.cleanup()
        servo.cleanup()
        ultrasonic.cleanup()
        camera.cleanup()
        audio_reader.cleanup()
        logger.info("系統安全關閉")
