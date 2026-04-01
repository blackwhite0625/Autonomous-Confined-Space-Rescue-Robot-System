"""
搜救機器人 — 警報系統模組
Telegram 即時通報（截圖 + 事件訊息）+ USB 喇叭語音警報
"""

import os
import time
import threading
import subprocess
import logging
import cv2
import numpy as np

logger = logging.getLogger("carbot.alert")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    logger.warning("requests 未安裝，Telegram 通知不可用")
    REQUESTS_AVAILABLE = False

import config
from location_service import LocationService

from tts_utils import speak_emergency


class AlertSystem:
    """多管道警報系統"""

    def __init__(self):
        logger.info("初始化警報系統...")
        self._last_alert_time = 0
        self._last_rescue_time = 0
        self._alert_lock = threading.Lock()
        self._alert_count = 0
        self._broadcasting = False      # 警報正在播放中
        self._cancel_requested = False  # 中斷請求旗標
        logger.info("警報系統初始化完成")

    @property
    def is_broadcasting(self):
        """警報是否正在播放（供 audio_loop 防自我觸發用）"""
        return self._broadcasting

    @property
    def last_alert_time(self):
        """最後一次警報觸發時間"""
        return max(self._last_alert_time, self._last_rescue_time)

    @property
    def alert_count(self):
        return self._alert_count

    def is_cooldown(self):
        """檢查是否在冷卻期間"""
        return (time.time() - self._last_alert_time) < config.ALERT_COOLDOWN

    def trigger_alert(self, frame=None, message=None, critical=False):
        """
        觸發警報（含冷卻與防重疊機制）
        """
        # 如果上一次警報都還沒廣播完，絕對不開新線程
        if getattr(self, "_active_alert_thread", None) and self._active_alert_thread.is_alive():
            logger.debug("目前已有警報廣播正在進行中，略過本次觸發確保系統穩定")
            return False

        with self._alert_lock:
            now = time.time()
            if critical:
                # 特急救援模式，獨立冷卻時間
                if (now - self._last_rescue_time) < 10.0:
                    return False
                self._last_rescue_time = now
            else:
                # 一般警報冷卻
                if self.is_cooldown():
                    logger.debug("警報冷卻中，略過本次觸發")
                    return False
                self._last_alert_time = now
                
            self._alert_count += 1

        alert_msg = message or "【搜救系統】偵測到疑似受困者！"
        if critical:
            logger.warning(f"[CRITICAL] 特急救援觸發: {alert_msg}")
        else:
            logger.warning(f"警報觸發: {alert_msg}")

        # 非同步執行警報動作
        t = threading.Thread(
            target=self._execute_alert,
            args=(frame, alert_msg, critical),
            daemon=True
        )
        t.start()
        self._active_alert_thread = t

        return True

    def cancel_alert(self):
        """中斷正在播放的警報（倒地者恢復正常時呼叫）"""
        self._cancel_requested = True
        logger.info("警報中斷請求已發送")

    def _execute_alert(self, frame, message, critical=False):
        """執行所有警報動作"""
        self._broadcasting = True
        self._cancel_requested = False
        try:
            # 1. 獨立非同步發送 Telegram 通知
            threading.Thread(
                target=self._send_telegram,
                args=(message, frame, critical),
                daemon=True
            ).start()

            # 2. 啟動交替語音與警報音效（可中斷）
            self._play_alert_sequence(critical)
        finally:
            self._broadcasting = False
            self._cancel_requested = False

    def _play_alert_sequence(self, critical):
        """播放警報序列：緊急時交替廣播三次，可被 cancel_alert() 中斷"""
        loop_count = 3 if critical else 1
        for i in range(loop_count):
            # 每次循環前檢查是否被中斷
            if self._cancel_requested:
                logger.info(f"警報已中斷（第 {i+1}/{loop_count} 次前）")
                return

            if critical:
                logger.info(f"[警報] 語音 ({i+1}/{loop_count})")
                self._tts_fallback(critical=True)

            if self._cancel_requested:
                logger.info(f"警報已中斷（語音後）")
                return

            logger.info(f"[警報] 警報音 ({i+1}/{loop_count})")
            self._play_audio_file()

            if critical and i < loop_count - 1:
                # 等待間隔中也檢查中斷（每 0.2 秒檢查一次）
                for _ in range(5):
                    if self._cancel_requested:
                        logger.info("警報已中斷（等待間隔中）")
                        return
                    time.sleep(0.2)

    def _play_audio_file(self):
        """僅播放底層實體警報音檔"""
        import sys
        sound_path = config.ALERT_SOUND
        if not os.path.exists(sound_path):
            logger.warning(f"警報音檔不存在: {sound_path}")
            return

        if sys.platform == "darwin":
            try:
                subprocess.run(["afplay", sound_path], timeout=10)
            except Exception as e:
                logger.warning(f"Mac 音訊播放失敗: {e}")
            return

        # 嘗試找到 USB 音訊裝置
        usb_device = self._find_usb_audio_device()

        try:
            if usb_device:
                cmd = ["aplay", "-D", usb_device, sound_path]
            else:
                cmd = ["aplay", sound_path]

            result = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15
            )
        except Exception as e:
            logger.warning(f"警報音檔播放失敗: {e}")

    def _find_usb_audio_device(self):
        """自動偵測 USB 音訊裝置，回傳 aplay 裝置名稱"""
        try:
            result = subprocess.run(
                ["aplay", "-l"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=5
            )
            output = result.stdout.decode("utf-8", errors="ignore")
            # 找到包含 USB 的音效卡
            for line in output.split("\n"):
                if "USB" in line.upper() and "card" in line.lower():
                    # 格式: "card 1: Device [USB Audio], device 0: ..."
                    try:
                        card_num = line.split("card")[1].strip().split(":")[0].strip()
                        return f"plughw:{card_num},0"
                    except (IndexError, ValueError):
                        pass
        except Exception as e:
            logger.debug(f"偵測 USB 音訊失敗: {e}")
        return None

    def _tts_fallback(self, critical=False):
        """使用共用 TTS 工具播放語音"""
        if critical:
            text_zh = "緊急警報！這裡有傷患！這裡有傷患！請盡速前往救援！"
            text_en = "Emergency! Victim located here! Please rescue immediately!"
        else:
            text_zh = "前方偵測到異常，系統正在確認中。"
            text_en = "Anomaly detected, system is verifying."
        speak_emergency(text_zh, text_en)

    def _send_telegram(self, message, frame=None, critical=False):
        """發送 Telegram 通知（含截圖）"""
        if not REQUESTS_AVAILABLE:
            logger.warning("requests 未安裝，無法發送 Telegram")
            return

        token = config.TELEGRAM_BOT_TOKEN
        chat_id = config.TELEGRAM_CHAT_ID
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        loc_str = ""
        if critical:
            loc_str = "\n" + LocationService.get_location()

        full_message = (
            f"{message}\n"
            f"時間: {timestamp}\n"
            f"累計警報: 第 {self._alert_count} 次"
            f"{loc_str}"
        )

        try:
            if frame is not None:
                # 發送截圖
                _, buffer = cv2.imencode('.jpg', frame,
                                         [cv2.IMWRITE_JPEG_QUALITY, 85])
                url = f"https://api.telegram.org/bot{token}/sendPhoto"
                files = {"photo": ("alert.jpg", buffer.tobytes(), "image/jpeg")}
                data = {"chat_id": chat_id, "caption": full_message}
                resp = requests.post(url, files=files, data=data, timeout=10)
                resp.raise_for_status()
                logger.info("Telegram 截圖通知已發送")
            else:
                # 僅發送文字
                url = f"https://api.telegram.org/bot{token}/sendMessage"
                data = {"chat_id": chat_id, "text": full_message}
                resp = requests.post(url, data=data, timeout=10)
                resp.raise_for_status()
                logger.info("Telegram 文字通知已發送")

        except Exception as e:
            err_msg = str(e)
            if hasattr(e, 'response') and e.response is not None:
                err_msg += f" | 回應: {e.response.text}"
            logger.error(f"Telegram 發送失敗: {err_msg}")

    def cleanup(self):
        """清理"""
        logger.info("警報系統已關閉")
