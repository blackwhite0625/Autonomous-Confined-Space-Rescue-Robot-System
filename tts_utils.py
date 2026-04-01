"""
搜救機器人 — 共用 TTS 語音合成工具
==================================
集中管理 gTTS / espeak-ng / espeak / say 語音播放邏輯，
供 hri_module.py 和 notifier.py 共用，避免代碼重複。

重要：pygame.mixer 非 thread-safe，所有播放操作必須經過 _mixer_lock 保護，
      避免多執行緒同時存取導致 double free 崩潰。
"""

import sys
import os
import time
import shutil
import subprocess
import logging
import tempfile
import threading

logger = logging.getLogger("rescue.tts")

# ── 全域互斥鎖：保護 pygame.mixer 避免多執行緒同時存取 ──
_mixer_lock = threading.Lock()

# ── gTTS + pygame（高音質，需網路）──
GTTS_AVAILABLE = False
try:
    from gtts import gTTS
    import pygame
    if not pygame.mixer.get_init():
        pygame.mixer.init()
    GTTS_AVAILABLE = True
except ImportError:
    pass

# ── 本機 TTS 引擎偵測 ──
_TTS_ENGINE = None
if sys.platform == "darwin":
    _TTS_ENGINE = "say"
elif shutil.which("espeak-ng"):
    _TTS_ENGINE = "espeak-ng"
elif shutil.which("espeak"):
    _TTS_ENGINE = "espeak"
elif shutil.which("piper"):
    _TTS_ENGINE = "piper"

if _TTS_ENGINE:
    logger.info(f"TTS 引擎: {_TTS_ENGINE}")
elif not GTTS_AVAILABLE:
    logger.warning("未偵測到任何 TTS 引擎（gTTS / espeak-ng / espeak），語音功能不可用")


def _play_gtts(text: str, lang: str = "zh-TW") -> bool:
    """
    用 gTTS + pygame 播放語音。
    回傳 True 表示播放成功，False 表示失敗（應回落到其他引擎）。
    必須在 _mixer_lock 保護下呼叫。
    """
    try:
        tts = gTTS(text=text, lang=lang)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
            tmp_name = fp.name
        tts.save(tmp_name)

        pygame.mixer.music.load(tmp_name)
        pygame.mixer.music.play()

        clock = pygame.time.Clock()
        wait_ticks = 0
        while pygame.mixer.music.get_busy() and wait_ticks < 150:  # max 15s
            clock.tick(10)
            wait_ticks += 1

        # 確保完全停止後再 unload，避免 double free
        pygame.mixer.music.stop()
        time.sleep(0.05)
        pygame.mixer.music.unload()

        try:
            os.remove(tmp_name)
        except OSError:
            pass
        return True
    except Exception as e:
        logger.debug(f"gTTS 失敗: {e}")
        # 安全清理：確保不留下殘餘的播放狀態
        try:
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
        except Exception:
            pass
        return False


def speak(text: str, fallback_alert_fn=None):
    """
    播放語音文字（thread-safe）。
    優先使用 gTTS（高音質），回落到系統 TTS 引擎，
    最後使用 fallback_alert_fn（警報音效）。
    """
    # 1. gTTS（需取得鎖保護 pygame.mixer）
    if GTTS_AVAILABLE:
        with _mixer_lock:
            if _play_gtts(text):
                return

    # 2. 本機 TTS 引擎（不需要鎖，用獨立行程）
    try:
        if _TTS_ENGINE == "say":
            proc = subprocess.Popen(
                ["say", text],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            proc.wait(timeout=8)
        elif _TTS_ENGINE in ("espeak-ng", "espeak"):
            voice = "cmn" if _TTS_ENGINE == "espeak-ng" else "zh"
            proc = subprocess.Popen(
                [_TTS_ENGINE, f"-v{voice}", "-s", "130", text],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
            proc.wait(timeout=10)
        else:
            if fallback_alert_fn:
                fallback_alert_fn()
            time.sleep(1)
            return

        logger.debug(f"TTS 播完: {text[:20]}...")

    except FileNotFoundError:
        logger.warning(f"TTS 程式 {_TTS_ENGINE} 不存在")
        if fallback_alert_fn:
            fallback_alert_fn()
        time.sleep(1)
    except subprocess.TimeoutExpired:
        logger.warning("TTS 語音播放超時")
    except Exception as e:
        logger.warning(f"TTS 錯誤: {e}")
        time.sleep(0.5)


def speak_emergency(text_zh: str, text_en: str, fallback_alert_fn=None):
    """
    播放緊急語音（thread-safe，含英文備案）。
    用於 notifier.py 的警報廣播。
    """
    # 1. gTTS（需取得鎖）
    if GTTS_AVAILABLE:
        with _mixer_lock:
            if _play_gtts(text_zh):
                return

    # 2. 系統 TTS（中文優先，英文備案）
    if sys.platform == "darwin":
        try:
            subprocess.run(["say", text_zh], timeout=10)
        except Exception:
            pass
        return

    try:
        result = subprocess.run(
            ["espeak", "-vzh", "-s", "130", text_zh],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=10
        )
        if result.returncode == 0:
            return
        subprocess.run(
            ["espeak", "-s", "130", text_en],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10
        )
    except FileNotFoundError:
        logger.warning("espeak 不可用")
        if fallback_alert_fn:
            fallback_alert_fn()
    except subprocess.TimeoutExpired:
        logger.warning("espeak 語音播放超時")
