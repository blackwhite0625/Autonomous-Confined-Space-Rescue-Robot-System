"""
搜救機器人 — 語音對講模組
============================
操作員透過網頁與受困者通話。

方向 1（操作員→受困者）：網頁輸入文字 → TTS → USB 喇叭播放
方向 2（受困者→操作員）：USB 麥克風 → 音訊串流 → 瀏覽器播放

不需要 HTTPS，不需要瀏覽器麥克風權限。
"""

import io
import time
import struct
import threading
import logging
import subprocess
import sys
import os
import tempfile

logger = logging.getLogger("rescue.intercom")

try:
    import numpy as np
    NP_OK = True
except ImportError:
    NP_OK = False

# 預設訊息（操作員快速選擇）
PRESET_MESSAGES = [
    "有人聽到嗎？我們是搜救隊",
    "請不要移動，救援即將到達",
    "請發出聲音讓我們定位你的位置",
    "你受傷了嗎？請回答",
    "請保持冷靜，我們正在靠近",
]


class Intercom:
    """語音對講（文字轉語音 + 麥克風串流）"""

    def __init__(self, audio_reader=None):
        self._audio_reader = audio_reader
        self._active = False
        self._speaking = False
        self._lock = threading.Lock()
        logger.info("對講模組初始化完成")

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    def start(self):
        self._active = True
        logger.info("對講模式已開啟")

    def stop(self):
        self._active = False
        logger.info("對講模式已關閉")

    # ── 方向 1：操作員 → 受困者（TTS）──

    def speak(self, text: str):
        """將文字轉語音播放到 USB 喇叭（非阻塞）"""
        if not text or self._speaking:
            return
        threading.Thread(target=self._do_speak, args=(text,), daemon=True).start()

    def _do_speak(self, text: str):
        """TTS 播放實作"""
        self._speaking = True
        try:
            # 方法 1: gTTS（需網路）
            try:
                from gtts import gTTS
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                    tts = gTTS(text=text, lang='zh-TW')
                    tts.save(f.name)
                    tmp_path = f.name

                if sys.platform == "darwin":
                    subprocess.run(["afplay", tmp_path], timeout=15)
                else:
                    # 用 mpg123 或 pygame 播放 mp3
                    try:
                        subprocess.run(["mpg123", "-q", tmp_path],
                                       timeout=15, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL)
                    except FileNotFoundError:
                        # 備援：用 pygame
                        try:
                            import pygame
                            pygame.mixer.init()
                            pygame.mixer.music.load(tmp_path)
                            pygame.mixer.music.play()
                            while pygame.mixer.music.get_busy():
                                time.sleep(0.1)
                        except Exception:
                            pass
                os.unlink(tmp_path)
                logger.info(f"[對講] 播放: {text[:20]}...")
                return
            except ImportError:
                pass

            # 方法 2: espeak（離線，聲音較差但不需網路）
            try:
                subprocess.run(
                    ["espeak", "-v", "zh", text],
                    timeout=15, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                logger.info(f"[對講] espeak: {text[:20]}...")
                return
            except FileNotFoundError:
                pass

            logger.warning("[對講] 無可用 TTS 引擎")

        except Exception as e:
            logger.error(f"[對講] 播放錯誤: {e}")
        finally:
            self._speaking = False

    # ── 方向 2：受困者 → 操作員（麥克風串流）──

    def generate_audio_stream(self):
        """
        串流 RPi 麥克風到瀏覽器。
        產生連續的 WAV 片段，瀏覽器用 <audio> 播放。
        """
        if not NP_OK or not self._audio_reader or not self._audio_reader.is_available:
            return

        while self._active:
            try:
                buf = self._audio_reader.get_audio_buffer(0.5)
                if buf is None or len(buf) == 0:
                    time.sleep(0.1)
                    continue

                pcm_16 = (buf * 32767).astype(np.int16)
                wav = self._pcm_to_wav(pcm_16.tobytes(), 48000)
                yield wav
                time.sleep(0.3)

            except Exception as e:
                logger.debug(f"對講串流錯誤: {e}")
                time.sleep(0.5)

    @staticmethod
    def get_presets() -> list:
        """取得預設訊息清單"""
        return PRESET_MESSAGES

    @staticmethod
    def _pcm_to_wav(pcm: bytes, sr: int, ch: int = 1, sw: int = 2) -> bytes:
        """PCM → WAV"""
        sz = len(pcm)
        buf = io.BytesIO()
        buf.write(b'RIFF')
        buf.write(struct.pack('<I', 36 + sz))
        buf.write(b'WAVE')
        buf.write(b'fmt ')
        buf.write(struct.pack('<IHHIIHH', 16, 1, ch, sr, sr * ch * sw, ch * sw, sw * 8))
        buf.write(b'data')
        buf.write(struct.pack('<I', sz))
        buf.write(pcm)
        return buf.getvalue()
