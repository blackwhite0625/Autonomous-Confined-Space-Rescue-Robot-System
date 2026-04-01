"""
搜救機器人 — USB 麥克風音訊讀取模組
====================================
背景執行緒連續讀取麥克風輸入，環形緩衝區保存最近 N 秒音訊。
使用 sounddevice 庫（跨平台，安裝簡單）。
"""

import threading
import time
import logging
import numpy as np

logger = logging.getLogger("rescue.audio_reader")

try:
    import sounddevice as sd
    SD_AVAILABLE = True
except (ImportError, OSError) as e:
    SD_AVAILABLE = False
    logger.warning(f"sounddevice 不可用（{e}）。音訊功能已停用。")
    logger.warning("請安裝: sudo apt install portaudio19-dev && pip install sounddevice")

import config


class AudioReader:
    """USB 麥克風連續讀取器（環形緩衝）"""

    def __init__(self):
        self._sample_rate = config.MIC_SAMPLE_RATE
        self._chunk_size = config.MIC_CHUNK_SIZE
        self._buffer_sec = config.MIC_BUFFER_SEC
        self._buffer_size = int(self._sample_rate * self._buffer_sec)

        self._buffer = np.zeros(self._buffer_size, dtype=np.float32)
        self._write_pos = 0
        self._lock = threading.Lock()
        self._running = False
        self._stream = None
        self._available = False

        # 最新一塊 chunk（供即時偵測用）
        self._latest_chunk = np.zeros(self._chunk_size, dtype=np.float32)
        self._chunk_ready = threading.Event()

        if SD_AVAILABLE:
            self._open()
        else:
            logger.warning("音訊讀取器：模擬模式（sounddevice 不可用）")

    def _open(self):
        """開啟麥克風串流"""
        try:
            device_index = config.MIC_DEVICE_INDEX
            if device_index is None:
                device_index = self._find_usb_mic()

            self._stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype='float32',
                blocksize=self._chunk_size,
                device=device_index,
                callback=self._audio_callback,
            )
            self._stream.start()
            self._running = True
            self._available = True
            dev_name = sd.query_devices(device_index)['name'] if device_index else "預設裝置"
            logger.info(f"✅ 麥克風開啟成功（{dev_name}, {self._sample_rate}Hz）")
        except Exception as e:
            logger.error(f"❌ 麥克風開啟失敗: {e}")
            self._available = False

    def _find_usb_mic(self):
        """自動偵測 USB 麥克風"""
        try:
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    name = dev['name'].lower()
                    if 'usb' in name or 'mic' in name:
                        logger.info(f"🎤 偵測到 USB 麥克風: [{i}] {dev['name']}")
                        return i
            # 找不到 USB 麥克風，使用預設輸入裝置
            default = sd.default.device[0]
            if default is not None and default >= 0:
                logger.info(f"🎤 使用預設輸入裝置: [{default}] {sd.query_devices(default)['name']}")
                return default
        except Exception as e:
            logger.warning(f"麥克風偵測失敗: {e}")
        return None

    def _audio_callback(self, indata, frames, time_info, status):
        """sounddevice 回呼函數（在音訊執行緒中執行）"""
        if status:
            logger.debug(f"音訊狀態: {status}")

        audio = indata[:, 0]  # 單聲道

        with self._lock:
            # 寫入環形緩衝
            end = self._write_pos + len(audio)
            if end <= self._buffer_size:
                self._buffer[self._write_pos:end] = audio
            else:
                first = self._buffer_size - self._write_pos
                self._buffer[self._write_pos:] = audio[:first]
                self._buffer[:len(audio) - first] = audio[first:]
            self._write_pos = end % self._buffer_size

            # 更新即時 chunk
            self._latest_chunk = audio.copy()

        self._chunk_ready.set()

    @property
    def is_available(self) -> bool:
        return self._available

    def get_latest_chunk(self) -> np.ndarray:
        """取得最新一塊音訊 chunk"""
        with self._lock:
            return self._latest_chunk.copy()

    def get_audio_buffer(self, duration_sec: float = None) -> np.ndarray:
        """取得指定長度的音訊緩衝（從最新時間往前）"""
        duration = duration_sec or self._buffer_sec
        samples = min(int(self._sample_rate * duration), self._buffer_size)

        with self._lock:
            end = self._write_pos
            start = end - samples
            if start >= 0:
                return self._buffer[start:end].copy()
            else:
                return np.concatenate([
                    self._buffer[start % self._buffer_size:],
                    self._buffer[:end]
                ]).copy()

    def wait_for_audio(self, timeout_sec: float = 5.0) -> bool:
        """等待新的音訊資料到達"""
        self._chunk_ready.clear()
        return self._chunk_ready.wait(timeout=timeout_sec)

    def get_rms_level(self) -> float:
        """取得當前音訊 RMS 音量（0~1）"""
        chunk = self.get_latest_chunk()
        return float(np.sqrt(np.mean(chunk ** 2)))

    def cleanup(self):
        self._running = False
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            logger.info("✅ 麥克風已釋放")
