"""
搜救機器人 — rPPG 遠端光體積描記術模組
========================================
透過攝影機分析臉部皮膚綠色通道的微小變化提取心率。
原理：血液流動造成皮膚顏色週期性變化，與心跳同步。

流程：
  1. 從 YOLOv8-pose 臉部關鍵點 (0-4) 定位臉部 ROI
  2. 提取綠色通道空間平均值（每幀一個數值）
  3. 累積滾動緩衝 (~4 秒)
  4. 帶通濾波 (0.7-3.5 Hz) + FFT → 心率 BPM
"""

import time
import logging
import numpy as np
from collections import deque

import config

logger = logging.getLogger("rescue.rppg")

# scipy 為選用依賴
try:
    from scipy import signal as sp_signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy 未安裝，rPPG 功能停用（pip install scipy）")


class rPPGDetector:
    """
    遠端光體積描記術心率偵測器。
    每幀呼叫 process_frame()，內部累積信號後定期計算心率。
    """

    def __init__(self):
        fps = config.CAMERA_FPS
        buf_len = int(config.RPPG_BUFFER_SECONDS * fps)

        self._buffer = deque(maxlen=buf_len)      # 綠色通道時間序列
        self._timestamps = deque(maxlen=buf_len)   # 對應時間戳
        self._prev_face_center = None              # 上一幀臉部中心（穩定度檢查）
        self._frame_count = 0
        self._fps = fps
        self._update_interval = config.RPPG_UPDATE_INTERVAL
        self._min_conf = config.RPPG_MIN_FACE_CONFIDENCE
        self._stability_px = config.RPPG_ROI_STABILITY_PX
        self._min_buffer = int(fps * 2)  # 至少 2 秒資料才開始計算

        # 快取結果（兩次 FFT 之間保持上次結果）
        self._last_result = {
            "bpm": -1.0,
            "confidence": 0.0,
            "valid": False,
            "roi_stable": False,
            "signal_quality": "UNKNOWN",
        }

        # 預建帶通濾波器（避免每次重建）
        self._sos = None
        if SCIPY_AVAILABLE:
            try:
                self._sos = sp_signal.butter(
                    4,
                    [config.RPPG_BANDPASS_LOW_HZ, config.RPPG_BANDPASS_HIGH_HZ],
                    btype='bandpass',
                    fs=fps,
                    output='sos'
                )
            except Exception as e:
                logger.error(f"帶通濾波器建立失敗: {e}")

        logger.info(
            f"rPPG 偵測器初始化 | "
            f"緩衝 {buf_len} 幀 ({config.RPPG_BUFFER_SECONDS}s) | "
            f"每 {self._update_interval} 幀更新"
        )

    def process_frame(self, frame: np.ndarray, face_keypoints: np.ndarray,
                      timestamp: float) -> dict:
        """
        處理一幀影像，提取臉部綠色通道並累積。
        定期執行 FFT 計算心率。

        Args:
            frame: BGR 影像
            face_keypoints: keypoints[0:5] (nose, left_eye, right_eye, left_ear, right_ear)
            timestamp: 當前時間

        Returns:
            {"bpm": float, "confidence": float, "valid": bool,
             "roi_stable": bool, "signal_quality": str}
        """
        if not SCIPY_AVAILABLE or self._sos is None:
            return self._last_result

        self._frame_count += 1

        # 1. 檢查臉部關鍵點可見性
        visible_count = sum(1 for kp in face_keypoints[:3] if kp[2] > self._min_conf)
        if visible_count < 2:
            # 臉部不夠可見，清空緩衝重來
            self._prev_face_center = None
            self._last_result["roi_stable"] = False
            self._last_result["signal_quality"] = "UNKNOWN"
            return self._last_result

        # 2. 計算臉部中心 + 穩定度檢查
        face_center = self._get_face_center(face_keypoints)
        roi_stable = self._check_stability(face_center)
        self._prev_face_center = face_center
        self._last_result["roi_stable"] = roi_stable

        if not roi_stable:
            # 臉部晃動太大，清空緩衝
            self._buffer.clear()
            self._timestamps.clear()
            self._last_result["signal_quality"] = "WEAK"
            return self._last_result

        # 3. 提取臉部 ROI 綠色通道
        green_val = self._extract_green(frame, face_keypoints)
        if green_val is None:
            return self._last_result

        self._buffer.append(green_val)
        self._timestamps.append(timestamp)

        # 4. 定期計算心率
        if (self._frame_count % self._update_interval == 0 and
                len(self._buffer) >= self._min_buffer):
            bpm, confidence, valid = self._compute_heart_rate()
            quality = "GOOD" if valid else ("WEAK" if confidence > 0.3 else "UNKNOWN")
            self._last_result = {
                "bpm": round(bpm, 1) if valid else -1.0,
                "confidence": round(confidence, 2),
                "valid": valid,
                "roi_stable": roi_stable,
                "signal_quality": quality,
            }
            if valid:
                logger.info(f"rPPG: {bpm:.0f} BPM (信心度 {confidence:.0%})")

        return self._last_result

    def _get_face_center(self, kps) -> tuple:
        """從臉部關鍵點計算中心座標"""
        valid = [(kp[0], kp[1]) for kp in kps[:3] if kp[2] > self._min_conf]
        if not valid:
            return (0, 0)
        cx = sum(p[0] for p in valid) / len(valid)
        cy = sum(p[1] for p in valid) / len(valid)
        return (cx, cy)

    def _check_stability(self, center) -> bool:
        """檢查臉部是否穩定（幀間位移 < 閾值）"""
        if self._prev_face_center is None:
            return True
        dx = center[0] - self._prev_face_center[0]
        dy = center[1] - self._prev_face_center[1]
        drift = (dx * dx + dy * dy) ** 0.5
        return drift < self._stability_px

    def _extract_green(self, frame, kps) -> float:
        """
        從臉部 ROI 提取綠色通道平均值。
        ROI：用 nose + eyes 建立矩形，向外擴展 30%。
        """
        h, w = frame.shape[:2]
        valid = [(int(kp[0]), int(kp[1])) for kp in kps[:3] if kp[2] > self._min_conf]
        if len(valid) < 2:
            return None

        xs = [p[0] for p in valid]
        ys = [p[1] for p in valid]

        # 臉部矩形 + 30% 擴展
        face_w = max(xs) - min(xs)
        face_h = max(ys) - min(ys)
        pad_x = max(int(face_w * 0.3), 10)
        pad_y = max(int(face_h * 0.3), 10)

        x1 = max(0, min(xs) - pad_x)
        y1 = max(0, min(ys) - pad_y)
        x2 = min(w, max(xs) + pad_x)
        y2 = min(h, max(ys) + pad_y)

        if x2 - x1 < 10 or y2 - y1 < 10:
            return None

        # BGR 的 G 通道 = index 1
        roi = frame[y1:y2, x1:x2, 1]
        return float(roi.mean())

    def _compute_heart_rate(self) -> tuple:
        """
        從綠色通道時間序列計算心率。
        回傳 (bpm, confidence, is_valid)。
        """
        signal_array = np.array(self._buffer, dtype=np.float64)

        # 去趨勢（移除緩慢漂移）
        signal_array = signal_array - np.mean(signal_array)

        # 標準化
        std = np.std(signal_array)
        if std < 1e-6:
            return -1.0, 0.0, False
        signal_array = signal_array / std

        # 帶通濾波
        try:
            filtered = sp_signal.sosfiltfilt(self._sos, signal_array)
        except Exception:
            return -1.0, 0.0, False

        # 估算實際 FPS（用時間戳）
        if len(self._timestamps) >= 2:
            dt = self._timestamps[-1] - self._timestamps[0]
            actual_fps = (len(self._timestamps) - 1) / dt if dt > 0 else self._fps
        else:
            actual_fps = self._fps

        # FFT
        n = len(filtered)
        fft_mag = np.abs(np.fft.rfft(filtered))
        freqs = np.fft.rfftfreq(n, d=1.0 / actual_fps)

        # 限制在生理心率範圍
        low_hz = config.RPPG_BANDPASS_LOW_HZ
        high_hz = config.RPPG_BANDPASS_HIGH_HZ
        mask = (freqs >= low_hz) & (freqs <= high_hz)

        if not np.any(mask):
            return -1.0, 0.0, False

        valid_fft = fft_mag[mask]
        valid_freqs = freqs[mask]

        # 找峰值
        peak_idx = np.argmax(valid_fft)
        peak_freq = valid_freqs[peak_idx]
        peak_mag = valid_fft[peak_idx]

        # BPM
        bpm = peak_freq * 60.0

        # 信心度：峰值與平均值的比值（信雜比）
        mean_mag = np.mean(valid_fft)
        if mean_mag > 0:
            snr = peak_mag / mean_mag
            confidence = min(snr / 5.0, 1.0)  # SNR=5 → 100% 信心
        else:
            confidence = 0.0

        # 有效性判斷
        is_valid = (
            40 <= bpm <= 180 and
            confidence >= config.RPPG_CONFIDENCE_MIN
        )

        return bpm, confidence, is_valid
