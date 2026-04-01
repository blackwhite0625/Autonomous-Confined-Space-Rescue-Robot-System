"""
CarBot 攝影機模組
==================
USB 攝影機管理：開啟、讀幀、高效能 MJPEG 串流。
"""

import threading
import time
import logging
import config

logger = logging.getLogger("rescue.camera")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("⚠️ opencv 未安裝，攝影機不可用")


class Camera:
    """USB 攝影機管理器"""

    def __init__(self):
        self._cap = None
        self._frame = None           # 原始影像
        self._display_frame = None   # 已標註影像（供串流用）
        self._jpeg_cache = None      # 預編碼 JPEG 快取
        self._lock = threading.Lock()
        self._running = False

        if CV2_AVAILABLE:
            self._open()
        else:
            logger.warning("⚠️ 攝影機：模擬模式（opencv 不可用）")

    def _open(self):
        """開啟攝影機"""
        try:
            self._cap = cv2.VideoCapture(config.CAMERA_INDEX)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
            self._cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 減少緩衝延遲

            if self._cap.isOpened():
                w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                logger.info(f"✅ 攝影機開啟成功（{w}×{h}）")
                self._running = True
                self._thread = threading.Thread(target=self._read_loop, daemon=True)
                self._thread.start()
            else:
                logger.error("❌ 無法開啟攝影機")
                self._cap = None
        except Exception as e:
            logger.error(f"❌ 攝影機開啟錯誤: {e}")
            self._cap = None

    def _read_loop(self):
        """背景持續讀取影像幀（含最低間隔防止空轉燒 CPU）"""
        while self._running and self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
            # 即使讀幀成功也短暫休眠，讓出 CPU 給其他執行緒
            time.sleep(0.001)

    def get_frame(self):
        """取得最新原始影像（零拷貝，供偵測用）"""
        with self._lock:
            return self._frame

    def set_display_frame(self, frame):
        """設定已標註的顯示幀，同時預編碼為 JPEG"""
        if frame is None:
            return
        ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
        if ret:
            with self._lock:
                self._display_frame = frame
                self._jpeg_cache = jpeg.tobytes()

    def generate_mjpeg(self):
        """高效能 MJPEG 串流（使用預編碼快取）"""
        while True:
            with self._lock:
                jpeg_data = self._jpeg_cache

            if jpeg_data is None:
                # 尚無顯示幀，用原始影像
                with self._lock:
                    frame = self._frame
                if frame is not None:
                    ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
                    if ret:
                        jpeg_data = jpeg.tobytes()

            if jpeg_data:
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n'
                    + jpeg_data
                    + b'\r\n'
                )
            time.sleep(0.033)   # ~30fps 串流（原 0.05=20fps 太慢）

    def is_opened(self):
        return self._cap is not None and self._cap.isOpened()

    def cleanup(self):
        self._running = False
        if self._cap:
            self._cap.release()
            logger.info("✅ 攝影機已釋放")
