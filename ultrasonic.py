"""
CarBot 超聲波模組
==================
HC-SR04 超聲波感測器測距。
"""

import time
import logging
import config

logger = logging.getLogger("rescue.ultrasonic")

try:
    from gpiozero import DistanceSensor
    GPIO_OK = config.GPIO_AVAILABLE
except ImportError:
    GPIO_OK = False


class UltrasonicSensor:
    """HC-SR04 超聲波感測器"""

    def __init__(self):
        self.gpio_ok = GPIO_OK
        self._distance_cm = -1

        if self.gpio_ok:
            try:
                self._sensor = DistanceSensor(
                    echo=config.ULTRASONIC_ECHO,
                    trigger=config.ULTRASONIC_TRIG,
                    max_distance=4.0,
                    queue_len=5,
                )
                # 等待感測器穩定
                time.sleep(0.5)
                logger.info("✅ 超聲波感測器初始化完成")
            except Exception as e:
                logger.error(f"❌ 超聲波初始化失敗: {e}")
                self.gpio_ok = False
        else:
            logger.warning("⚠️ 超聲波感測器：模擬模式（GPIO 不可用）")

    def get_distance_cm(self):
        """
        讀取距離 (公分)

        回傳:
            float: 距離公分數，-1 代表讀取失敗
        """
        if not self.gpio_ok:
            return -1

        try:
            d = self._sensor.distance
            if d is not None:
                self._distance_cm = round(d * 100, 1)
            else:
                self._distance_cm = -1
        except Exception as e:
            logger.warning(f"超聲波讀取錯誤: {e}")
            self._distance_cm = -1

        return self._distance_cm

    def cleanup(self):
        if self.gpio_ok:
            self._sensor.close()
            logger.info("✅ 超聲波 GPIO 已釋放")
