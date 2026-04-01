"""
CarBot 車子移動模組
===================
麥克拉姆輪全向移動控制。
使用 4 個馬達 + 2 個 L298N 驅動板。
"""

import logging
import config

logger = logging.getLogger("rescue.motor")

# GPIO 已在 config.py 統一初始化（含 PiGPIOFactory）
try:
    from gpiozero import PWMOutputDevice, OutputDevice
    GPIO_OK = config.GPIO_AVAILABLE
except ImportError:
    GPIO_OK = False


class MotorController:
    """麥克拉姆輪控制器"""

    def __init__(self):
        self.gpio_ok = GPIO_OK

        if self.gpio_ok:
            try:
                f = config.L298N_FRONT
                r = config.L298N_REAR

                # 左前輪 FL
                self._fl_pwm = PWMOutputDevice(f["ena"])
                self._fl_in1 = OutputDevice(f["in1"])
                self._fl_in2 = OutputDevice(f["in2"])

                # 右前輪 FR
                self._fr_pwm = PWMOutputDevice(f["enb"])
                self._fr_in1 = OutputDevice(f["in3"])
                self._fr_in2 = OutputDevice(f["in4"])

                # 左後輪 RL
                self._rl_pwm = PWMOutputDevice(r["ena"])
                self._rl_in1 = OutputDevice(r["in1"])
                self._rl_in2 = OutputDevice(r["in2"])

                # 右後輪 RR
                self._rr_pwm = PWMOutputDevice(r["enb"])
                self._rr_in1 = OutputDevice(r["in3"])
                self._rr_in2 = OutputDevice(r["in4"])

                logger.info("✅ 馬達控制器初始化完成")
            except Exception as e:
                logger.error(f"❌ 馬達初始化失敗: {e}")
                self.gpio_ok = False
        else:
            logger.warning("⚠️ 馬達控制器：模擬模式（GPIO 不可用）")

    def _set_motor(self, pwm, in1, in2, speed):
        """控制單一馬達，速度範圍 -1.0 ~ 1.0"""
        if speed > 0.05:
            in1.on()
            in2.off()
            pwm.value = min(speed, 1.0)
        elif speed < -0.05:
            in1.off()
            in2.on()
            pwm.value = min(abs(speed), 1.0)
        else:
            in1.off()
            in2.off()
            pwm.value = 0

    def move(self, x, y, r):
        """
        麥克拉姆輪全向移動

        參數:
            x: 橫移 (-1 ~ 1)，正值向右
            y: 前後 (-1 ~ 1)，正值向前
            r: 旋轉 (-1 ~ 1)，正值順時針
        """
        if not self.gpio_ok:
            return

        x = -x

        speed_fl = y + x + r
        speed_fr = y - x - r
        speed_rl = y - x + r
        speed_rr = y + x - r

        # 前輪極性修正
        speed_fl = -speed_fl
        speed_fr = -speed_fr

        # 歸一化
        max_speed = max(abs(speed_fl), abs(speed_fr),
                        abs(speed_rl), abs(speed_rr))
        if max_speed > 1.0:
            speed_fl /= max_speed
            speed_fr /= max_speed
            speed_rl /= max_speed
            speed_rr /= max_speed

        self._set_motor(self._fl_pwm, self._fl_in1, self._fl_in2, speed_fl)
        self._set_motor(self._fr_pwm, self._fr_in1, self._fr_in2, speed_fr)
        self._set_motor(self._rl_pwm, self._rl_in1, self._rl_in2, speed_rl)
        self._set_motor(self._rr_pwm, self._rr_in1, self._rr_in2, speed_rr)

    def stop(self):
        self.move(0, 0, 0)

    def cleanup(self):
        self.stop()
        if self.gpio_ok:
            for dev in [self._fl_pwm, self._fl_in1, self._fl_in2,
                        self._fr_pwm, self._fr_in1, self._fr_in2,
                        self._rl_pwm, self._rl_in1, self._rl_in2,
                        self._rr_pwm, self._rr_in1, self._rr_in2]:
                dev.close()
            logger.info("✅ 馬達 GPIO 已釋放")
