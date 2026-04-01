"""
CarBot 喇叭模組
================
USB 喇叭播放警報音效。
"""

import os
import sys
import subprocess
import threading
import logging
import config

logger = logging.getLogger("rescue.speaker")


class Speaker:
    """USB 喇叭控制"""

    def __init__(self):
        self._sound_file = config.ALERT_SOUND
        self._playing = False

        if os.path.exists(self._sound_file):
            logger.info(f"✅ 喇叭模組就緒（音檔: {os.path.basename(self._sound_file)}）")
        else:
            logger.warning(f"⚠️ 警報音檔不存在: {self._sound_file}")

    def play_alert(self):
        """
        非同步播放警報音（不阻塞主程式）。
        重複呼叫時，若正在播放中則忽略。
        """
        if self._playing:
            return
        if not os.path.exists(self._sound_file):
            return

        thread = threading.Thread(target=self._play, daemon=True)
        thread.start()

    def _play(self):
        """實際播放（在背景執行緒中執行）"""
        self._playing = True
        try:
            if sys.platform == "darwin":
                cmd = ["afplay", self._sound_file]
            else:
                cmd = ["aplay", self._sound_file]

            subprocess.run(
                cmd, timeout=15,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            logger.warning(f"⚠️ 警報音播放失敗: {e}")
        finally:
            self._playing = False
