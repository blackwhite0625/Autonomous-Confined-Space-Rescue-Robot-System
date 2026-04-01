"""
搜救機器人系統 — 集中設定檔
所有 GPIO 腳位、模型路徑、閾值、搜救參數集中管理
"""

import os

# ============================================================
# 路徑設定
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
SOUND_DIR = os.path.join(BASE_DIR, "sounds")
EVENT_DIR = os.path.join(BASE_DIR, "events")

# ============================================================
# AI 模型路徑
# ============================================================

# --- ONNX 模型（預設使用）---
YOLO_GENERAL_MODEL = "model/yolov8n.onnx"       # 通用物件偵測 (人體)
YOLO_POSE_MODEL    = "model/yolov8n-pose.onnx"   # 姿態偵測

# 各模型推理尺寸（ONNX 後端使用，Hailo 有自己的固定 640）
YOLO_GENERAL_INFER_SIZE = 416
YOLO_POSE_INFER_SIZE    = 416

# --- Hailo HEF 模型（安裝 Hailo 驅動後啟用）---
USE_HAILO = True                                  # True = 啟用 Hailo NPU 推論
HEF_PERSON_MODEL = "model/yolov8n.hef"
HEF_POSE_MODEL   = "model/yolov8s_pose.hef"
HAILO_DEVICE_ID  = None                           # None = 自動偵測

# ============================================================
# AI 偵測參數
# ============================================================
GENERAL_CONFIDENCE = 0.50     # yolov8n 人體偵測置信度閾值
POSE_CONFIDENCE    = 0.40     # 姿態偵測信心度閾值
PERSON_CLASS_ID    = 0        # 'person' 在 COCO 中的 ID

# ============================================================
# GPIO 腳位配置 — 馬達驅動 (L298N x 2 塊)
# ============================================================
# 前輪 L298N: 控制左前 (FL) 與 右前 (FR)
L298N_FRONT = {
    "ena": 12, "in1": 6, "in2": 5,    # 左前輪 (FL)
    "enb": 26, "in3": 19, "in4": 13   # 右前輪 (FR)
}

# 後輪 L298N: 控制左後 (RL) 與 右後 (RR)
L298N_REAR = {
    "ena": 16, "in1": 27, "in2": 17,  # 左後輪 (RL)
    "enb": 20, "in3": 23, "in4": 22   # 右後輪 (RR)
}

# 馬達預設速度 (0.0 ~ 1.0)
DEFAULT_SPEED = 0.6

# ============================================================
# GPIO 腳位配置 — 舵機 (攝影機雲台)
# ============================================================
SERVO_PAN_PIN  = 24     # 水平旋轉
SERVO_TILT_PIN = 25     # 垂直傾斜

# 舵機角度範圍
SERVO_MIN_ANGLE = -90
SERVO_MAX_ANGLE = 90

# 舵機脈衝寬度 (秒) — 解放完整 180° 物理極限
SERVO_MIN_PULSE = 0.0005
SERVO_MAX_PULSE = 0.0025

# 舵機預設角度
SERVO_PAN_DEFAULT  = -12.9   # 水平預設偏移
SERVO_TILT_DEFAULT = 45.0    # 垂直預設仰角

# 巡航掃描參數
SCAN_STEP  = 5       # 每次步進角度
SCAN_DELAY = 0.15    # 每步間隔 (秒)

# ============================================================
# GPIO 腳位配置 — 超聲波感測器 (HC-SR04)
# ============================================================
ULTRASONIC_TRIG = 4
ULTRASONIC_ECHO = 21

# 安全距離 (公分)
SAFE_DISTANCE_CM     = 50   # 行人優先停止距離
OBSTACLE_DISTANCE_CM = 30   # 障礙物緊急停止

# ============================================================
# 攝影機設定
# ============================================================
CAMERA_INDEX  = 0
CAMERA_WIDTH  = 1280          # 1080p 攝影機：用 720p 平衡解析度與 CPU 負擔
CAMERA_HEIGHT = 720           # （Hailo 推論前會 resize 到 640x640，高解析度讓骨架更精準）
CAMERA_FPS    = 60            # 攝影機支援 120fps，用 60fps 平衡流暢度與 CPU/USB 頻寬

# ============================================================
# 搜救任務狀態機
# ============================================================
# 7 階段：STANDBY / SEARCH / ANOMALY / LOCK_ON / INQUIRY / CONFIRM / REPORT
# 額外：MANUAL（手動操控）

VICTIM_SUSPECT_THRESHOLD   = 0.30   # VictimScore ≥ 此值 → ANOMALY
VICTIM_HIGH_RISK_THRESHOLD = 0.60   # VictimScore ≥ 此值 → INQUIRY/REPORT

# ============================================================
# 多模態融合權重 (VictimScore)
# ============================================================
VICTIM_SCORE_WEIGHTS = {
    "person":      0.40,
    "pose":        0.18,
    "audio":       0.18,
    "motion":      0.09,
    "distance":    0.05,
    "vital_signs": 0.10,
}

# ============================================================
# 音訊設定 (V2 啟用)
# ============================================================
MIC_DEVICE_INDEX = None    # None = 自動偵測 USB 麥克風
MIC_SAMPLE_RATE  = 48000   # 取樣率 (Hz)
MIC_CHUNK_SIZE   = 4096    # 每次讀取的樣本數
MIC_BUFFER_SEC   = 5.0     # 環形緩衝區長度 (秒)

VAD_THRESHOLD    = 0.10    # Voice Activity Detection 閾值（再降，提高靈敏度）
HELP_THRESHOLD   = 0.4     # 呼救聲分類閾值（降低，減少漏報）
KNOCK_THRESHOLD  = 0.08    # 敲擊聲偵測閾值（降低，提高靈敏度）

# ============================================================
# 搜索策略
# ============================================================
SEARCH_MODE = "D"                  # D=靜止掃描 / E=智慧巡邏 / F=掃描巡邏

# ============================================================
# 智慧巡邏 (模式 E) — 走走停停掃描
# ============================================================
SMART_PATROL_SPEED       = 0.35   # 前進速度 (0~1)
SMART_PATROL_MOVE_SEC    = 3.0    # 每段前進持續秒數
SMART_PATROL_OBSTACLE    = 20     # cm, 前方障礙停車距離（與超聲波安全煞車一致）
SMART_PATROL_REVERSE_SEC = 0.8    # 避障後退時間 (秒)
SMART_PATROL_TURN_SEC    = 1.0    # 避障轉彎時間 (秒)
STRAFE_SPEED             = 0.25   # 麥克拉姆側移速度
STRAFE_DURATION          = 0.5    # 側移持續時間 (秒)

# ============================================================
# 警報與事件回報
# ============================================================
ALERT_COOLDOWN = 30            # 事件回報後最短停留時間（秒），確保警報播完再進入下一階段
ALERT_SOUND = os.path.join(BASE_DIR, "參考資料/警報聲參考.wav")

# Telegram Bot（從環境變數讀取，請勿將 Token 寫入原始碼）
# 設定方式：export TELEGRAM_BOT_TOKEN="你的Token"  export TELEGRAM_CHAT_ID="你的ChatID"
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")

# ============================================================
# Flask 伺服器
# ============================================================
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5001

# ============================================================
# 自動巡檢 / 搜索邏輯
# ============================================================
PATROL_SPEED              = 0.3    # 前進速度 (0~1)
PATROL_TURN_DISTANCE      = 40     # 避障警戒距離 (cm)
PATROL_EMERGENCY_DISTANCE = 15     # 緊急停止距離 (cm)
PATROL_TURN_SPEED         = 0.35   # 轉彎速度（降低避免原地轉圈）
PATROL_TURN_DURATION      = 0.5    # 轉彎持續時間 (秒)（從1.2縮小）
PATROL_REVERSE_SPEED      = 0.35   # 後退速度
PATROL_REVERSE_DURATION   = 0.4    # 一般警戒後退時間 (秒)
PATROL_EMERGENCY_REVERSE  = 0.7    # 緊急距離後退時間 (秒)

PATROL_PAN_MIN     = -50    # 掃描最小角度
PATROL_PAN_MAX     = 50     # 掃描最大角度
PATROL_PAN_STEP    = 4      # 每步角度變化
PATROL_SWEEP_DELAY = 0.2    # 掃描間隔 (秒)
PATROL_SCAN_PAUSE  = 1.5    # 停車掃描時間 (秒)
PATROL_STUCK_THRESHOLD = 5  # 連續相同距離讀數次數 → 判定卡住
PATROL_MOVE_DURATION   = 2.5 # 每段前進持續時間 (秒)

# ============================================================
# 熱區記憶地圖 (Heat Map)
# ============================================================
HEAT_MAP_ENABLED          = True
HEAT_MAP_GRID_SIZE        = 40     # 格狀地圖大小 (40×40)
HEAT_MAP_CELL_CM          = 30     # 每格代表的實際距離 (cm)
HEAT_MAP_SCAN_RADIUS      = 2      # 掃描標記半徑 (格)
HEAT_MAP_CM_PER_SPEED_SEC = 30     # 速度 1.0 每秒移動距離 (cm) [需校準]
HEAT_MAP_RAD_PER_SPEED_SEC = 2.0   # 速度 1.0 每秒旋轉角度 (rad) [需校準]

# ============================================================
# 多人追蹤 (Multi-Person Tracking)
# ============================================================
TRACKER_MAX_LOST_FRAMES = 30       # 遺失超過此幀數則移除追蹤
TRACKER_MIN_IOU         = 0.25     # IoU 匹配最低閾值

# ============================================================
# 物件偵測擴展 (Object Detection Extension)
# ============================================================
OBJECT_DETECTION_ENABLED = True    # 僅 ONNX 後端有效，Hailo 不支援
RESCUE_OBJECT_CLASSES = {
    24: "backpack",     # 背包
    26: "handbag",      # 手提包
    28: "suitcase",     # 行李箱
    67: "cell phone",   # 手機
}

# ============================================================
# rPPG 生命跡象偵測（遠端光體積描記術）
# ============================================================
RPPG_ENABLED              = True
RPPG_BUFFER_SECONDS       = 4.0    # 滾動緩衝長度 (秒)
RPPG_UPDATE_INTERVAL      = 15     # 每 N 幀計算一次心率
RPPG_MIN_FACE_CONFIDENCE  = 0.40   # 臉部關鍵點最低信心度
RPPG_ROI_STABILITY_PX     = 20     # 幀間臉部最大位移 (px)
RPPG_BANDPASS_LOW_HZ      = 0.7    # 42 bpm
RPPG_BANDPASS_HIGH_HZ     = 3.5    # 210 bpm
RPPG_CONFIDENCE_MIN       = 0.5    # 結果可信度門檻

# ============================================================
# 低光增強 (CLAHE)
# ============================================================
CLAHE_MODE            = "off"      # "auto" / "on" / "off"（預設關閉，需要時再開啟）
CLAHE_AUTO_THRESHOLD  = 80         # 平均亮度低於此值自動啟用 (0-255)
CLAHE_CLIP_LIMIT      = 3.0        # CLAHE 對比限制
CLAHE_TILE_SIZE       = (8, 8)     # CLAHE 網格大小

# ============================================================
# 本地硬體 GPIO 相容性（集中初始化 pigpio，避免多模組重複建立連線）
# ============================================================
try:
    import gpiozero
    from gpiozero import Device
    try:
        from gpiozero.pins.pigpio import PiGPIOFactory
        Device.pin_factory = PiGPIOFactory()
        PIGPIO_OK = True
    except Exception:
        PIGPIO_OK = False
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    PIGPIO_OK = False
    print("⚠️ GPIO 不可用，硬體模組將進入模擬模式")

SERVO_DEFAULT_PAN  = SERVO_PAN_DEFAULT
SERVO_DEFAULT_TILT = SERVO_TILT_DEFAULT
