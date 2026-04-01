"""
搜救機器人 — 雙模型 AI 偵測模組 (V3 — Hailo NPU)
===================================================
人體偵測 (yolov8n) + 姿態/倒地偵測 (yolov8s-pose)

支援兩種推論後端（config.USE_HAILO 切換）：
  - ONNX (ultralytics) — USE_HAILO = False
  - Hailo HEF (NPU)   — USE_HAILO = True
"""

import cv2
import threading
import time
import logging
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# 多人追蹤
try:
    from tracker import PersonTracker
    TRACKER_AVAILABLE = True
except ImportError:
    TRACKER_AVAILABLE = False

try:
    from rppg import rPPGDetector
    RPPG_AVAILABLE = True
except ImportError:
    RPPG_AVAILABLE = False

logger = logging.getLogger("rescue.detector")

# ============================================================
# ONNX 後端 (ultralytics)
# ============================================================
try:
    from ultralytics import YOLO
    import ultralytics
    ultralytics.checks = lambda: None
    try:
        from ultralytics.utils import SETTINGS
        SETTINGS['sync'] = False
    except Exception:
        pass
    import os
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# ============================================================
# Hailo 後端 (hailort)
# ============================================================
HAILO_AVAILABLE = False
try:
    from hailo_platform import (
        HEF, VDevice, HailoStreamInterface,
        InferVStreams, ConfigureParams,
        InputVStreamParams, OutputVStreamParams,
        FormatType
    )
    HAILO_AVAILABLE = True
    logger.info("HailoRT SDK 載入成功")
except ImportError:
    pass

import config


@dataclass
class DetectionResult:
    """單幀偵測結果"""
    person_count: int = 0
    persons: List[dict] = field(default_factory=list)
    fallen_count: int = 0
    pose_anomaly_score: float = 0.0
    wave_detected: bool = False
    annotated_frame: np.ndarray = None
    timestamp: float = 0.0
    # 多人追蹤
    tracks: list = field(default_factory=list)
    unique_person_count: int = 0
    unreported_count: int = 0
    # 物件偵測擴展
    objects: List[dict] = field(default_factory=list)
    # 眼睛狀態
    eye_state: str = "UNKNOWN"   # "OPEN" / "CLOSED" / "UNKNOWN"
    # rPPG 生命跡象
    heart_rate_bpm: float = -1.0           # -1 = 未量測
    rppg_confidence: float = 0.0           # 0~1 信號品質
    rppg_signal_quality: str = "UNKNOWN"   # "GOOD" / "WEAK" / "UNKNOWN"


# 人體骨架連線定義
SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

FALLEN_CONFIRM_FRAMES = 3   # 連續 3 幀確認，防止單幀誤判（坐著偶爾框變寬）


# ============================================================
# 姿態判定函數
# ============================================================
def _get_center(p1, p2, min_conf=0.3):
    """取得兩個關鍵點的中心座標（至少一個信心度足夠即可）"""
    if p1[2] > min_conf and p2[2] > min_conf:
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    elif p1[2] > min_conf:
        return (p1[0], p1[1])
    elif p2[2] > min_conf:
        return (p2[0], p2[1])
    return None


def is_fallen(keypoints, person_bbox=None) -> bool:
    """
    判斷人員是否倒地（收緊條件，減少坐姿誤判）。
    核心邏輯：邊界框寬高比 > 1.8 才算倒地（原 1.5 太寬鬆）。
    骨架輔助：肩-臀水平距離必須 > 60px 且垂直 < 水平的 0.35 倍。
    """
    min_conf = 0.3

    # ═══ 主判斷：邊界框寬高比 ═══
    if person_bbox:
        bw = person_bbox["x2"] - person_bbox["x1"]
        bh = person_bbox["y2"] - person_bbox["y1"]
        if bh > 0 and bw > 0:
            bbox_ratio = bw / bh
            if bbox_ratio > 1.8:   # 寬遠大於高 → 明確橫躺（原 1.5 太敏感）
                return True
            if bbox_ratio < 0.8:   # 高 >> 寬 → 直立或坐著 → 排除
                return False

    # ═══ 輔助：骨架幾何（框比例模糊 0.8~1.8 時）═══
    shoulder = _get_center(keypoints[5], keypoints[6], min_conf)
    hip = _get_center(keypoints[11], keypoints[12], min_conf)

    if shoulder and hip:
        vertical = abs(hip[1] - shoulder[1])
        horizontal = abs(hip[0] - shoulder[0])
        # 收緊：水平距離要 > 60px 且垂直 < 水平的 0.35（原 40px / 0.5 太鬆）
        if horizontal > 60 and vertical < horizontal * 0.35:
            # 額外排除坐姿：如果臀部明顯低於肩膀，是坐著不是躺著
            if hip[1] > shoulder[1] + 30:
                return False
            return True

    # 腳踝與肩膀在相近高度（全身平躺，收緊門檻）
    ankle = _get_center(keypoints[15], keypoints[16], min_conf)
    if shoulder and ankle:
        height_diff = abs(shoulder[1] - ankle[1])
        body_width = abs(shoulder[0] - ankle[0])
        # 收緊：寬 > 80px 且高差 < 寬的 0.25（原 50px / 0.3）
        if body_width > 80 and height_diff < body_width * 0.25:
            return True

    return False


def is_crouching(keypoints) -> bool:
    """判斷人員是否蜷縮。"""
    min_conf = 0.3
    core_indices = [5, 6, 11, 12, 13, 14, 15, 16]
    valid_pts = [keypoints[i] for i in core_indices if keypoints[i][2] > min_conf]
    if len(valid_pts) < 4:
        return False

    xs = [p[0] for p in valid_pts]
    ys = [p[1] for p in valid_pts]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)

    if h <= 0 or w <= 0:
        return False

    aspect = w / h
    if 0.6 < aspect < 1.6 and h < 200:
        shoulder = None
        hip = None
        if keypoints[5][2] > min_conf and keypoints[6][2] > min_conf:
            shoulder = ((keypoints[5][1] + keypoints[6][1]) / 2)
        if keypoints[11][2] > min_conf and keypoints[12][2] > min_conf:
            hip = ((keypoints[11][1] + keypoints[12][1]) / 2)
        if shoulder and hip:
            torso_vertical = abs(hip - shoulder)
            if torso_vertical < 80:
                return True
    return False


def is_distressed(keypoints, person_bbox=None) -> bool:
    """
    判斷人員是否呈現疑似不適姿勢（介於正常站立與完全倒地之間）。
    只保留最可靠的判斷：邊界框寬高比 1.4~1.8（接近倒地但未完全橫躺）。
    肩膀傾斜太容易誤判，已移除。
    """
    if not person_bbox:
        return False

    bw = person_bbox["x2"] - person_bbox["x1"]
    bh = person_bbox["y2"] - person_bbox["y1"]
    if bh <= 0:
        return False

    bbox_ratio = bw / bh
    # 1.4~1.8：半躺/側靠，比倒地門檻(1.8)低但比正常(~0.5-1.0)高
    if 1.4 < bbox_ratio <= 1.8:
        return True

    return False


def detect_eye_state(frame, keypoints) -> str:
    """
    估計眼睛開閉狀態。
    用 Laplacian 邊緣方差分析眼部區域：
      開眼 → 虹膜/瞳孔邊緣多 → 方差高
      閉眼 → 皮膚平滑 → 方差低
    回傳 "OPEN" / "CLOSED" / "UNKNOWN"
    """
    min_conf = 0.3
    nose = keypoints[0]
    left_eye = keypoints[1]
    right_eye = keypoints[2]

    if nose[2] < min_conf:
        return "UNKNOWN"

    h, w = frame.shape[:2]
    results = []

    for eye_kp in [left_eye, right_eye]:
        if eye_kp[2] < min_conf:
            continue
        ex, ey = int(eye_kp[0]), int(eye_kp[1])

        # 根據臉部大小估算眼睛區域
        face_ref = max(10, abs(eye_kp[0] - nose[0]) * 0.8)
        hw = max(5, int(face_ref * 0.5))
        hh = max(3, int(face_ref * 0.25))

        x1 = max(0, ex - hw)
        x2 = min(w, ex + hw)
        y1 = max(0, ey - hh)
        y2 = min(h, ey + hh)

        if x2 - x1 < 4 or y2 - y1 < 3:
            continue

        crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        edge_var = cv2.Laplacian(crop, cv2.CV_64F).var()

        # 開眼邊緣方差高（虹膜/瞳孔）、閉眼方差低（平滑皮膚）
        results.append("OPEN" if edge_var > 100 else "CLOSED")

    if not results:
        return "UNKNOWN"
    if all(r == "CLOSED" for r in results):
        return "CLOSED"
    return "OPEN"


def compute_pose_anomaly_score(keypoints_list, fallen_ids: set,
                                crouching_ids: set, distressed_ids: set) -> float:
    """計算姿態異常分數 (0~1)。"""
    if not keypoints_list:
        return 0.0
    max_score = 0.0
    for kps in keypoints_list:
        kid = id(kps)
        if kid in fallen_ids:
            max_score = max(max_score, 1.0)
        elif kid in crouching_ids:
            max_score = max(max_score, 0.6)
        elif kid in distressed_ids:
            max_score = max(max_score, 0.4)
    return max_score


# ============================================================
# 揮手偵測器
# ============================================================
class WaveDetector:
    """
    揮手手勢偵測器（適用於倒地/站立者）。
    追蹤任何可見手腕位置的擺動模式（水平 + 垂直兩軸皆檢測）。

    條件：
      1. 至少一隻手腕可見（信心度 > 0.3）
      2. 在最近 N 幀中，手腕位移振幅 ≥ min_amplitude 像素
      3. 方向變換次數 ≥ min_changes（左→右 或 上→下 算一次）
    """

    def __init__(self, history_len: int = 20, min_changes: int = 2,
                 min_amplitude: int = 25):
        self._history_x: deque = deque(maxlen=history_len)
        self._history_y: deque = deque(maxlen=history_len)
        self._min_changes = min_changes
        self._min_amplitude = min_amplitude
        self._last_wave_time = 0.0
        self._cooldown = 5.0
        self._init_time = time.time()  # 啟動後 3 秒內不偵測，避免開機誤觸

    def update(self, keypoints_list: list) -> bool:
        """每幀呼叫。回傳 True = 偵測到揮手。"""
        # 啟動冷卻期
        if time.time() - self._init_time < 3.0:
            return False

        wx, wy = self._find_best_wrist(keypoints_list)
        self._history_x.append(wx)
        self._history_y.append(wy)

        if wx is None:
            return False

        # 需要至少 6 幀有效資料
        valid_x = [v for v in self._history_x if v is not None]
        valid_y = [v for v in self._history_y if v is not None]
        if len(valid_x) < 6:
            return False

        # 檢查水平軸和垂直軸，取較大的振幅
        amp_x = max(valid_x) - min(valid_x)
        amp_y = max(valid_y) - min(valid_y)

        # 選擇振幅較大的軸來分析方向變換
        if amp_x >= amp_y and amp_x >= self._min_amplitude:
            changes = self._count_direction_changes(valid_x)
        elif amp_y >= self._min_amplitude:
            changes = self._count_direction_changes(valid_y)
        else:
            return False

        if changes >= self._min_changes:
            now = time.time()
            if now - self._last_wave_time > self._cooldown:
                self._last_wave_time = now
                axis = "水平" if amp_x >= amp_y else "垂直"
                logger.info(f"👋 偵測到揮手！({axis} 振幅={max(amp_x,amp_y):.0f}px, 方向變換={changes}次)")
                return True

        return False

    @staticmethod
    def _count_direction_changes(values: list) -> int:
        changes = 0
        prev_dir = 0
        for i in range(1, len(values)):
            diff = values[i] - values[i - 1]
            if abs(diff) < 5:
                continue
            cur_dir = 1 if diff > 0 else -1
            if prev_dir != 0 and cur_dir != prev_dir:
                changes += 1
            prev_dir = cur_dir
        return changes

    @staticmethod
    def _find_best_wrist(keypoints_list: list):
        """
        找出信心度最高的可見手腕 (x, y) 座標。
        不要求手腕高於肩膀，適用於倒地者揮手。
        """
        if not keypoints_list:
            return None, None

        best_x, best_y = None, None
        best_conf = 0.0

        for kps in keypoints_list:
            for w_idx in [9, 10]:  # 左手腕、右手腕
                wrist = kps[w_idx]
                if wrist[2] > 0.3 and wrist[2] > best_conf:
                    best_conf = float(wrist[2])
                    best_x = float(wrist[0])
                    best_y = float(wrist[1])

        return best_x, best_y


# ============================================================
# Hailo NPU 推理引擎
# ============================================================
class HailoInferenceEngine:
    """
    封裝 HailoRT HEF 推理（持久 pipeline，最大化 NPU 吞吐量）。
    關鍵優化：InferVStreams pipeline 僅建立一次並持續復用，
    避免每幀重複分配 DMA 緩衝區和配置 PCIe 通道。
    """

    def __init__(self, hef_path: str, input_size: int = 640):
        self._hef_path = hef_path
        self._input_size = input_size
        self._owns_vdevice = False
        self._vdevice = None
        self._network_group = None
        self._activated_ng = None
        self._pipeline = None          # 持久 InferVStreams pipeline
        self._input_name = None        # 快取輸入名稱
        self._input_vstream_info = None
        self._output_vstream_info = None
        self._input_params = None
        self._output_params = None
        self._configured = False
        self._lock = threading.Lock()

    def configure(self, shared_vdevice=None):
        """
        初始化並配置 HEF + 建立持久推理 pipeline。
        shared_vdevice: 傳入已建立的 VDevice 來共享，None 則自建。
        """
        try:
            self._hef = HEF(self._hef_path)

            if shared_vdevice is not None:
                self._vdevice = shared_vdevice
                self._owns_vdevice = False
            else:
                self._vdevice = VDevice()
                self._owns_vdevice = True

            configure_params = ConfigureParams.create_from_hef(
                hef=self._hef, interface=HailoStreamInterface.PCIe
            )
            self._network_group = self._vdevice.configure(self._hef, configure_params)[0]

            self._input_vstream_info = self._hef.get_input_vstream_infos()
            self._output_vstream_info = self._hef.get_output_vstream_infos()

            self._input_params = InputVStreamParams.make(
                self._network_group,
                format_type=FormatType.UINT8
            )
            self._output_params = OutputVStreamParams.make(
                self._network_group,
                format_type=FormatType.FLOAT32
            )

            # 啟動 Network Group
            self._activated_ng = self._network_group.activate()
            self._activated_ng.__enter__()

            # 建立持久 pipeline（核心優化：不再每幀重建）
            self._pipeline = InferVStreams(
                self._network_group,
                self._input_params,
                self._output_params
            )
            self._pipeline.__enter__()

            info = self._input_vstream_info[0]
            self._input_size = info.shape[1]
            self._input_name = info.name
            self._configured = True

            logger.info(
                f"Hailo HEF 就緒: {self._hef_path} | "
                f"輸入: {info.shape} | "
                f"輸出: {len(self._output_vstream_info)} 個 vstream | "
                f"持久 pipeline 已建立"
            )
        except Exception as e:
            logger.error(f"Hailo HEF 配置失敗: {self._hef_path} | {e}")
            self._configured = False

    @property
    def is_ready(self):
        return self._configured

    def infer(self, frame: np.ndarray) -> dict:
        """
        執行推理（使用持久 pipeline，零開銷呼叫）。
        回傳 {output_name: np.ndarray}。
        """
        if not self._configured:
            return {}
        try:
            # 前處理：resize + BGR→RGB（INTER_NEAREST 比 INTER_LINEAR 快 ~30%）
            resized = cv2.resize(frame, (self._input_size, self._input_size),
                                 interpolation=cv2.INTER_NEAREST)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            input_data = np.expand_dims(rgb, axis=0)

            with self._lock:
                results = self._pipeline.infer({self._input_name: input_data})
            return results
        except Exception as e:
            logger.error(f"Hailo 推理錯誤: {e}")
            # pipeline 可能損壞，嘗試重建
            self._try_rebuild_pipeline()
            return {}

    def _try_rebuild_pipeline(self):
        """推理失敗時嘗試重建 pipeline"""
        try:
            if self._pipeline:
                try:
                    self._pipeline.__exit__(None, None, None)
                except Exception:
                    pass
            self._pipeline = InferVStreams(
                self._network_group,
                self._input_params,
                self._output_params
            )
            self._pipeline.__enter__()
            logger.info("Hailo pipeline 重建成功")
        except Exception as e:
            logger.error(f"Hailo pipeline 重建失敗: {e}")
            self._configured = False

    def close(self):
        """釋放資源。"""
        if self._pipeline:
            try:
                self._pipeline.__exit__(None, None, None)
            except Exception:
                pass
            self._pipeline = None
        if self._activated_ng:
            try:
                self._activated_ng.__exit__(None, None, None)
            except Exception:
                pass
            self._activated_ng = None
        if self._owns_vdevice and self._vdevice:
            try:
                self._vdevice.release()
            except Exception:
                pass
            self._vdevice = None
        self._configured = False


# ============================================================
# YOLOv8-pose Hailo 後處理（完整 DFL + NMS）
# ============================================================
# Hailo yolov8s_pose 輸出 9 個 vstream = 3 尺度 × 3 分支:
#   bbox DFL: (H, W, 64) — 4 座標 × 16-bin 分布
#   score:    (H, W, 1)  — 人體類別分數（Hailo 單類可能全為 0）
#   keypoints:(H, W, 51) — 17 × 3 (x_offset, y_offset, visibility)
# ============================================================

def _nms(boxes, scores, iou_thresh=0.45):
    """簡易 NMS。"""
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]; y1 = boxes[:, 1]
    x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-6)
        order = rest[iou < iou_thresh]
    return keep


def yolov8_pose_hailo_postprocess(outputs: dict,
                                   frame_h: int, frame_w: int,
                                   kp_conf_thresh: float = 0.55,
                                   min_kp_count: int = 6,
                                   min_box_px: int = 30,
                                   iou_thresh: float = 0.45):
    """
    解析 Hailo yolov8s_pose 原始輸出，回傳 (persons, keypoints)。
    使用 numpy 向量化避免 Python 逐格迴圈。
    """
    INPUT_SIZE = 640

    # 依空間尺寸分組
    bbox_maps = {}   # (h,w) → (h,w,64)
    score_maps = {}  # (h,w) → (h,w)
    kps_maps = {}    # (h,w) → (h,w,51)

    for name, tensor in outputs.items():
        data = tensor.squeeze()
        if data.ndim == 2 and data.shape[-1] not in (51, 64):
            # (H, W) = objectness score
            score_maps[data.shape[:2]] = data
        elif data.ndim == 3 and data.shape[-1] == 64:
            bbox_maps[data.shape[:2]] = data
        elif data.ndim == 3 and data.shape[-1] == 51:
            kps_maps[data.shape[:2]] = data

    all_boxes = []
    all_scores = []
    all_kps_list = []
    arange16 = np.arange(16, dtype=np.float32)

    for key in sorted(bbox_maps.keys(), key=lambda k: -k[0]):
        gh, gw = key
        stride = INPUT_SIZE / gh
        sx = frame_w / INPUT_SIZE
        sy = frame_h / INPUT_SIZE

        bbox_data = bbox_maps[key]   # (gh, gw, 64)
        kps_data = kps_maps.get(key)  # (gh, gw, 51) or None

        if kps_data is None:
            continue

        N = gh * gw

        # 建立網格
        cx_grid, cy_grid = np.meshgrid(np.arange(gw, dtype=np.float32),
                                        np.arange(gh, dtype=np.float32))
        cx_flat = cx_grid.flatten() + 0.5   # (N,)
        cy_flat = cy_grid.flatten() + 0.5

        # ---- DFL bbox 解碼 (向量化) ----
        bbox_flat = bbox_data.reshape(N, 4, 16)                       # (N, 4, 16)
        bbox_exp = np.exp(bbox_flat - bbox_flat.max(axis=2, keepdims=True))
        bbox_softmax = bbox_exp / bbox_exp.sum(axis=2, keepdims=True)  # (N, 4, 16)
        distances = (bbox_softmax * arange16).sum(axis=2)              # (N, 4)

        x1 = (cx_flat - distances[:, 0]) * stride * sx
        y1 = (cy_flat - distances[:, 1]) * stride * sy
        x2 = (cx_flat + distances[:, 2]) * stride * sx
        y2 = (cy_flat + distances[:, 3]) * stride * sy

        # ---- Keypoint 解碼 (向量化) ----
        kps_flat = kps_data.reshape(N, 17, 3)
        kp_x = (kps_flat[:, :, 0] * 2.0 + cx_flat.reshape(-1, 1) - 0.5) * stride * sx
        kp_y = (kps_flat[:, :, 1] * 2.0 + cy_flat.reshape(-1, 1) - 0.5) * stride * sy
        kp_vis = 1.0 / (1.0 + np.exp(-kps_flat[:, :, 2].clip(-20, 20)))  # sigmoid

        # ---- Object Score (向量化) ----
        score_data = score_maps.get(key)
        if score_data is not None:
            score_flat = score_data.reshape(N)
            # 如果 logits 有負數代表尚未 sigmoid，若都是 0~1 則已是機率
            if score_flat.min() < 0 or score_flat.max() > 1.0:
                score_flat = 1.0 / (1.0 + np.exp(-score_flat.clip(-20, 20)))
            obj_score = score_flat
        else:
            obj_score = np.ones(N, dtype=np.float32)

        # ---- 過濾：框大小 + keypoint 品質 + obj score ----
        w_box = x2 - x1
        h_box = y2 - y1
        kp_valid_mask = kp_vis > kp_conf_thresh               # (N, 17)
        kp_valid_count = kp_valid_mask.sum(axis=1)             # (N,)

        valid = (w_box > min_box_px) & (h_box > min_box_px) & \
                (kp_valid_count >= min_kp_count) & (obj_score > 0.5)

        indices = np.where(valid)[0]
        if len(indices) == 0:
            continue

        for i in indices:
            all_boxes.append([x1[i], y1[i], x2[i], y2[i]])
            all_scores.append(float(obj_score[i]))
            kps_result = np.stack([kp_x[i], kp_y[i], kp_vis[i]], axis=1)
            all_kps_list.append(kps_result.astype(np.float32))

    if not all_boxes:
        return [], []

    # NMS
    boxes_arr = np.array(all_boxes, dtype=np.float32)
    scores_arr = np.array(all_scores, dtype=np.float32)
    keep = _nms(boxes_arr, scores_arr, iou_thresh)

    persons = []
    result_kps = []
    for i in keep:
        bx = boxes_arr[i]
        persons.append({
            "x1": int(max(0, bx[0])),
            "y1": int(max(0, bx[1])),
            "x2": int(min(frame_w, bx[2])),
            "y2": int(min(frame_h, bx[3])),
            "conf": round(float(scores_arr[i]), 2)
        })
        result_kps.append(all_kps_list[i])

    return persons, result_kps


# ============================================================
# 主偵測器
# ============================================================
class RescueDetector:
    """
    搜救用雙模型偵測器（支援 ONNX / Hailo 雙後端）
    """

    def __init__(self):
        logger.info("初始化搜救 AI 偵測器...")
        self._general_model = None
        self._pose_model = None
        self._hailo_person = None
        self._hailo_pose = None
        self._use_hailo = False
        self._frame_count = 0
        self._latest_result = DetectionResult()
        self._lock = threading.Lock()
        self._fallen_consecutive = 0
        self._all_keypoints: List = []
        self._wave_detector = WaveDetector()
        self._tracker = PersonTracker() if TRACKER_AVAILABLE else None
        self._rppg = rPPGDetector() if (RPPG_AVAILABLE and config.RPPG_ENABLED) else None

        if config.USE_HAILO and HAILO_AVAILABLE:
            self._init_hailo()
        elif YOLO_AVAILABLE:
            self._init_onnx()
        else:
            logger.warning("無可用推理後端（YOLO 和 Hailo 均不可用）")

    def _init_hailo(self):
        """
        初始化 Hailo NPU 後端。
        Hailo-8L 一次只能啟動一個 Network Group，
        因此只載入 pose 模型（同時提供人體偵測 + 骨架姿態）。
        """
        logger.info("初始化 Hailo NPU 推理後端...")
        try:
            self._hailo_pose = HailoInferenceEngine(config.HEF_POSE_MODEL)
            self._hailo_pose.configure()

            if self._hailo_pose.is_ready:
                self._use_hailo = True
                logger.info("Hailo NPU 姿態模型載入成功（同時用於人體偵測）")
            else:
                logger.warning("Hailo 配置失敗，回退到 ONNX")
                self._init_onnx()
        except Exception as e:
            logger.error(f"Hailo 初始化失敗: {e}，回退到 ONNX")
            self._init_onnx()

    def _init_onnx(self):
        """初始化 ONNX/ultralytics 後端。"""
        if not YOLO_AVAILABLE:
            logger.warning("ultralytics 未安裝，AI 偵測功能停用")
            return
        try:
            logger.info(f"載入 ONNX 通用模型: {config.YOLO_GENERAL_MODEL}")
            self._general_model = YOLO(config.YOLO_GENERAL_MODEL, task='detect')
            logger.info(f"載入 ONNX 姿態模型: {config.YOLO_POSE_MODEL}")
            self._pose_model = YOLO(config.YOLO_POSE_MODEL, task='pose')
            logger.info("ONNX 雙模型載入完成")
        except Exception as e:
            logger.error(f"ONNX 模型載入失敗: {e}")

    @property
    def is_loaded(self) -> bool:
        if self._use_hailo:
            return self._hailo_pose is not None and self._hailo_pose.is_ready
        return self._general_model is not None

    @property
    def backend_name(self) -> str:
        return "Hailo-8L" if self._use_hailo else "ONNX"

    @property
    def latest_result(self) -> DetectionResult:
        with self._lock:
            return self._latest_result

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> DetectionResult:
        """對單幀執行 AI 偵測。"""
        if frame is None:
            return self._latest_result

        self._frame_count += 1
        now = time.time()
        h, w = frame.shape[:2]

        # Hailo: 一次 pose 推理同時取得 persons + keypoints
        # ONNX: 分兩次推理（含物件偵測擴展）
        objects = []
        if self._use_hailo:
            persons, all_kps = self._detect_all_hailo(frame, h, w)
        else:
            persons, objects = self._detect_persons_onnx(frame)
            all_kps = []

        # 多人追蹤
        if self._tracker:
            try:
                self._tracker.update(persons)
            except Exception as e:
                logger.debug(f"追蹤器錯誤: {e}")

        with self._lock:
            self._latest_result.persons = persons
            self._latest_result.person_count = len(persons)
            self._latest_result.objects = objects
            if self._tracker:
                self._latest_result.tracks = self._tracker.get_tracks_info()
                self._latest_result.unique_person_count = self._tracker.total_unique_persons
                self._latest_result.unreported_count = self._tracker.get_unreported_count()

        # (2) 姿態分析
        fallen_list: List = []
        crouching_list: List = []
        distressed_list: List = []

        if not self._use_hailo and len(persons) > 0:
            all_kps = self._detect_pose_onnx(frame)

        self._all_keypoints = all_kps

        for i, kps in enumerate(all_kps):
            bbox = persons[i] if i < len(persons) else None
            if is_fallen(kps, bbox):
                fallen_list.append(kps)
            elif is_crouching(kps):
                crouching_list.append(kps)
            elif is_distressed(kps, bbox):
                distressed_list.append(kps)

        # 倒地確認（單幀即確認，狀態機有獨立 ANOMALY 2 秒確認期）
        if fallen_list:
            self._fallen_consecutive += 1
            if self._fallen_consecutive >= FALLEN_CONFIRM_FRAMES:
                if self._fallen_consecutive == FALLEN_CONFIRM_FRAMES:
                    logger.warning(f"確認倒地！({len(fallen_list)} 人)")
            else:
                fallen_list = []
        else:
            self._fallen_consecutive = 0

        fallen_ids = {id(kps) for kps in fallen_list}
        crouching_ids = {id(kps) for kps in crouching_list}
        distressed_ids = {id(kps) for kps in distressed_list}
        pose_score = compute_pose_anomaly_score(all_kps, fallen_ids, crouching_ids, distressed_ids)

        # (2.5) 揮手偵測
        wave = self._wave_detector.update(all_kps)

        # (2.6) 眼睛開閉偵測（距離夠近時才有意義）
        eye_state = "UNKNOWN"
        if all_kps:
            try:
                eye_state = detect_eye_state(frame, all_kps[0])
            except Exception:
                pass

        # (2.7) rPPG 心率偵測（需臉部關鍵點 + 穩定 ROI）
        rppg_result = {"bpm": -1.0, "confidence": 0.0, "signal_quality": "UNKNOWN"}
        if self._rppg and all_kps:
            try:
                rppg_result = self._rppg.process_frame(frame, all_kps[0][:5], now)
            except Exception:
                pass

        with self._lock:
            self._latest_result.fallen_count = len(fallen_list) + len(distressed_list)
            self._latest_result.pose_anomaly_score = pose_score
            self._latest_result.wave_detected = wave
            self._latest_result.eye_state = eye_state
            self._latest_result.heart_rate_bpm = rppg_result.get("bpm", -1.0)
            self._latest_result.rppg_confidence = rppg_result.get("confidence", 0.0)
            self._latest_result.rppg_signal_quality = rppg_result.get("signal_quality", "UNKNOWN")

        # (3) 繪製標註（所有參數直接傳入，不在繪製中取鎖，避免凍結）
        annotated = frame.copy()
        annotated = self._draw_annotations(
            annotated, persons, fallen_list, crouching_list,
            getattr(self, '_all_keypoints', []),
            distressed_list=distressed_list,
            objects=objects,
            eye_state=eye_state,
            rppg=rppg_result,
        )

        with self._lock:
            self._latest_result.annotated_frame = annotated
            self._latest_result.timestamp = now

        return self._latest_result

    # ------------------------------------------------------------------
    # Hailo 後端：單一 pose 模型同時取 persons + keypoints
    # ------------------------------------------------------------------
    def _detect_all_hailo(self, frame, h, w):
        """
        用 pose 模型一次推理，回傳 (persons, keypoints)。
        persons 從 keypoints 的有效點推算 bounding box。
        """
        if not self._hailo_pose or not self._hailo_pose.is_ready:
            return [], []

        outputs = self._hailo_pose.infer(frame)
        if not outputs:
            return [], []

        # DEBUG: 每 30 幀印出輸出格式（調試用，確認後可移除）
        if self._frame_count % 30 == 1:
            for name, tensor in outputs.items():
                data = tensor.squeeze()
                logger.debug(f"[HAILO DEBUG] output '{name}': shape={data.shape}, "
                             f"dtype={data.dtype}, min={data.min():.4f}, max={data.max():.4f}")

        # 使用新的向量化後處理器
        persons, all_kps = yolov8_pose_hailo_postprocess(
            outputs, h, w,
            kp_conf_thresh=0.4,    # 降低（躺平時關鍵點信心度較低）
            min_kp_count=4,        # 降低（躺平時可見關鍵點較少）
            min_box_px=20,         # 降低（遠處的人框較小）
            iou_thresh=0.45
        )

        return persons, all_kps

    # ------------------------------------------------------------------
    # ONNX 後端
    # ------------------------------------------------------------------
    def _detect_persons_onnx(self, frame) -> tuple:
        """回傳 (persons, objects)"""
        if not self._general_model:
            return [], []
        persons = []
        objects = []

        # 建立偵測類別清單
        detect_classes = [config.PERSON_CLASS_ID]
        if config.OBJECT_DETECTION_ENABLED:
            detect_classes.extend(config.RESCUE_OBJECT_CLASSES.keys())

        try:
            results = self._general_model(
                frame, conf=config.GENERAL_CONFIDENCE,
                classes=detect_classes,
                imgsz=config.YOLO_GENERAL_INFER_SIZE, verbose=False
            )
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cls_id = int(box.cls[0])
                    conf = round(float(box.conf[0]), 2)

                    if cls_id == config.PERSON_CLASS_ID:
                        persons.append({
                            "x1": int(x1), "y1": int(y1),
                            "x2": int(x2), "y2": int(y2),
                            "conf": conf
                        })
                    elif config.OBJECT_DETECTION_ENABLED and cls_id in config.RESCUE_OBJECT_CLASSES:
                        objects.append({
                            "x1": int(x1), "y1": int(y1),
                            "x2": int(x2), "y2": int(y2),
                            "conf": conf,
                            "class_id": cls_id,
                            "class_name": config.RESCUE_OBJECT_CLASSES[cls_id],
                        })
        except Exception as e:
            logger.error(f"ONNX 偵測錯誤: {e}")
        return persons, objects

    def _detect_pose_onnx(self, frame) -> List[np.ndarray]:
        if not self._pose_model:
            return []
        all_kps = []
        try:
            results = self._pose_model(
                frame, conf=config.POSE_CONFIDENCE,
                imgsz=config.YOLO_POSE_INFER_SIZE, verbose=False
            )
            for r in results:
                if r.keypoints is not None:
                    for kp in r.keypoints.data:
                        all_kps.append(kp.cpu().numpy())
        except Exception as e:
            logger.error(f"ONNX 姿態偵測錯誤: {e}")
        return all_kps

    # ------------------------------------------------------------------
    # 繪製標註
    # ------------------------------------------------------------------
    @property
    def tracker(self):
        """暴露追蹤器供外部存取（標記已通報等）"""
        return self._tracker

    def _draw_annotations(self, frame, persons,
                           fallen_kps=None, crouching_kps=None, all_kps=None,
                           distressed_list=None, objects=None,
                           eye_state="UNKNOWN", rppg=None):
        fallen_kps = fallen_kps or []
        crouching_kps = crouching_kps or []
        distressed_list = distressed_list or []
        all_kps = all_kps or []
        objects = objects or []

        # 預建 id 集合，O(1) 查詢取代 O(n) array_equal
        fallen_id_set = {id(kps) for kps in fallen_kps}
        crouching_id_set = {id(kps) for kps in crouching_kps}
        distressed_id_set = {id(kps) for kps in distressed_list}

        for kps in all_kps:
            is_this_fallen = id(kps) in fallen_id_set
            is_this_crouching = id(kps) in crouching_id_set
            is_this_distressed = id(kps) in distressed_id_set
            is_abnormal = is_this_fallen or is_this_crouching or is_this_distressed

            if is_abnormal:
                color = (0, 0, 255)       # 紅色：所有異常姿態
            else:
                color = (0, 255, 128)     # 綠色：正常

            for i, j in SKELETON:
                if kps[i][2] > 0.3 and kps[j][2] > 0.3:
                    # 手臂骨架用更粗的線條 + 亮色突顯
                    is_arm = (i in (5, 6, 7, 8, 9, 10) and j in (5, 6, 7, 8, 9, 10))
                    line_color = (0, 255, 255) if is_arm else color
                    line_w = 3 if is_arm else 2
                    cv2.line(frame,
                             (int(kps[i][0]), int(kps[i][1])),
                             (int(kps[j][0]), int(kps[j][1])),
                             line_color, line_w)
            valid_pts = []
            for idx, kp in enumerate(kps):
                if kp[2] > 0.3:
                    pt = (int(kp[0]), int(kp[1]))
                    valid_pts.append(pt)
                    # 手腕用大圓圈 + 標籤突顯
                    if idx in (9, 10):
                        cv2.circle(frame, pt, 8, (0, 255, 255), 2)
                        cv2.circle(frame, pt, 3, (0, 255, 255), -1)
                    else:
                        cv2.circle(frame, pt, 4, color, -1)

            if is_abnormal and valid_pts:
                top_pt = min(valid_pts, key=lambda p: p[1])
                if is_this_fallen:
                    label = "FALLEN!"
                elif is_this_crouching:
                    label = "CROUCHING"
                else:
                    label = "DISTRESSED"
                cv2.putText(frame, label, (top_pt[0] - 35, top_pt[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        for p in persons:
            cv2.rectangle(frame, (p["x1"], p["y1"]), (p["x2"], p["y2"]), (0, 255, 0), 2)
            cv2.putText(frame, f"Person {p['conf']:.0%}", (p["x1"], p["y1"] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # OSD 資訊（不取鎖，直接用已知數值）
        n_persons = len(persons)
        n_abnormal = len(fallen_kps) + len(crouching_kps) + len(distressed_list)
        backend = "NPU" if self._use_hailo else "ONNX"
        eye_tag = f"  Eye:{eye_state}" if eye_state != "UNKNOWN" else ""
        info = f"[{backend}] P:{n_persons}  F:{n_abnormal}  {eye_tag}"
        cv2.putText(frame, info, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if eye_state == "CLOSED":
            cv2.putText(frame, "EYES CLOSED!", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # rPPG 心率顯示
        if rppg and rppg.get("valid"):
            bpm = rppg["bpm"]
            conf = rppg["confidence"]
            hr_color = (0, 255, 0) if 50 <= bpm <= 100 else (0, 0, 255)
            cv2.putText(frame, f"HR: {bpm:.0f} BPM ({conf:.0%})", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, hr_color, 2)
        elif rppg and rppg.get("signal_quality") == "WEAK":
            cv2.putText(frame, "HR: measuring...", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)

        # 繪製追蹤 ID
        if self._tracker:
            for track in self._tracker.active_tracks:
                b = track.bbox
                label = f"#{track.track_id}"
                if track.reported:
                    label += " [R]"
                cv2.putText(frame, label, (b["x1"], b["y2"] + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

        # 繪製偵測到的災區物件（直接用傳入參數，不取鎖）
        for obj in objects:
            obj_color = (255, 165, 0)
            cv2.rectangle(frame, (obj["x1"], obj["y1"]), (obj["x2"], obj["y2"]),
                          obj_color, 2)
            label = f"{obj['class_name']} {obj['conf']:.0%}"
            cv2.putText(frame, label, (obj["x1"], obj["y1"] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, obj_color, 2)

        return frame

    def cleanup(self):
        logger.info("釋放 AI 模型資源...")
        self._general_model = None
        self._pose_model = None
        if self._hailo_person:
            self._hailo_person.close()
        if self._hailo_pose:
            self._hailo_pose.close()
