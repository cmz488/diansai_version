"""矩形目标检测与位姿估计。

管线：BGR → remap去畸变(alpha=1) → LAB → inRange → 形态学 → Canny → 轮廓筛选 → PnP → 最优目标
角点反投影回畸变坐标输出，可直接在原图上绘制。
"""

import cv2
import numpy as np
from typing import Optional, Tuple

# ============================================================
# 目标物理尺寸（米）
# ============================================================
RECT_WIDTH = 0.179
RECT_HEIGHT = 0.108
ASPECT_RATIO = RECT_WIDTH / RECT_HEIGHT  # ≈ 1.657

# ============================================================
# LAB 色彩阈值（MVLab 格式：L[0-100], A[-128,127], B[-128,127]）
# ============================================================
MVLAB_PARAM = [7, 32, -13, 9, -13, 21]

# ============================================================
# 形态学
# ============================================================
MORPH_KERNEL = 2  # 核半尺寸（实际核 = 2K+1 = 5×5）

# ============================================================
# Canny 边缘
# ============================================================
CANNY_LOW = 50
CANNY_HIGH = 150

# ============================================================
# 轮廓筛选
# ============================================================
MIN_CONTOUR_AREA = 2000
RATIO_TOLERANCE = 0.4
MAX_CANDIDATES = 50

# ============================================================
# PnP
# ============================================================
PNP_MAX_REPROJ_ERROR = 20.0

# ============================================================
# ROI 跟踪
# ============================================================
ROI_EXPAND = 1.5
ROI_MIN_SIZE = 90
MAX_TRACK_LOST = 10
HOLD_LAST_FRAMES = 4


def _cvt_mvlab2cv(param: list) -> Tuple[np.ndarray, np.ndarray]:
    lmin, lmax, amin, amax, bmin, bmax = param
    lower = np.array([int(lmin * 2.55), int(amin + 128), int(bmin + 128)])
    upper = np.array([int(lmax * 2.55), int(amax + 128), int(bmax + 128)])
    return lower, upper


def _order_points(pts: np.ndarray) -> np.ndarray:
    """四点排序：左上 → 右上 → 右下 → 左下。"""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmax(diff)]
    rect[3] = pts[np.argmin(diff)]
    return rect.astype(np.int32)


class RectDetect:
    """已知尺寸矩形的单目检测与位姿估计。

    在去畸变图像上检测，角点反投影回畸变坐标输出。
    """

    def __init__(self, calib_path: str = "param.npz") -> None:
        with np.load(calib_path) as p:
            mtx: np.ndarray = p["mtx"]
            dist: np.ndarray = p["dist"]

        self.fx: float = float(mtx[0, 0])
        self.fy: float = float(mtx[1, 1])
        self.cx: float = float(mtx[0, 2])
        self.cy: float = float(mtx[1, 2])
        self._mtx: np.ndarray = mtx
        self._dist: np.ndarray = dist

        # ---- LAB 阈值 ----
        self._lab_lower, self._lab_upper = _cvt_mvlab2cv(MVLAB_PARAM)

        # ---- 3D 物点 ----
        _hw, _hh = RECT_WIDTH / 2, RECT_HEIGHT / 2
        self._obj_points: np.ndarray = np.array(
            [[-_hw, -_hh, 0.0], [_hw, -_hh, 0.0],
             [_hw,  _hh, 0.0], [-_hw,  _hh, 0.0]],
            dtype=np.float64,
        )

        # ---- 跟踪状态 ----
        self.last_quad: Optional[np.ndarray] = None
        self.last_rect: Optional[Tuple[int, int, int, int]] = None
        self._last_rect_undist: Optional[Tuple[int, int, int, int]] = None
        self.last_depth: float = -1.0
        self.last_ox: float = 0.0
        self.last_oy: float = 0.0
        self.lost_count: int = MAX_TRACK_LOST
        self.hold_count: int = 0

        # ---- 世界坐标偏移 ----
        self.err_x_world: float = 0.0
        self.err_y_world: float = 0.0

        # ---- 诊断 ----
        self.pass_rate: float = 0.0
        self._last_err: float = -1.0
        self._diag_contours: int = 0

        # ---- 去畸变 remap（懒初始化） ----
        self._remap_size: Optional[Tuple[int, int]] = None
        self._mapx: Optional[np.ndarray] = None
        self._mapy: Optional[np.ndarray] = None
        self._new_mtx: Optional[np.ndarray] = None

        # ---- 灰度图缓存 ----
        self._gray_undist: Optional[np.ndarray] = None

        # ---- 形态学核 ----
        ks = 2 * MORPH_KERNEL + 1
        self._morph_kernel: np.ndarray = cv2.getStructuringElement(
            cv2.MORPH_RECT, (ks, ks),
        )

    # ============================================================
    # 去畸变 remap
    # ============================================================

    def _ensure_remap(self, w: int, h: int) -> None:
        size = (w, h)
        if self._remap_size == size:
            return
        self._new_mtx, _ = cv2.getOptimalNewCameraMatrix(
            self._mtx, self._dist, size, alpha=1,
        )
        self._mapx, self._mapy = cv2.initUndistortRectifyMap(
            self._mtx, self._dist, None, self._new_mtx, size, cv2.CV_32FC1,
        )
        self._remap_size = size

    def _undist_to_dist(self, pts_undist: np.ndarray) -> np.ndarray:
        """去畸变像素坐标 → 畸变像素坐标（正向畸变模型）。"""
        k1, k2, p1, p2, k3 = self._dist[0]
        if self._new_mtx is None:
            raise RuntimeError("去畸变映射尚未初始化")
        src_fx, src_fy = self._new_mtx[0, 0], self._new_mtx[1, 1]
        src_cx, src_cy = self._new_mtx[0, 2], self._new_mtx[1, 2]
        dst_fx, dst_fy = self._mtx[0, 0], self._mtx[1, 1]
        dst_cx, dst_cy = self._mtx[0, 2], self._mtx[1, 2]

        out = pts_undist.astype(np.float32).reshape(-1, 2).copy()
        for i in range(len(out)):
            xn = (out[i, 0] - src_cx) / src_fx
            yn = (out[i, 1] - src_cy) / src_fy
            r2 = xn * xn + yn * yn
            radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
            xd = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
            yd = yn * radial + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
            out[i, 0] = dst_fx * xd + dst_cx
            out[i, 1] = dst_fy * yd + dst_cy
        return out.astype(np.int32)

    # ============================================================
    # 预处理（去畸变后）
    # ============================================================

    def _preprocess(self, bgr: np.ndarray) -> np.ndarray:
        """remap 去畸变 → LAB 分割 → 形态学 → Canny。"""
        h, w = bgr.shape[:2]
        self._ensure_remap(w, h)
        assert self._mapx is not None and self._mapy is not None
        bgr_undist = cv2.remap(bgr, self._mapx, self._mapy, cv2.INTER_LINEAR)

        self._gray_undist = cv2.cvtColor(bgr_undist, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(bgr_undist, cv2.COLOR_BGR2LAB)
        binary = cv2.inRange(lab, self._lab_lower, self._lab_upper)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self._morph_kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self._morph_kernel)
        return cv2.Canny(binary, CANNY_LOW, CANNY_HIGH)

    # ============================================================
    # PnP（去畸变坐标，dist=None）
    # ============================================================

    def _solve_pnp(
        self, image_pts: np.ndarray,
    ) -> Tuple[bool, float, float, float, float]:
        pts_f = image_pts.astype(np.float32).reshape(4, 1, 2)
        if self._gray_undist is not None:
            cv2.cornerSubPix(
                self._gray_undist, pts_f, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001),
            )
        pts = pts_f.astype(np.float64)

        if self._new_mtx is None:
            return False, float("inf"), 0.0, 0.0, 0.0
        ok, rvec, tvec = cv2.solvePnP(
            self._obj_points, pts, self._new_mtx, None,
            flags=cv2.SOLVEPNP_IPPE,
        )
        if not ok:
            return False, float("inf"), 0.0, 0.0, 0.0

        reproj, _ = cv2.projectPoints(
            self._obj_points, rvec, tvec, self._new_mtx, None,
        )
        reproj = reproj.reshape(4, 2)
        err = float(np.mean(np.hypot(
            reproj[:, 0] - image_pts[:, 0],
            reproj[:, 1] - image_pts[:, 1],
        )))

        tx, ty, tz = float(tvec[0]), float(tvec[1]), float(tvec[2])
        return True, err, tz, tx, ty

    # ============================================================
    # ROI
    # ============================================================

    def _expand_roi(
        self, rect: Tuple[int, int, int, int],
        img_w: int, img_h: int,
    ) -> Tuple[int, int, int, int]:
        x, y, w, h = rect
        cx, cy = x + w / 2, y + h / 2
        nw = max(int(w * ROI_EXPAND), ROI_MIN_SIZE)
        nh = max(int(h * ROI_EXPAND), ROI_MIN_SIZE)
        x0 = max(0, int(cx - nw / 2))
        y0 = max(0, int(cy - nh / 2))
        x1 = min(img_w, x0 + nw)
        y1 = min(img_h, y0 + nh)
        return (x0, y0, x1 - x0, y1 - y0)

    # ============================================================
    # 核心搜索
    # ============================================================

    def _search(
        self, mask: np.ndarray,
        offset_x: int = 0, offset_y: int = 0,
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]],
               float, float, float]:
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        self._diag_contours = len(contours)
        if not contours:
            return None, None, -1.0, 0.0, 0.0

        scored = [(cv2.contourArea(c), c) for c in contours]
        scored.sort(key=lambda x: x[0], reverse=True)
        scored = scored[:MAX_CANDIDATES]

        for area, cnt in scored:
            if area < MIN_CONTOUR_AREA:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) != 4:
                continue

            ordered = _order_points(approx.reshape(4, 2))

            if not cv2.isContourConvex(ordered):
                continue
            w_px = (np.hypot(ordered[1][0] - ordered[0][0], ordered[1][1] - ordered[0][1]) +
                    np.hypot(ordered[2][0] - ordered[3][0], ordered[2][1] - ordered[3][1])) / 2
            h_px = (np.hypot(ordered[2][0] - ordered[1][0], ordered[2][1] - ordered[1][1]) +
                    np.hypot(ordered[3][0] - ordered[0][0], ordered[3][1] - ordered[0][1])) / 2
            if h_px < 1.0:
                continue
            if abs(w_px / h_px - ASPECT_RATIO) / ASPECT_RATIO > RATIO_TOLERANCE:
                continue

            ok, err, depth, ox, oy = self._solve_pnp(ordered)
            if not ok or err >= PNP_MAX_REPROJ_ERROR:
                continue

            self._last_err = err
            self.pass_rate = 1.0

            x, y, w, h = cv2.boundingRect(ordered)
            quad = ordered + np.array([offset_x, offset_y], dtype=np.int32)
            rect = (x + offset_x, y + offset_y, w, h)
            return quad, rect, depth, ox, oy

        self._last_err = -1.0
        self.pass_rate = 0.0
        return None, None, -1.0, 0.0, 0.0

    # ============================================================
    # 外部接口
    # ============================================================

    def detect(
        self, bgr: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]], float]:
        """在 BGR 畸变图像上检测矩形。

        内部：去畸变 → 检测 → 角点反投影回畸变坐标
        返回值可直接在原图上绘制。
        """
        h, w = bgr.shape[:2]

        # ---- 0/1. 去畸变、缓存灰度图并预处理（只 remap 一次） ----
        mask = self._preprocess(bgr)

        quad_undist: Optional[np.ndarray] = None
        rect_undist: Optional[Tuple[int, int, int, int]] = None
        depth: float = -1.0
        found = False

        # ---- 2. ROI 搜索 ----
        if self._last_rect_undist is not None and self.lost_count < MAX_TRACK_LOST:
            rx, ry, rw, rh = self._expand_roi(self._last_rect_undist, w, h)
            quad_undist, rect_undist, depth, ox, oy = self._search(
                mask[ry:ry + rh, rx:rx + rw], rx, ry,
            )
            if quad_undist is not None:
                found = True

        # ---- 3. 全图搜索 ----
        if not found:
            quad_undist, rect_undist, depth, ox, oy = self._search(mask, 0, 0)
            found = quad_undist is not None

        # ---- 4. 反投影回畸变坐标 ----
        quad: Optional[np.ndarray] = None
        rect: Optional[Tuple[int, int, int, int]] = None

        if found:
            assert quad_undist is not None and rect_undist is not None
            quad = self._undist_to_dist(quad_undist)

            x, y, rw, rh = rect_undist
            rect_corners = np.array([[x, y], [x + rw, y], [x + rw, y + rh], [x, y + rh]])
            rect_corners_dist = self._undist_to_dist(rect_corners)
            rx = int(np.min(rect_corners_dist[:, 0]))
            ry = int(np.min(rect_corners_dist[:, 1]))
            rect = (rx, ry,
                    int(np.max(rect_corners_dist[:, 0]) - rx),
                    int(np.max(rect_corners_dist[:, 1]) - ry))

            self.lost_count = 0
            self.hold_count = 0
            self.last_quad = quad.copy()
            self.last_rect = rect
            self._last_rect_undist = rect_undist
            self.last_depth = depth
            self.last_ox = ox
            self.last_oy = oy
            self.err_x_world = ox
            self.err_y_world = oy
        else:
            self.lost_count += 1
            if self.last_rect is not None and self.hold_count < HOLD_LAST_FRAMES:
                self.hold_count += 1
                quad = self.last_quad
                rect = self.last_rect
                depth = self.last_depth
                self.err_x_world = self.last_ox
                self.err_y_world = self.last_oy

        return quad, rect, depth
