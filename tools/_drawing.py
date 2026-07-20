"""透视绘图工具 — 在检测到的矩形区域上进行透视绘图与标注。"""

from typing import Optional, Tuple

import cv2
import numpy as np

MatLike = np.ndarray


class DrawGraph:
    """在检测到的矩形区域上进行透视绘图。

    将"目标平面"（正交矩形）上的图形通过逆透视变换映射回原图。
    """

    def __init__(
        self,
        src_pts: np.ndarray,
        plane_width: int,
        plane_height: int,
    ) -> None:
        if src_pts.shape != (4, 2):
            raise ValueError(f"src_pts 形状必须为 (4, 2)，实际为 {src_pts.shape}")

        self.src_pts = src_pts.astype(np.float32)
        self.plane_w = int(plane_width)
        self.plane_h = int(plane_height)

        dst_pts = np.array(
            [
                [0, 0],
                [self.plane_w - 1, 0],
                [self.plane_w - 1, self.plane_h - 1],
                [0, self.plane_h - 1],
            ],
            dtype=np.float32,
        )

        self.M_forward = cv2.getPerspectiveTransform(self.src_pts, dst_pts)
        self.M_inverse = cv2.getPerspectiveTransform(dst_pts, self.src_pts)

    # ------------------------------------------------------------------
    # 坐标映射
    # ------------------------------------------------------------------

    def map_to_image(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.M_inverse).reshape(-1, 2)

    def map_from_image(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.M_forward).reshape(-1, 2)

    def norm_to_image(self, u: float, v: float) -> Tuple[float, float]:
        px = u * (self.plane_w - 1)
        py = v * (self.plane_h - 1)
        result = self.map_to_image([[px, py]])
        return float(result[0, 0]), float(result[0, 1])

    # ------------------------------------------------------------------
    # 图像变换
    # ------------------------------------------------------------------

    def warp(self, image: MatLike) -> MatLike:
        return cv2.warpPerspective(image, self.M_forward, (self.plane_w, self.plane_h))

    def unwarp(
        self, plane_image: MatLike, output_size: Optional[Tuple[int, int]] = None
    ) -> MatLike:
        if output_size is None:
            x_min = int(np.floor(self.src_pts[:, 0].min()))
            x_max = int(np.ceil(self.src_pts[:, 0].max()))
            y_min = int(np.floor(self.src_pts[:, 1].min()))
            y_max = int(np.ceil(self.src_pts[:, 1].max()))
            output_size = (x_max - x_min, y_max - y_min)
        return cv2.warpPerspective(plane_image, self.M_inverse, output_size)

    # ------------------------------------------------------------------
    # 网格
    # ------------------------------------------------------------------

    def grid_points(self, rows: int, cols: int) -> np.ndarray:
        if rows < 1 or cols < 1:
            raise ValueError("rows 和 cols 必须 ≥ 1")
        xs = np.linspace(0, self.plane_w - 1, cols)
        ys = np.linspace(0, self.plane_h - 1, rows)
        xx, yy = np.meshgrid(xs, ys)
        plane_pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
        img_pts = self.map_to_image(plane_pts)
        return img_pts.reshape(rows, cols, 2)

    def draw_grid(
        self,
        image: MatLike,
        rows: int,
        cols: int,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 1,
        draw_points: bool = True,
        point_radius: int = 2,
    ) -> MatLike:
        gp = self.grid_points(rows, cols)
        for r in range(rows):
            pts = gp[r, :, :].astype(np.int32)
            for c in range(cols - 1):
                cv2.line(image, tuple(pts[c]), tuple(pts[c + 1]), color, thickness)
        for c in range(cols):
            pts = gp[:, c, :].astype(np.int32)
            for r in range(rows - 1):
                cv2.line(image, tuple(pts[r]), tuple(pts[r + 1]), color, thickness)
        if draw_points:
            for r in range(rows):
                for c in range(cols):
                    cv2.circle(
                        image,
                        (int(gp[r, c, 0]), int(gp[r, c, 1])),
                        point_radius,
                        color,
                        -1,
                    )
        return image

    # ------------------------------------------------------------------
    # 点 / 十字标注
    # ------------------------------------------------------------------

    def draw_point(
        self,
        image: MatLike,
        u: float,
        v: float,
        color: Tuple[int, int, int] = (0, 0, 255),
        radius: int = 5,
        filled: bool = True,
    ) -> MatLike:
        x, y = self.norm_to_image(u, v)
        cv2.circle(image, (int(x), int(y)), radius, color, -1 if filled else 1)
        return image

    def draw_cross(
        self,
        image: MatLike,
        u: float,
        v: float,
        size: int = 12,
        color: Tuple[int, int, int] = (0, 0, 255),
        thickness: int = 2,
    ) -> MatLike:
        x, y = self.norm_to_image(u, v)
        ix, iy = int(x), int(y)
        half = size // 2
        cv2.line(image, (ix - half, iy), (ix + half, iy), color, thickness)
        cv2.line(image, (ix, iy - half), (ix, iy + half), color, thickness)
        return image

    def draw_label(
        self,
        image: MatLike,
        u: float,
        v: float,
        text: str,
        color: Tuple[int, int, int] = (0, 0, 255),
        font_scale: float = 0.5,
        offset_x: int = 8,
        offset_y: int = -8,
    ) -> MatLike:
        x, y = self.norm_to_image(u, v)
        cv2.putText(
            image, text,
            (int(x) + offset_x, int(y) + offset_y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1,
        )
        return image

    # ------------------------------------------------------------------
    # 边框
    # ------------------------------------------------------------------

    def draw_border(
        self,
        image: MatLike,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> MatLike:
        pts = self.src_pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [pts], True, color, thickness)
        return image

    def draw_corners(
        self,
        image: MatLike,
        color: Tuple[int, int, int] = (0, 255, 0),
        radius: int = 6,
        labels: bool = True,
    ) -> MatLike:
        corner_names = ["TL", "TR", "BR", "BL"]
        for i, name in enumerate(corner_names):
            x, y = int(self.src_pts[i, 0]), int(self.src_pts[i, 1])
            cv2.circle(image, (x, y), radius, color, -1)
            if labels:
                cv2.putText(
                    image, name, (x + radius + 2, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                )
        return image
