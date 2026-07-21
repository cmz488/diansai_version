import unittest
from pathlib import Path

import cv2
import numpy as np

from tools._colors import cvt_mvlab2cv
from tools._laser_detect import detect_laser_binary
from tools._threshold import AutoThresholder
from tools._tracking import LaserSpotDetector


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LASER_THRESHOLD = (99, 100, -32, 28, -38, 26)


def make_detector(**overrides):
    options = {
        "track_radius": 120,
        "smooth_alpha": 0.65,
        "full_search_interval": 30,
        "min_area": 5,
        "max_area": 1000,
        "morph_kernel_size": 3,
        "roi_margin": 4,
        "max_aspect_ratio": 3.0,
        "min_confidence": 0.25,
        "color_mode": "blue",
        "threshold": LASER_THRESHOLD,
    }
    options.update(overrides)
    return LaserSpotDetector(**options)


def synthetic_laser_frame():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    # 蓝色光晕和近白色过曝核心。
    frame[110:127, 150:167, 0] = 180
    frame[115:122, 155:162] = 252
    return frame


class ColorConversionTests(unittest.TestCase):
    def test_l_100_maps_to_255(self):
        lower, upper = cvt_mvlab2cv((99, 100, -32, 28, -38, 26))
        self.assertEqual(int(lower[0]), 252)
        self.assertEqual(int(upper[0]), 255)
        self.assertEqual(lower.dtype, np.uint8)
        self.assertEqual(upper.dtype, np.uint8)

    def test_invalid_threshold_order_is_rejected(self):
        with self.assertRaises(ValueError):
            cvt_mvlab2cv((90, 80, -10, 10, -10, 10))


class LaserSpotDetectorTests(unittest.TestCase):
    def test_detects_small_blue_laser(self):
        spot = make_detector().detect(synthetic_laser_frame())
        self.assertIsNotNone(spot)
        self.assertAlmostEqual(spot.x, 158.0, delta=1.0)
        self.assertAlmostEqual(spot.y, 118.0, delta=1.0)
        self.assertGreater(spot.confidence, 0.8)

    def test_polygon_excludes_outside_glare(self):
        frame = synthetic_laser_frame()
        frame[20:40, 20:40] = 255
        polygon = np.array(
            [[100, 70], [220, 70], [220, 180], [100, 180]], dtype=np.float32
        )
        spot = make_detector().detect(frame, search_polygon=polygon)
        self.assertIsNotNone(spot)
        self.assertAlmostEqual(spot.x, 158.0, delta=1.0)
        self.assertAlmostEqual(spot.y, 118.0, delta=1.0)

        outside_polygon = np.array(
            [[250, 180], [310, 180], [310, 230], [250, 230]], dtype=np.float32
        )
        self.assertIsNone(
            make_detector().detect(frame, search_polygon=outside_polygon)
        )

    def test_large_highlight_is_rejected(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[40:100, 40:100] = 253
        self.assertIsNone(make_detector().detect(frame))

    def test_success_resets_consecutive_misses(self):
        detector = make_detector(max_consecutive_misses=3)
        frame = synthetic_laser_frame()
        blank = np.zeros_like(frame)

        self.assertIsNotNone(detector.detect(frame))
        self.assertIsNone(detector.detect(blank))
        self.assertEqual(detector.stats["consecutive_misses"], 1)
        self.assertIsNotNone(detector.detect(frame))
        self.assertEqual(detector.stats["consecutive_misses"], 0)
        self.assertEqual(detector.stats["misses"], 1)

    def test_second_frame_uses_tracking_roi(self):
        detector = make_detector()
        frame = synthetic_laser_frame()
        first = detector.detect(frame)
        second = detector.detect(frame)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(first.search_mode, "full")
        self.assertEqual(second.search_mode, "tracking")
        self.assertEqual(detector.stats["track_hits"], 1)

    def test_project_positive_sample(self):
        frame = cv2.imread(str(PROJECT_ROOT / "1.jpg"))
        self.assertIsNotNone(frame)
        spot = make_detector().detect(frame)
        self.assertIsNotNone(spot)
        self.assertAlmostEqual(spot.x, 44.7, delta=2.0)
        self.assertAlmostEqual(spot.y, 179.0, delta=2.0)

    def test_project_negative_sample_inside_target_rect(self):
        frame = cv2.imread(
            str(PROJECT_ROOT / "photos" / "snapshot_20260717_203625_898931.jpg")
        )
        self.assertIsNotNone(frame)
        target_rect = np.array(
            [[162, 205], [130, 578], [744, 527], [672, 122]],
            dtype=np.float32,
        )
        self.assertIsNone(
            make_detector().detect(frame, search_polygon=target_rect)
        )


class BinaryDetectorTests(unittest.TestCase):
    def test_untrained_auto_threshold_is_not_learned_implicitly(self):
        auto = AutoThresholder(strategy="histogram")
        self.assertIsNone(
            detect_laser_binary(synthetic_laser_frame(), auto_thresh=auto)
        )

    def test_histogram_bounds_ignore_distant_small_outliers(self):
        values = np.array([100] * 100 + [10] * 5 + [200] * 5, dtype=np.float32)
        lower, upper = AutoThresholder.approx_threshold(values)
        self.assertGreater(lower, 90)
        self.assertLess(upper, 110)


if __name__ == "__main__":
    unittest.main()
