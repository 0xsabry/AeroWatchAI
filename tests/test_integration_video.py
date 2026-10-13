"""Integration tests for AeroWatch AI Video Pipeline using synthetic video frames."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from model_manager import ModelManager
from video_processor import VideoProcessingMode, VideoProcessor


class TestVideoPipelineIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.temp_path = Path(cls.temp_dir.name)

        # Generate a short controlled 15-frame 320x240 test video with flying disc (drone proxy)
        cls.synthetic_video = cls.temp_path / "synthetic_airspace_test.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(cls.synthetic_video), fourcc, 15.0, (320, 240))

        for f in range(15):
            # Sky backdrop
            frame = np.ones((240, 320, 3), dtype=np.uint8) * 210
            frame[:, :, 0] = 235  # Sky blue
            # Moving civilian object
            cx = int(40 + f * 16)
            cy = int(80 + np.sin(f * 0.4) * 20)
            cv2.circle(frame, (cx, cy), 16, (40, 40, 40), -1)
            cv2.circle(frame, (cx, cy), 6, (0, 180, 255), -1)
            writer.write(frame)

        writer.release()

        cls.mm = ModelManager()
        # Load custom checkpoint if available, else baseline
        target_pt = Path("weights/aerowatch_v7_balanced.pt")
        if not target_pt.exists():
            target_pt = Path("weights/yolo26n.pt")
        if not target_pt.exists():
            target_pt = Path("yolo26n.pt")

        cls.mm.load_model(target_pt)
        cls.vp = VideoProcessor(cls.mm)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_full_pipeline_execution_and_output_reopening(self):
        """Processes synthetic video, writes annotated output, and validates reopening."""
        out_mp4 = self.temp_path / "output_annotated.mp4"

        metrics = self.vp.process_video(
            video_path=self.synthetic_video,
            output_path=out_mp4,
            mode=VideoProcessingMode.SAVE_ANNOTATED,
            vid_stride=1,
            detection_threshold=0.10,
        )

        # Validate decoded and processed counts
        self.assertEqual(metrics.frames_decoded, 15)
        self.assertEqual(metrics.frames_processed, 15)
        self.assertEqual(metrics.frames_skipped, 0)
        self.assertGreater(metrics.processed_fps, 0.0)

        # Validate stage latencies were measured
        self.assertGreater(metrics.avg_read_ms, 0.0)
        self.assertGreater(metrics.avg_infer_ms, 0.0)
        self.assertGreater(metrics.avg_render_ms, 0.0)
        self.assertGreater(metrics.avg_write_ms, 0.0)

        # Validate output video actually exists and can be reopened by OpenCV
        self.assertTrue(out_mp4.exists())
        self.assertTrue(metrics.output_verified)

        test_cap = cv2.VideoCapture(str(out_mp4))
        self.assertTrue(test_cap.isOpened())
        ret, frame = test_cap.read()
        self.assertTrue(ret)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (240, 320, 3))
        test_cap.release()

    def test_invalid_video_path_handling(self):
        """Passing non-existent video path raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            self.vp.process_video(video_path="non_existent_folder/missing_video.mp4")

    def test_cancellation_handler(self):
        """Cancellation stops loop cleanly without exceptions."""
        called_frames = []

        def on_rendered(frame, idx):
            called_frames.append(idx)
            if idx >= 3:
                self.vp.cancel()

        metrics = self.vp.process_video(
            video_path=self.synthetic_video,
            vid_stride=1,
            on_frame_rendered=on_rendered,
        )

        self.assertLessEqual(metrics.frames_processed, 5)


if __name__ == "__main__":
    unittest.main()
