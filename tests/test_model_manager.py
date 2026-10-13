"""Unit tests for AeroWatch AI ModelManager and Class Normalization."""
from __future__ import annotations

import unittest
from pathlib import Path

from model_manager import (
    CANONICAL_CIVILIAN_MAP,
    CheckpointMetadata,
    ModelManager,
    ModelType,
    discover_checkpoints,
)


class TestModelManager(unittest.TestCase):

    def setUp(self):
        self.mm = ModelManager()

    def test_canonical_class_mappings(self):
        """Verifies canonical civilian mappings without invented military subtypes."""
        self.assertEqual(CANONICAL_CIVILIAN_MAP["drone"], ("drone", "Drone"))
        self.assertEqual(CANONICAL_CIVILIAN_MAP["aircraft"], ("airplane", "Aircraft"))
        self.assertEqual(CANONICAL_CIVILIAN_MAP["military_jet"], ("airplane", "Aircraft"))
        self.assertEqual(CANONICAL_CIVILIAN_MAP["bird"], ("bird", "Bird"))
        self.assertEqual(CANONICAL_CIVILIAN_MAP["balloon"], ("balloon", "Balloon"))

    def test_checkpoint_introspection_custom(self):
        """Validates metadata parsing for custom aerial checkpoint."""
        pt = Path("weights/aerowatch_v7_balanced.pt")
        if not pt.exists():
            self.skipTest("aerowatch_v7_balanced.pt not available")

        meta = self.mm.introspect_checkpoint(pt)
        self.assertEqual(meta.model_type, ModelType.CUSTOM_CIVILIAN)
        self.assertTrue(meta.supports_drones())
        self.assertTrue(meta.supports_balloons())
        self.assertEqual(meta.raw_names[0], "drone")
        self.assertEqual(meta.raw_names[1], "aircraft")

    def test_checkpoint_introspection_coco(self):
        """Ensures pretrained COCO model is NOT treated as custom drone model."""
        pt = Path("weights/yolo26n.pt")
        if not pt.exists():
            pt = Path("yolo26n.pt")
        if not pt.exists():
            self.skipTest("yolo26n.pt baseline not found")

        meta = self.mm.introspect_checkpoint(pt)
        self.assertEqual(meta.model_type, ModelType.PRETRAINED_COCO)
        self.assertFalse(meta.supports_drones())
        self.assertFalse(meta.supports_balloons())
        self.assertEqual(meta.raw_names[4], "airplane")

    def test_coco_airplane_mapping(self):
        """COCO class 4 (airplane) maps to civilian Aircraft; other COCO classes are filtered."""
        pt = Path("weights/yolo26n.pt")
        if not pt.exists():
            pt = Path("yolo26n.pt")
        if not pt.exists():
            self.skipTest("yolo26n.pt baseline not found")

        self.mm.load_model(pt, device="cpu", enable_half=False)

        # COCO class 4 (airplane) -> Aircraft
        internal, display, color = self.mm.map_detection_label(4, 0.85)
        self.assertEqual(internal, "airplane")
        self.assertEqual(display, "Aircraft")

        # COCO class 14 (bird) -> Bird
        internal, display, color = self.mm.map_detection_label(14, 0.70)
        self.assertEqual(internal, "bird")
        self.assertEqual(display, "Bird")

        # COCO class 0 (person) should NOT be mapped to drone or aerial target
        internal, display, color = self.mm.map_detection_label(0, 0.90)
        self.assertIsNone(internal)
        self.assertIsNone(display)

    def test_custom_warplane_normalized_to_aircraft(self):
        """Custom class 4 (military_jet) must normalize to civilian Aircraft."""
        pt = Path("weights/aerowatch_v7_balanced.pt")
        if not pt.exists():
            self.skipTest("aerowatch_v7_balanced.pt not available")

        self.mm.load_model(pt)
        internal, display, color = self.mm.map_detection_label(4, 0.80)
        self.assertEqual(internal, "airplane")
        self.assertEqual(display, "Aircraft")

    def test_low_confidence_uncertainty_threshold(self):
        """Detections between detection_threshold and uncertainty_threshold are flagged as Uncertain."""
        pt = Path("weights/aerowatch_v7_balanced.pt")
        if not pt.exists():
            self.skipTest("aerowatch_v7_balanced.pt not available")

        self.mm.load_model(pt)
        # 0.30 is above detection (0.25) but below uncertainty (0.40)
        internal, display, color = self.mm.map_detection_label(0, 0.30, detection_threshold=0.25, uncertainty_threshold=0.40)
        self.assertEqual(internal, "drone")
        self.assertIn("Uncertain", display)

        # 0.15 is below detection threshold -> None
        internal, display, color = self.mm.map_detection_label(0, 0.15, detection_threshold=0.25)
        self.assertIsNone(internal)
        self.assertIsNone(display)

    def test_missing_model_exception(self):
        """Loading a non-existent checkpoint raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            self.mm.load_model("weights/non_existent_weights_xyz.pt")


if __name__ == "__main__":
    unittest.main()
