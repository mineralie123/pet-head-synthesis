"""Regression checks for the v2 gate and deterministic region geometry."""
import unittest

from pipeline_rules_v2 import (Detection, Square, compare, decide_candidate, describe_legacy_scores,
                               generate_candidates, plan_square, verify_square)

CFG = dict(min_head_fraction=0.60, max_head_fraction=0.80, target_head_fraction=0.70,
           duplicate_iou=0.55, nested_containment=0.90)


class RulesTests(unittest.TestCase):
    def test_frame_boundaries(self):
        self.assertEqual(verify_square(Square(0, 0, 40), (40, 40)), ())
        self.assertTrue(verify_square(Square(0, 0, 39), (50, 50)))
        self.assertTrue(verify_square(Square(2, 0, 40), (40, 40)))
        self.assertTrue(verify_square(Square(0, 0, 40, 45), (40, 40)))
        self.assertFalse(verify_square(Square(20, 20, 40, 45), (80, 80)))
        square, reason, flags = plan_square((0.1, 0.1, 32.1, 30.0), (40, 40), CFG)
        self.assertEqual((square, reason), (Square(0, 0, 40), ""))
        self.assertIn("BOX_NEAR_SOURCE_EDGE", flags)
        self.assertEqual(plan_square((1, 1, 32, 32), (100, 100), CFG)[1], "BOX_WIDTH_PROXY_TOO_SMALL")
        self.assertEqual(plan_square((0, 0, 99, 89), (100, 90), CFG)[1], "BOX_RATIO_PROXY_FAIL")
        self.assertEqual(plan_square((0, 0, float("nan"), 89), (100, 90), CFG)[1], "BOX_DEGENERATE")

    def test_best_overlap_not_transitive_merge(self):
        a = Detection("cat head", 0, 0.70, (25, 30, 65, 70))
        b = Detection("cat head", 1, 0.60, (45, 30, 85, 70))
        bridge = Detection("dog head", 0, 0.80, (33.5, 28, 77, 72))
        computed = compare([a, b, bridge], (150, 150), CFG)
        inverted = compare([bridge, b, a], (150, 150), CFG)
        self.assertEqual(computed, inverted)
        self.assertEqual(len(computed["groups"]), 2)
        self.assertTrue(all(g["links"] for g in computed["groups"]))
        self.assertEqual(len(computed["cross"]), 2)
        self.assertEqual(computed, compare([b, bridge, a], (150, 150), CFG))
        with self.assertRaisesRegex(ValueError, "Duplicate detection key"):
            compare([a, a], (150, 150), CFG)

    def test_other_member_warning_is_scoped(self):
        valid = Detection("cat head", 0, 0.60, (30, 30, 70, 70))
        narrow = Detection("dog head", 0, 0.80, (33, 33, 63, 67), ("box_width_under_32px",))
        analysis = compare([narrow, valid], (125, 125), CFG)
        candidates, _ = generate_candidates("sample", "hash", (125, 125), analysis, "test")
        self.assertEqual(len(candidates), 1)
        entry = candidates[0]
        self.assertEqual(entry["representative_key"], valid.key)
        self.assertEqual(entry["representative_model_flags"], [])
        self.assertNotIn("box_width_under_32px", entry["local_flags"])
        self.assertEqual(entry["other_member_flags"][0]["plan_error"], "BOX_WIDTH_PROXY_TOO_SMALL")
        self.assertIn("DINO_REGION_CROSS_PROMPT", entry["local_flags"])

    def test_human_required_and_scoped(self):
        record = dict(crop_id="crop", original_class="Real Cat", crop_class="Real Cat", local_flags=[],
                      square=Square(0, 0, 40), geometry_errors=(), preview_verified=True, preview_hash="xyz")
        self.assertEqual(decide_candidate(record)["decision"], "REVIEW")
        head_review = dict(crop_id="crop", preview_hash="xyz", human_species="cat", real_photo="yes",
                           head_quality="good", crop_quality="good", final_decision="accept",
                           reviewer="tester", reviewed_at="2026-11-18",
                           exactly_one_head="yes", coverage_includes_ears="yes", centered="yes",
                           upright="yes", visual_scale="yes", native_width_verified="yes")
        self.assertEqual(decide_candidate(record, head_review)["label"], 0)
        self.assertEqual(decide_candidate(record, dict(head_review, preview_hash="stale"))["reason_codes"],
                         ["HUMAN_REVIEW_STALE"])
        conflict = dict(record, crop_class="Real Dog")
        self.assertEqual(decide_candidate(conflict, head_review)["decision"], "REVIEW")
        self.assertEqual(decide_candidate(conflict, dict(head_review, conflict_resolution="confirmed cat"))["decision"],
                         "ACCEPT")
        is_toy = dict(head_review, final_decision="reject", real_photo="no")
        self.assertEqual(decide_candidate(record, is_toy)["decision"], "REJECT")
        result = decide_candidate(dict(record, square=None, planning_error="BOX_WIDTH_PROXY_TOO_SMALL"))
        self.assertEqual(result["decision"], "REVIEW")

    def test_legacy_scores(self):
        self.assertEqual(describe_legacy_scores([0.20, 0.18, 0.1, 0.1, 0.1, 0.1])["prediction"], "Real Cat")
        self.assertEqual(describe_legacy_scores([0.30, 0.30, 0.20, 0.10, 0.10, 0.0])["prediction"], "Uncertain")
        with self.assertRaisesRegex(ValueError, "finite"):
            describe_legacy_scores([0.20, 0.18, 0.1, float("nan"), 0.1, 0.1])


if __name__ == "__main__":
    unittest.main()

