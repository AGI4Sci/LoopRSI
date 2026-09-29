import math
import unittest

from hier_loop.evaluation_authority import (
    EvaluationRecord,
    promotion_eligible,
    validation_reward,
)


def result(**overrides):
    value = {
        "authority": "search_validation",
        "candidate_pcc": 0.31,
        "candidate_hash": "sha256:" + "a" * 64,
        "full_validation": True,
        "metric_source": "real",
        "selection_feedback_allowed": True,
    }
    value.update(overrides)
    return value


class EvaluationAuthorityTests(unittest.TestCase):
    def test_two_allowed_authorities_parse(self):
        search = EvaluationRecord.from_mapping(result())
        final = EvaluationRecord.from_mapping(
            result(authority="final_official", selection_feedback_allowed=False)
        )
        self.assertEqual(search.authority, "search_validation")
        self.assertEqual(final.authority, "final_official")

    def test_missing_or_unknown_authority_is_rejected(self):
        missing = result()
        del missing["authority"]
        with self.assertRaisesRegex(ValueError, "authority"):
            EvaluationRecord.from_mapping(missing)
        with self.assertRaisesRegex(ValueError, "authority"):
            EvaluationRecord.from_mapping(result(authority="search_validaton"))

    def test_full_search_validation_is_required_for_promotion(self):
        self.assertTrue(promotion_eligible(EvaluationRecord.from_mapping(result())))
        self.assertFalse(
            promotion_eligible(EvaluationRecord.from_mapping(result(full_validation=False)))
        )
        self.assertFalse(
            promotion_eligible(
                EvaluationRecord.from_mapping(
                    result(authority="final_official", selection_feedback_allowed=False)
                )
            )
        )

    def test_validation_reward_is_candidate_minus_same_contract_baseline(self):
        record = EvaluationRecord.from_mapping(result(candidate_pcc=0.318))
        self.assertAlmostEqual(validation_reward(record, baseline=0.301), 0.017)

    def test_non_reward_inputs_return_none(self):
        excluded = (
            result(full_validation=False, metric_source="smoke_only"),
            result(metric_source="synthetic"),
            result(metric_source="plan_proxy"),
            result(metric_source="historical"),
            result(authority="final_official", selection_feedback_allowed=False),
        )
        for value in excluded:
            with self.subTest(value=value):
                record = EvaluationRecord.from_mapping(value)
                self.assertIsNone(validation_reward(record, baseline=0.301))
                self.assertFalse(promotion_eligible(record))

    def test_non_finite_metrics_and_missing_candidate_hash_are_rejected(self):
        for candidate_pcc in (math.nan, math.inf, -math.inf):
            with self.subTest(candidate_pcc=candidate_pcc):
                with self.assertRaisesRegex(ValueError, "candidate_pcc"):
                    EvaluationRecord.from_mapping(result(candidate_pcc=candidate_pcc))
        for baseline in (math.nan, math.inf, -math.inf):
            with self.subTest(baseline=baseline):
                with self.assertRaisesRegex(ValueError, "baseline"):
                    validation_reward(EvaluationRecord.from_mapping(result()), baseline)
        with self.assertRaisesRegex(ValueError, "candidate_hash"):
            EvaluationRecord.from_mapping(result(candidate_hash=""))

    def test_final_record_must_disable_selection_feedback(self):
        with self.assertRaisesRegex(ValueError, "selection_feedback_allowed=false"):
            EvaluationRecord.from_mapping(
                result(authority="final_official", selection_feedback_allowed=True)
            )


if __name__ == "__main__":
    unittest.main()
