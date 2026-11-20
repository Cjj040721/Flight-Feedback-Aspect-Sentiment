import tempfile
import unittest
from pathlib import Path

import aspects
import sentiment
import storage
import web
from analysis import compute_aspect_stats
from evaluate_sentiment import load_gold_from_csv
from importer import load_long_csv


class FakeZeroShot:
    def __call__(self, sequences, candidate_labels, multi_label):
        return {
            "labels": candidate_labels,
            "scores": [
                0.95 if label == aspects.ZS_LABELS["SITE_SPEED"] else 0.01
                for label in candidate_labels
            ],
        }


class FakeHighScoreZeroShot:
    def __call__(self, sequences, candidate_labels, multi_label):
        raw = {
            "BOOKING_FLOW": 0.9361,
            "SITE_SPEED": 0.9973,
            "CUSTOMER_SERVICE": 0.7745,
            "PRICE_TRANSPARENCY": 0.9528,
            "FLIGHT_OPTIONS": 0.7622,
            "PAYMENT": 0.9794,
            "REFUND_CHANGE": 0.8854,
        }
        return {
            "labels": candidate_labels,
            "scores": [raw[code] for code in aspects.ASPECTS if code != "OTHER"],
        }


class AspectTests(unittest.TestCase):
    def tearDown(self):
        aspects._ZS = None

    def test_zero_shot_prediction_survives_without_keyword_match(self):
        aspects._ZS = FakeZeroShot()
        self.assertEqual(aspects.detect_aspects("The portal is sluggish"), ["SITE_SPEED"])

    def test_model_failure_uses_lexicon_without_substring_false_positive(self):
        self.assertFalse(aspects._token_in_text("fast", "breakfast"))
        aspects._ZS = False
        self.assertEqual(aspects.detect_aspects("The breakfast was quick"), ["OTHER"])

    def test_bilingual_lexicon_fallback(self):
        result = aspects.detect_aspects("客服不回复，payment failed", use_zero_shot=False)
        self.assertEqual(result, ["CUSTOMER_SERVICE", "PAYMENT"])
    def test_only_near_top_nli_labels_are_kept(self):
        aspects._ZS = FakeHighScoreZeroShot()
        result = aspects.detect_aspects("The website was fast but the payment failed twice.")
        self.assertEqual(result, ["SITE_SPEED", "PAYMENT"])

    def test_contrast_is_split_for_aspect_sentiment(self):
        clauses = sentiment._split_clauses("The website was fast but the payment failed twice.")
        self.assertEqual(clauses, ["The website was fast", "the payment failed twice."])


class StorageAndAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        storage.DB_PATH = str(Path(self.tempdir.name) / "nested" / "feedback.db")
        storage.init_db()

    def tearDown(self):
        self.tempdir.cleanup()

    def test_storage_and_statistics(self):
        fid = storage.insert_feedback("Payment failed", review_id="test_001")
        storage.insert_aspect_opinions(fid, [
            {"aspect": "PAYMENT", "sentiment": "negative", "score": 0.9, "source": "test"},
            {"aspect": "PAYMENT", "sentiment": "positive", "score": 0.8, "source": "test"},
        ])
        rows = storage.fetch_all_aspects()
        stats = compute_aspect_stats(rows)
        self.assertEqual(stats["PAYMENT"]["total"], 2)
        self.assertEqual(stats["PAYMENT"]["ratios"]["negative"], 50.0)
        self.assertEqual(storage.fetch_recent(1)[0]["review_id"], "test_001")

    def test_web_submission_and_summary_api(self):
        web.detect_aspects = lambda text: ["PAYMENT"]
        web.aspect_sentiment = lambda text, keywords: ("negative", 0.1, "fake")
        client = web.app.test_client()
        response = client.post("/", data={"text": "The card was declined"})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"The card was declined", response.data)
        summary = client.get("/api/summary").get_json()
        self.assertEqual(summary["PAYMENT"]["counts"]["negative"], 1)


class DatasetTests(unittest.TestCase):
    def test_sample_csv_and_majority_labels(self):
        path = Path(__file__).parents[1] / "data" / "sample_feedback.csv"
        records = list(load_long_csv(str(path)))
        texts, labels = load_gold_from_csv(str(path))
        self.assertEqual(len(records), 6)
        self.assertEqual(sum(len(record["opinions"]) for record in records), 12)
        self.assertEqual(len(texts), 6)
        self.assertEqual(labels[:2], ["positive", "positive"])


if __name__ == "__main__":
    unittest.main()
