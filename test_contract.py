"""CPU-only request boundary checks; these do not validate model inference."""

import json
from pathlib import Path
import unittest

from app import CATALOG, validate_input


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.job = json.loads(Path("examples/request.json").read_text())

    def test_documented_request(self):
        self.assertEqual(validate_input(self.job), self.job["input"])

    def test_rejects_unsupported_generation_requests(self):
        with self.assertRaises(ValueError):
            validate_input({"input": {"prompt": "hello", "max_tokens": 10}})

    def test_rejects_empty_and_oversized_question_sets(self):
        with self.assertRaises(ValueError):
            validate_input({"input": {"state": "hello", "questions": {}}})
        with self.assertRaises(ValueError):
            validate_input(self.job, max_questions=2)

    def test_rejects_nonfinite_or_oversized_input(self):
        with self.assertRaises(ValueError):
            validate_input(self.job, max_input_bytes=10)
        self.job["input"]["state"] = {"number": float("nan")}
        with self.assertRaises(ValueError):
            validate_input(self.job)

    def test_rejects_unsupported_type_and_scalar_state(self):
        self.job["input"]["questions"]["receipt"]["type"] = "generate"
        with self.assertRaises(ValueError):
            validate_input(self.job)
        self.job["input"]["questions"]["receipt"]["type"] = "noul"
        self.job["input"]["state"] = 42
        with self.assertRaises(ValueError):
            validate_input(self.job)

    def test_all_models_have_immutable_revisions(self):
        self.assertEqual(len(CATALOG), 6)
        for item in CATALOG.values():
            self.assertRegex(item["revision"], r"^[0-9a-f]{40}$")
        self.assertGreater(CATALOG["vllm-sr/Decision-2.0-Vega-27B"]["approx_download_gb"], 70)


if __name__ == "__main__":
    unittest.main()
