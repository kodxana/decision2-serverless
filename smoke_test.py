"""Real inference acceptance test. Run inside the built container, with a GPU."""

import json
import math
from pathlib import Path

from app import DecisionWorker


def verify(result):
    answers = result["answers"]
    assert answers["route"]["choice"] == "returns", result
    assert 0.5 < answers["receipt"]["noul"] <= 1.0, result
    assert 0.0 <= answers["urgency"]["score"] <= 2.0, result
    for name in ("route", "urgency"):
        probabilities = answers[name]["probabilities"].values()
        assert all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities), result
        assert math.isclose(sum(probabilities), 1.0, abs_tol=1e-5), result
    assert result["usage"]["input_tokens"] > 0, result
    assert result["usage"]["output_tokens"] == 0, result


def main():
    job = json.loads(Path(__file__).with_name("examples").joinpath("request.json").read_text())
    worker = DecisionWorker()
    first = worker.handle(job)
    verify(first)
    # Exercise an actual native-runtime question error, then prove recovery.
    invalid = {"input": {"state": "hello", "questions": {
        "bad": {"type": "score", "instructions": "Rate this", "criteria": ["only one"]}
    }}}
    try:
        worker.handle(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid score criteria should fail the job")
    for _ in range(2):
        verify(worker.handle(job))
    print(json.dumps(first, indent=2))
    print("PASS: real decisions, probabilities, repeated requests, and invalid-job recovery")


if __name__ == "__main__":
    main()
