"""Submit real jobs and verify a deployed Runpod queue endpoint.

Uses RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID from the environment. This creates
billable inference jobs; run only after deliberately deploying an endpoint.
"""

import json
import os
from pathlib import Path
import time
import urllib.request

from smoke_test import verify


def main():
    endpoint = os.environ["RUNPOD_ENDPOINT_ID"]
    if not endpoint.isalnum():
        raise ValueError("RUNPOD_ENDPOINT_ID must be an alphanumeric endpoint ID")
    base = f"https://api.runpod.ai/v2/{endpoint}"
    key = os.environ["RUNPOD_API_KEY"]

    def request(path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(
            base + path,
            data=data,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.load(response)

    def submit(payload, expected_status):
        status = request("/run", payload)
        job_id = status["id"]
        print("Submitted job:", job_id)
        deadline = time.monotonic() + 900
        while status.get("status") not in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Job {job_id} still pending; inspect or cancel it in Runpod")
            time.sleep(3)
            status = request(f"/status/{job_id}")
        if status["status"] != expected_status:
            raise RuntimeError(json.dumps(status))
        return status

    payload = json.loads(Path(__file__).with_name("examples").joinpath("request.json").read_text())
    first = submit(payload, "COMPLETED")
    verify(first["output"])
    print(json.dumps(first, indent=2))
    submit({"input": {"state": "hello", "questions": {}}}, "FAILED")
    verify(submit(payload, "COMPLETED")["output"])
    print("PASS: external Runpod inference, expected decisions, and invalid-job recovery")


if __name__ == "__main__":
    main()
