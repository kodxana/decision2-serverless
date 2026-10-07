"""Queue worker: one model initialization, sequential jobs, native JSON output."""

import logging

from app import DecisionWorker


def main():
    import runpod

    logging.basicConfig(level=logging.INFO)
    worker = DecisionWorker()
    runpod.serverless.start({"handler": worker.handle})


if __name__ == "__main__":
    main()
