"""STEP 12 - concurrency demonstration.

Run:  locust -f loadtest/locustfile.py --host http://localhost:8000
Then open http://localhost:8089, set users/spawn-rate, and watch RPS / latency while
Prometheus (`/metrics`) and Grafana chart the same traffic server-side.
"""
from __future__ import annotations

import json
import os
import random

from locust import HttpUser, between, task

API_KEY = os.environ.get("API_KEYS", "dev-key-change-me").split(",")[0]
QUESTIONS = [
    "What is the refund policy for cancellations?",
    "How long is the product warranty?",
    "What file formats can be ingested?",
    "Summarize the key points from the documentation.",
    "What happens if a payment fails?",
]


class ChatUser(HttpUser):
    wait_time = between(0.5, 2.0)

    def on_start(self):
        self.session_id = f"locust-{random.randint(1, 100000)}"
        self.headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    @task(5)
    def chat_stream(self):
        payload = {"message": random.choice(QUESTIONS), "session_id": self.session_id, "k": 5}
        with self.client.post("/chat/stream", data=json.dumps(payload), headers=self.headers,
                              stream=True, catch_response=True, name="/chat/stream") as r:
            ok = False
            try:
                for line in r.iter_lines():
                    if line and line.startswith(b"event: done"):
                        ok = True
                        break
            finally:
                r.success() if ok else r.failure("stream did not reach 'done'")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
