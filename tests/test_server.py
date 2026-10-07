import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
import stubs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = os.path.join(ROOT, "share", "jev-local", "server.py")
KEY = "test-key"
MEMORY_MB = 4096
CACHE_MB = 256
REVISION = "0e5e6aa7d6d750e2b1504ba11a8136cb58aeb3cd"


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class Sandbox:
    def __init__(self, pick=0):
        self.root = tempfile.mkdtemp(prefix="jev-local-test-")
        stub = os.path.join(self.root, "stub")
        files = {
            "mlx/__init__.py": "",
            "mlx/core.py": stubs.MLX_CORE_STUB,
            "huggingface_hub/__init__.py": stubs.HUGGINGFACE_HUB_STUB,
            "transformers/__init__.py": stubs.TRANSFORMERS_STUB,
            "mlx_lm/__init__.py": "",
            "mlx_lm/__main__.py": stubs.MLX_LM_MAIN_STUB,
            "mlx_lm/utils.py": stubs.MLX_LM_UTILS_STUB.replace("PICK = 0", "PICK = %d" % pick),
        }
        for name, body in files.items():
            path = os.path.join(stub, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as handle:
                handle.write(body)
        self.stub = stub
        self.model = os.path.join(self.root, "model")
        self.lock = os.path.join(self.root, "owner.lock")
        self.hub = os.path.join(self.root, "hf")
        os.makedirs(self.hub)
        self.process = None
        self.port = None

    def start(self, grace="30"):
        self.port = free_port()
        environment = dict(os.environ)
        environment.update(
            PYTHONPATH=self.stub,
            HF_HOME=self.hub,
            DECISION_HOST="127.0.0.1",
            DECISION_PORT=str(self.port),
            DECISION_API_KEY=KEY,
            DECISION_REPO="internlm/Intern-Decision-4B",
            DECISION_REVISION=REVISION,
            DECISION_MODEL_DIR=os.path.join(self.model, "prepared"),
            DECISION_MEMORY_LIMIT_MB=str(MEMORY_MB),
            DECISION_CACHE_LIMIT_MB=str(CACHE_MB),
            DECISION_MAX_PROMPT_TOKENS="8192",
        )
        self.process = subprocess.Popen([sys.executable, SERVER, self.lock, grace], env=environment, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise AssertionError("server exited early: %s" % self.process.stderr.read().decode())
            try:
                self.get("/health")
                return
            except (urllib.error.URLError, ConnectionError):
                time.sleep(0.05)
        raise AssertionError("server never became healthy")

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.process.kill()
            self.process.wait()
        if self.process is not None and self.process.stderr is not None:
            self.process.stderr.close()

    def close(self):
        self.stop()
        shutil.rmtree(self.root, ignore_errors=True)

    def get(self, path):
        with urllib.request.urlopen("http://127.0.0.1:%d%s" % (self.port, path), timeout=5) as response:
            return json.loads(response.read())

    def post(self, path, body, key=KEY):
        request = urllib.request.Request("http://127.0.0.1:%d%s" % (self.port, path), data=json.dumps(body).encode(), method="POST", headers={"Content-Type": "application/json", **({"Authorization": "Bearer " + key} if key else {})})
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())


QUESTIONS = {
    "q": {"type": "noul", "instructions": "Is it about money?"},
    "t": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments", "shipping": "Delivery"}},
}


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.sandbox = Sandbox()
        self.addCleanup(self.sandbox.close)

    def test_memory_is_capped_before_the_model_loads(self):
        self.sandbox.start()
        health = self.sandbox.get("/health")
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["memory"]["limit_mb"], MEMORY_MB)
        self.assertEqual(health["memory"]["cache_limit_mb"], CACHE_MB)

    def test_answers_in_the_jev_shape(self):
        self.sandbox.start()
        status, body = self.sandbox.post("/v1/systemone", {"state": "A duplicate charge.", "questions": QUESTIONS})
        self.assertEqual(status, 200)
        self.assertLess(body["answers"]["q"]["noul"], 0.1)
        self.assertEqual(body["answers"]["t"]["choice"], "billing")
        self.assertGreater(body["answers"]["t"]["confidence"], 0.9)
        self.assertEqual(body["model"], "Intern-Decision-4B")
        self.assertEqual(body["usage"]["output_tokens"], 0)

    def test_input_over_the_context_is_refused_without_echoing_it(self):
        self.sandbox.start()
        status, body = self.sandbox.post("/v1/systemone", {"state": "x" * 9000, "questions": QUESTIONS})
        self.assertEqual(status, 413)
        self.assertIn("too long", body["detail"])
        self.assertNotIn("xxxx", body["detail"])

    def test_a_missing_or_wrong_token_is_refused(self):
        self.sandbox.start()
        for key in (None, "wrong"):
            status, body = self.sandbox.post("/v1/systemone", {"state": "x", "questions": QUESTIONS}, key=key)
            self.assertEqual(status, 401)
            self.assertIn("bearer", body["detail"])

    def test_unknown_paths_are_not_found(self):
        self.sandbox.start()
        status, _ = self.sandbox.post("/v1/other", {})
        self.assertEqual(status, 404)

    def test_the_model_is_prepared_once_and_a_restart_reuses_it(self):
        self.sandbox.start()
        prepared = os.path.join(self.sandbox.model, "prepared")
        with open(os.path.join(prepared, "READY")) as handle:
            self.assertEqual(handle.read(), REVISION)
        self.assertEqual(os.listdir(self.sandbox.model), ["prepared"])
        self.sandbox.stop()
        source = os.path.join(self.sandbox.stub, "source")
        shutil.rmtree(source, ignore_errors=True)
        self.sandbox.start()
        self.assertFalse(os.path.exists(source), "a prepared model is not downloaded again")

    def test_a_server_without_an_owner_exits_after_the_grace_period(self):
        self.sandbox.start(grace="1")
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and self.sandbox.process.poll() is None:
            time.sleep(0.1)
        self.assertEqual(self.sandbox.process.poll(), 0)

    def test_a_server_with_a_live_owner_keeps_running(self):
        import fcntl

        with open(self.sandbox.lock, "a") as owner:
            fcntl.flock(owner, fcntl.LOCK_EX)
            self.sandbox.start(grace="1")
            time.sleep(3)
            self.assertIsNone(self.sandbox.process.poll())
            fcntl.flock(owner, fcntl.LOCK_UN)


if __name__ == "__main__":
    unittest.main()
