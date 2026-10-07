import fcntl
import hmac
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_QUESTIONS = 64
MAX_STATE_CHARS = 50000
MAX_BODY_BYTES = 2 * 1024 * 1024
DECISION_TOKEN = "<decision>"
ANSWER_SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
TEMPERATURE = 1.99241824
QUANT_BITS = "4"
QUANT_GROUP = "64"
SOURCE_FILES = ["model-language-*.safetensors", "model.safetensors.index.json", "config.json", "tokenizer.json", "tokenizer_config.json", "added_tokens.json", "special_tokens_map.json", "vocab.json", "merges.txt", "chat_template.jinja", "generation_config.json"]
SUPPORT_FILES = ["added_tokens.json", "special_tokens_map.json", "vocab.json", "merges.txt"]
SYSTEM_PROMPT = "You are a careful decision assistant. Use the state and decision schema in the user message to make the requested decisions. For every field, choose exactly one answer symbol (e.g. A, B, C, ...) from its listed options and return one valid JSON object mapping each field name to its chosen symbol. Use the field names and symbols exactly as given. Do not include explanations, Markdown, or extra text."


def owner_alive(path):
    try:
        with open(path, "a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(handle, fcntl.LOCK_UN)
            return False
    except OSError:
        return False


def watch(path, grace):
    orphaned = None
    while True:
        time.sleep(1)
        if owner_alive(path):
            orphaned = None
        elif orphaned is None:
            orphaned = time.monotonic()
        elif time.monotonic() - orphaned >= grace:
            os._exit(0)


class Failure(Exception):
    def __init__(self, status, detail):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def request_body(headers, stream):
    length = headers.get("Content-Length")
    if length is None:
        raise Failure(411, "request body needs a Content-Length")
    try:
        size = int(length)
    except ValueError:
        raise Failure(400, "invalid Content-Length")
    if size < 0 or size > MAX_BODY_BYTES:
        raise Failure(413, "request body too large")
    try:
        body = json.loads(stream.read(size))
    except ValueError:
        raise Failure(400, "request body must be valid JSON")
    if not isinstance(body, dict) or not isinstance(body.get("questions"), dict) or not body["questions"]:
        raise Failure(400, "request body must be an object with a 'questions' object")
    state = body.get("state")
    if len(body["questions"]) > MAX_QUESTIONS:
        raise Failure(413, "too many questions (%d > %d)" % (len(body["questions"]), MAX_QUESTIONS))
    if len(state if isinstance(state, str) else json.dumps(state)) > MAX_STATE_CHARS:
        raise Failure(413, "state too large")
    return state, body["questions"]


def options(question):
    kind = question.get("type")
    criteria = question.get("criteria")
    if kind == "choice":
        if not isinstance(criteria, dict) or not criteria:
            raise ValueError("choice criteria must be a non-empty object")
        return [(str(key), str(value)) for key, value in criteria.items()]
    if kind == "score":
        if isinstance(criteria, list) and criteria:
            return [(str(index), str(value)) for index, value in enumerate(criteria)]
        raise ValueError("score criteria must be a non-empty list")
    if kind == "noul":
        return [("no", "The answer is no (negative, or disagree with the claim)."), ("yes", "The answer is yes (affirmative, or align with the claim).")]
    raise ValueError("question type must be noul, choice or score")


def compile_request(state, questions):
    fields, symbols, lines = [], {}, []
    for field, question in questions.items():
        if not isinstance(question, dict):
            raise ValueError("question %s is not an object" % field)
        choices = options(question)
        if len(choices) > len(ANSWER_SYMBOLS):
            raise ValueError("at most %d options per question" % len(ANSWER_SYMBOLS))
        fields.append(str(field))
        symbols[str(field)] = ANSWER_SYMBOLS[: len(choices)]
        lines.append("%s: %s" % (field, question.get("instructions", "")))
        for symbol, (value, description) in zip(symbols[str(field)], choices):
            lines.append("    %s = %s: %s" % (symbol, value, description))
    text = "Return one answer for every field using the supplied answer symbols.\n\n## State\n%s\n## Decision schema\n%s" % (json.dumps(state, ensure_ascii=False, indent=2), "\n".join(lines))
    if DECISION_TOKEN in text:
        raise ValueError("the reserved decision marker appears in the input")
    skeleton = json.dumps(dict.fromkeys(fields, DECISION_TOKEN), ensure_ascii=False, indent=4)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}, {"role": "assistant", "content": skeleton}]
    return messages, fields, symbols


def answer(kind, choices, scores):
    top = max(scores)
    weights = [math.exp((value - top) / TEMPERATURE) for value in scores]
    total = sum(weights)
    names = [value for value, _ in choices]
    probabilities = {name: round(weight / total, 6) for name, weight in zip(names, weights)}
    best = min(names, key=lambda name: (-probabilities[name], name))
    if kind == "noul":
        return {"type": "noul", "noul": round(probabilities["yes"], 4)}
    if kind == "score":
        return {"type": "score", "score": round(sum(float(name) * probabilities[name] for name in names), 4), "confidence": round(probabilities[best], 4), "probabilities": probabilities}
    return {"type": "choice", "choice": best, "confidence": round(probabilities[best], 4), "probabilities": probabilities}


class Engine:
    def __init__(self, mx, tokenizer, language, name, max_tokens):
        self.mx = mx
        self.tokenizer = tokenizer
        self.language = language
        self.name = name
        self.max_tokens = max_tokens
        self.marker = tokenizer.convert_tokens_to_ids(DECISION_TOKEN)

    def decide(self, state, questions):
        messages, fields, symbols = compile_request(state, questions)
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False, add_vision_id=True)
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) > self.max_tokens:
            raise Failure(413, "local model input is too long (%d > %d tokens); shorten state, question instructions or criteria" % (len(ids), self.max_tokens))
        positions = [index - 1 for index, token in enumerate(ids) if token == self.marker]
        if len(positions) != len(fields) or any(position < 0 for position in positions):
            raise ValueError("decision marker count does not match the questions")
        mx = self.mx
        body = self.language.model
        hidden = body(mx.array(ids)[None])[0]
        rows = hidden[mx.array(positions)]
        logits = self.language.lm_head(rows) if hasattr(self.language, "lm_head") else body.embed_tokens.as_linear(rows)
        mx.eval(logits)
        answers = {}
        for index, field in enumerate(fields):
            question = questions[field]
            picks = mx.array([self.tokenizer.encode(symbol, add_special_tokens=False)[0] for symbol in symbols[field]])
            scores = [float(value) for value in logits[index][picks].astype(mx.float32).tolist()]
            answers[field] = answer(question["type"], options(question), scores)
        return {"model": self.name, "answers": answers, "usage": {"input_tokens": len(ids), "output_tokens": 0}}


def mlx_memory(mx, limit, cache_limit):
    return {
        "limit_mb": limit >> 20,
        "cache_limit_mb": cache_limit >> 20,
        "active_mb": mx.get_active_memory() >> 20,
        "cache_mb": mx.get_cache_memory() >> 20,
        "peak_mb": mx.get_peak_memory() >> 20,
    }


def prepare(repo, revision, target):
    ready = os.path.join(target, "READY")
    if os.path.exists(ready):
        return
    from huggingface_hub import snapshot_download

    source = snapshot_download(repo, revision=revision, allow_patterns=SOURCE_FILES)
    staging = target + ".partial"
    shutil.rmtree(staging, ignore_errors=True)
    done = subprocess.run([sys.executable, "-m", "mlx_lm", "convert", "--hf-path", source, "--mlx-path", staging, "-q", "--q-bits", QUANT_BITS, "--q-group-size", QUANT_GROUP], stdin=subprocess.DEVNULL)
    if done.returncode != 0:
        shutil.rmtree(staging, ignore_errors=True)
        raise SystemExit("could not prepare the local decision model")
    for name in SUPPORT_FILES:
        if os.path.exists(os.path.join(source, name)) and not os.path.exists(os.path.join(staging, name)):
            shutil.copy(os.path.join(source, name), staging)
    shutil.rmtree(target, ignore_errors=True)
    os.rename(staging, target)
    with open(ready, "w") as handle:
        handle.write(revision)
    for name in ("hub", "xet"):
        shutil.rmtree(os.path.join(os.environ["HF_HOME"], name), ignore_errors=True)


def serve_mlx():
    import mlx.core as mx

    limit = int(os.environ["DECISION_MEMORY_LIMIT_MB"]) << 20
    cache_limit = int(os.environ["DECISION_CACHE_LIMIT_MB"]) << 20
    mx.set_memory_limit(limit)
    mx.set_cache_limit(cache_limit)

    repo = os.environ["DECISION_REPO"]
    target = os.environ["DECISION_MODEL_DIR"]
    prepare(repo, os.environ["DECISION_REVISION"], target)

    from mlx_lm.utils import load_model
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(target, local_files_only=True)
    model, _ = load_model(__import__("pathlib").Path(target))
    language = getattr(model, "language_model", model)
    engine = Engine(mx, tokenizer, language, repo.rsplit("/", 1)[-1], int(os.environ["DECISION_MAX_PROMPT_TOKENS"]))
    expected = ("Bearer " + os.environ["DECISION_API_KEY"]).encode()
    gate = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, value):
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                return self.reply(200, {"status": "ok", "loaded": [repo], "device": "gpu", "memory": mlx_memory(mx, limit, cache_limit)})
            self.reply(404, {"detail": "not found"})

        def do_POST(self):
            if self.path != "/v1/systemone":
                return self.reply(404, {"detail": "not found"})
            supplied = (self.headers.get("Authorization") or "").encode("utf-8", "surrogateescape")
            if not hmac.compare_digest(supplied, expected):
                return self.reply(401, {"detail": "invalid or missing bearer token"})
            try:
                state, questions = request_body(self.headers, self.rfile)
                with gate:
                    result = engine.decide(state, questions)
            except Failure as failure:
                return self.reply(failure.status, {"detail": failure.detail})
            except ValueError as error:
                return self.reply(422, {"detail": str(error)})
            except Exception as error:
                sys.stderr.write("inference failed: %s: %s\n" % (type(error).__name__, str(error)[:300]))
                sys.stderr.flush()
                return self.reply(500, {"detail": "inference failed"})
            self.reply(200, result)

    server = ThreadingHTTPServer((os.environ["DECISION_HOST"], int(os.environ["DECISION_PORT"])), Handler)
    server.daemon_threads = True
    server.serve_forever()


def main():
    lock = sys.argv[1]
    grace = float(sys.argv[2])
    threading.Thread(target=watch, args=(lock, grace), daemon=True).start()
    serve_mlx()


if __name__ == "__main__":
    main()
