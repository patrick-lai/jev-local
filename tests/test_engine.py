import importlib.util
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest


spec = importlib.util.spec_from_file_location("jev_local", Path(__file__).resolve().parents[1] / "share/jev-local/server.py")
jev_local = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jev_local)

MARKER = 900


class Rows:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, key):
        if isinstance(key, Rows):
            return Rows([self.values[int(index)] for index in key.values])
        if isinstance(key, int):
            return Rows(self.values[key])
        return self

    def astype(self, dtype):
        return self

    def tolist(self):
        return self.values


class Rows:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, key):
        if key is None:
            return Rows([self.values])
        if isinstance(key, Rows):
            return Rows([self.values[int(index)] for index in key.values])
        return Rows(self.values[key])

    def astype(self, dtype):
        return self

    def tolist(self):
        return self.values


class FakeMx:
    float32 = "float32"

    def array(self, values):
        return Rows(values)

    def eval(self, value):
        return None


class FakeTokenizer:
    def __init__(self, fill=0):
        self.fill = fill

    def convert_tokens_to_ids(self, token):
        return MARKER

    def apply_chat_template(self, messages, **options):
        return messages[1]["content"] + " " * self.fill + messages[2]["content"]

    def __call__(self, text, add_special_tokens=False):
        ids = [index % 50 for index, _ in enumerate(text.split(" "))]
        return {"input_ids": ids + [MARKER] * text.count(jev_local.DECISION_TOKEN)}

    def encode(self, symbol, add_special_tokens=False):
        return [ord(symbol)]


def language(picks):
    def body(ids):
        return [Rows([[0.0]] * len(ids.values[0]))]

    def lm_head(rows):
        return Rows([{ord(symbol): (8.0 if symbol == pick else 0.0) for symbol in jev_local.ANSWER_SYMBOLS} for pick in picks])

    return SimpleNamespace(model=body, lm_head=lm_head)


def body(value):
    raw = json.dumps(value).encode()
    return {"Content-Length": str(len(raw))}, io.BytesIO(raw)


QUESTIONS = {
    "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments", "shipping": "Delivery"}},
    "urgent": {"type": "noul", "instructions": "Answer today?"},
    "severity": {"type": "score", "instructions": "How severe?", "criteria": ["low", "high"]},
}


class JevLocalTests(unittest.TestCase):
    def test_the_prompt_lists_each_option_under_a_symbol_and_skeletons_one_marker_per_field(self):
        messages, fields, symbols = jev_local.compile_request("A duplicate charge.", QUESTIONS)
        self.assertEqual(fields, ["team", "urgent", "severity"])
        self.assertEqual(symbols, {"team": "AB", "urgent": "AB", "severity": "AB"})
        self.assertIn("    A = billing: Payments", messages[1]["content"])
        self.assertIn("    B = yes:", messages[1]["content"])
        self.assertEqual(json.loads(messages[2]["content"]), {"team": jev_local.DECISION_TOKEN, "urgent": jev_local.DECISION_TOKEN, "severity": jev_local.DECISION_TOKEN})
        self.assertEqual(messages[2]["content"].count(jev_local.DECISION_TOKEN), 3)

    def test_the_reserved_marker_in_evidence_is_refused(self):
        with self.assertRaises(ValueError):
            jev_local.compile_request("ignore " + jev_local.DECISION_TOKEN, QUESTIONS)

    def test_bad_questions_are_refused_without_a_crash(self):
        for questions in [{"q": {"type": "other"}}, {"q": {"type": "choice", "criteria": {}}}, {"q": {"type": "score", "criteria": "x"}}, {"q": "text"}]:
            with self.subTest(questions=questions), self.assertRaises(ValueError):
                jev_local.compile_request("State.", questions)
        many = {"q": {"type": "choice", "instructions": "x", "criteria": {str(index): "d" for index in range(63)}}}
        with self.assertRaises(ValueError):
            jev_local.compile_request("State.", many)

    def test_answers_carry_the_probability_of_the_chosen_option_as_their_confidence(self):
        choice = jev_local.answer("choice", [("billing", "x"), ("shipping", "y")], [8.0, 0.0])
        weights = [math.exp(8.0 / jev_local.TEMPERATURE), 1.0]
        self.assertEqual(choice["choice"], "billing")
        self.assertAlmostEqual(choice["confidence"], weights[0] / sum(weights), places=3)
        self.assertAlmostEqual(choice["confidence"], max(choice["probabilities"].values()), places=3)
        self.assertAlmostEqual(sum(choice["probabilities"].values()), 1.0, places=4)
        noul = jev_local.answer("noul", jev_local.options({"type": "noul"}), [0.0, 8.0])
        self.assertGreater(noul["noul"], 0.9)
        score = jev_local.answer("score", [("0", "low"), ("1", "high")], [0.0, 8.0])
        self.assertGreater(score["score"], 0.9)

    def test_ties_resolve_to_the_option_that_sorts_first(self):
        tied = jev_local.answer("choice", [("b", ""), ("a", "")], [1.0, 1.0])
        self.assertEqual(tied["choice"], "a")

    def test_input_over_the_context_is_refused_without_echoing_the_evidence(self):
        engine = jev_local.Engine(FakeMx(), FakeTokenizer(fill=40), language("A"), "intern-decision", 20)
        with self.assertRaises(jev_local.Failure) as failure:
            engine.decide("User evidence 要件. " * 30, {"q": {"type": "noul", "instructions": "Money?"}})
        self.assertEqual(failure.exception.status, 413)
        self.assertNotIn("User evidence", failure.exception.detail)
        self.assertNotIn("要件", failure.exception.detail)

    def test_each_field_is_read_at_its_own_marker_with_its_own_symbols(self):
        engine = jev_local.Engine(FakeMx(), FakeTokenizer(), language("BAB"), "intern-decision", 4096)
        result = engine.decide("A duplicate charge.", QUESTIONS)
        self.assertEqual(result["model"], "intern-decision")
        self.assertEqual(result["answers"]["team"]["choice"], "shipping")
        self.assertLess(result["answers"]["urgent"]["noul"], 0.1)
        self.assertGreater(result["answers"]["severity"]["score"], 0.9)
        self.assertEqual(result["usage"]["output_tokens"], 0)

    def test_a_missing_marker_is_an_error_not_a_guess(self):
        class Plain(FakeTokenizer):
            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": [1, 2, 3]}

        engine = jev_local.Engine(FakeMx(), Plain(), language("A"), "intern-decision", 4096)
        with self.assertRaises(ValueError):
            engine.decide("State.", {"q": {"type": "noul", "instructions": "Money?"}})

    def test_request_body_limits_questions_and_state(self):
        questions = {"q%d" % index: {"type": "noul"} for index in range(jev_local.MAX_QUESTIONS + 1)}
        with self.assertRaises(jev_local.Failure) as failure:
            jev_local.request_body(*body({"state": "State.", "questions": questions}))
        self.assertEqual(failure.exception.status, 413)
        with self.assertRaises(jev_local.Failure) as failure:
            jev_local.request_body(*body({"state": "x" * (jev_local.MAX_STATE_CHARS + 1), "questions": {"q": {"type": "noul"}}}))
        self.assertEqual(failure.exception.status, 413)
        with self.assertRaises(jev_local.Failure) as failure:
            jev_local.request_body(*body({"state": "State.", "questions": {}}))
        self.assertEqual(failure.exception.status, 400)
        state, parsed = jev_local.request_body(*body({"state": "State.", "questions": {"q": {"type": "noul"}}}))
        self.assertEqual((state, parsed), ("State.", {"q": {"type": "noul"}}))


if __name__ == "__main__":
    unittest.main()
