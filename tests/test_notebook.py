from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path


NOTEBOOK_PATH = Path("train_model.ipynb")
FORBIDDEN_TEXT = (
    "99.8",
    "SOTA",
    "FP16",
    "fp16",
    "RDrop",
    "R-Drop",
    "beat KLUE",
    "Beat KLUE",
    "torch.cuda",
    "--device cuda",
)


def load_notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return source


class NotebookTests(unittest.TestCase):
    def test_notebook_is_valid_json_with_empty_outputs(self):
        notebook = load_notebook()

        self.assertEqual(notebook["nbformat"], 4)
        self.assertGreaterEqual(len(notebook["cells"]), 4)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs"), [])

    def test_notebook_removes_legacy_claims_and_accelerator_autoselection(self):
        text = NOTEBOOK_PATH.read_text(encoding="utf-8")

        for phrase in FORBIDDEN_TEXT:
            self.assertNotIn(phrase, text)
        self.assertIn("historical experiment", text)
        self.assertIn("strict validation metrics", text)

    def test_code_cells_do_not_execute_training(self):
        notebook = load_notebook()

        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            tree = ast.parse(cell_source(cell))
            for node in ast.walk(tree):
                self.assertFalse(
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in {"run", "call", "check_call", "check_output", "Popen"},
                    "notebook code cells must not launch subprocesses",
                )
                self.assertFalse(
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id in {"exec", "eval"},
                    "notebook code cells must not execute dynamic code",
                )

    def test_notebook_demonstrates_cpu_train_py_commands_without_running_them(self):
        text = NOTEBOOK_PATH.read_text(encoding="utf-8")

        self.assertIn("train.py", text)
        self.assertIn("--device", text)
        self.assertIn("cpu", text)
        self.assertIn("--train-limit", text)
        self.assertIn("--eval-limit", text)


if __name__ == "__main__":
    unittest.main()
