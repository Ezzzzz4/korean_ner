import ast
import html
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def load_render_helpers():
    source = APP_PATH.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(APP_PATH))
    wanted_names = {
        "MAX_LENGTH",
        "MAX_USER_CHARS",
        "ENTITY_COLORS",
        "ENTITY_LABELS",
        "_escape_text",
        "_escape_markdown_inline_code",
        "_label_at",
        "_entity_parts",
        "create_length_error_html",
        "create_output_html",
    }
    body = []

    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if names & wanted_names:
                body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in wanted_names:
            body.append(node)

    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)

    namespace = {"html": html}
    exec(compile(module, str(APP_PATH), "exec"), namespace)
    return namespace


class AppRenderTests(unittest.TestCase):
    def test_output_html_escapes_plain_text_without_entities(self):
        helpers = load_render_helpers()
        rendered = helpers["create_output_html"]("<script>alert(1)</script>", list("<script>alert(1)</script>"), {}, [])

        self.assertNotIn("<script>", rendered)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", rendered)

    def test_output_html_escapes_highlighted_text_and_entity_cards(self):
        helpers = load_render_helpers()
        text = "<b>A</b>"
        labels = {0: "B-PS", 1: "I-PS", 2: "I-PS", 3: "I-PS", 4: "I-PS", 5: "I-PS", 6: "I-PS", 7: "I-PS"}
        rendered = helpers["create_output_html"](text, list(text), labels, [(text, "PS", 0)])

        self.assertNotIn("<b>", rendered)
        self.assertIn("&lt;b&gt;A&lt;/b&gt;", rendered)

    def test_markdown_inline_code_escape_keeps_backticks_literal(self):
        helpers = load_render_helpers()

        self.assertEqual(helpers["_escape_markdown_inline_code"]("a`<b>"), r"a\`&lt;b&gt;")

    def test_app_launches_on_localhost_only(self):
        source = APP_PATH.read_text(encoding="utf-8-sig")

        self.assertIn('share=False', source)
        self.assertIn('server_name="127.0.0.1"', source)
        self.assertNotIn('server_name="0.0.0.0"', source)

    def test_app_uses_lazy_runtime_loading(self):
        source = APP_PATH.read_text(encoding="utf-8-sig")
        tree = ast.parse(source, filename=str(APP_PATH))
        top_level_calls = [
            node
            for node in ast.walk(ast.Module(body=tree.body, type_ignores=[]))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "from_pretrained"
        ]
        runtime_func = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_runtime_components")

        self.assertEqual(len(top_level_calls), 1)
        self.assertIn(top_level_calls[0], list(ast.walk(runtime_func)))
        self.assertIn('os.environ.get("KOREAN_NER_DEVICE", "cpu")', source)
        self.assertIn('MAX_LENGTH = 192', source)
        self.assertNotIn("F1: 85.9%", source)

    def test_failed_checkpoint_load_does_not_publish_partial_model(self):
        source = APP_PATH.read_text(encoding="utf-8-sig")
        tree = ast.parse(source, filename=str(APP_PATH))
        runtime_func = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_runtime_components")
        module = ast.Module(body=[runtime_func], type_ignores=[])
        ast.fix_missing_locations(module)
        model = MagicMock()
        checkpoint_loader = MagicMock(side_effect=FileNotFoundError("missing checkpoint"))
        namespace = {
            "_TOKENIZER": None,
            "_MODEL": None,
            "_DEVICE": None,
            "_RUNTIME_LOCK": threading.Lock(),
            "AutoTokenizer": SimpleNamespace(from_pretrained=lambda _name: object()),
            "MODEL_NAME": "unused",
            "MODEL_PATH": "missing.pt",
            "NUM_LABELS": 3,
            "resolve_device": lambda: "cpu",
            "load_checkpoint": checkpoint_loader,
            "extract_model_state_dict": lambda state: state,
        }
        exec(compile(module, str(APP_PATH), "exec"), namespace)

        with patch("korean_ner.KoElectraNER", return_value=model):
            with self.assertRaisesRegex(FileNotFoundError, "missing checkpoint"):
                namespace["get_runtime_components"]()
            self.assertIsNone(namespace["_MODEL"])
            self.assertIsNone(namespace["_DEVICE"])

            checkpoint_loader.side_effect = None
            checkpoint_loader.return_value = {"weights": "loaded"}
            _tokenizer, loaded_model, device = namespace["get_runtime_components"]()
            self.assertIs(loaded_model, model)
            self.assertEqual(device, "cpu")
            model.load_state_dict.assert_called_once_with({"weights": "loaded"}, strict=True)


if __name__ == "__main__":
    unittest.main()
