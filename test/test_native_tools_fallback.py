from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import b4_local_agent_llm as b4


class NativeToolsFallbackTests(unittest.TestCase):
    def test_falls_back_to_prompt_json_when_native_template_is_unsupported(self) -> None:
        prompt_result = {
            "raw_text": '{"content":"fallback answer","tool_calls":[]}',
            "parsed_candidate": {"content": "fallback answer", "tool_calls": []},
            "ai_message": {"role": "assistant", "content": "fallback answer", "tool_calls": []},
            "status": "success",
            "error": None,
            "attempts": [{"attempt_index": 1}],
        }

        with patch.object(
            b4,
            "_generate_with_retry",
            side_effect=[
                RuntimeError("native_tools mode requires tokenizer.apply_chat_template(..., tools=...) support"),
                prompt_result,
            ],
        ):
            result = b4.generate_ai_message(
                str(ROOT / "configs" / "model.yaml"),
                [{"role": "user", "content": "test"}],
                [],
                "native_tools",
            )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["ai_message"]["content"], "fallback answer")
        self.assertEqual(result["raw_record"]["mode"], "native_tools")
        self.assertEqual(result["raw_record"]["effective_mode"], "prompt_json")
        self.assertEqual(result["raw_record"]["fallback"]["to"], "prompt_json")


if __name__ == "__main__":
    unittest.main()
