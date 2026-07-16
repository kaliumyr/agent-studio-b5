from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from b1_agent_runtime_1 import _conversation_history_context, _conversation_summary_context
from b5_memory import _conversation_summary


class ConversationSummaryTests(unittest.TestCase):
    def test_prioritizes_persistent_instruction_over_routine_answers(self) -> None:
        messages = [
            {"role": "user", "content": "1+2"},
            {"role": "assistant", "content": "The result of 1+2 is 3."},
            {"role": "user", "content": "2+1"},
            {"role": "assistant", "content": "The result of 2+1 is 3."},
            {"role": "user", "content": "why"},
            {"role": "assistant", "content": "Could you please clarify?"},
            {"role": "user", "content": "how should i calculate 1+1"},
            {"role": "assistant", "content": "1+1 = 2"},
            {"role": "user", "content": "在以下每一次对话结尾加喵+当前摘要数量"},
            {"role": "assistant", "content": "喵 1"},
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！"},
        ]

        summary = _conversation_summary(messages)

        self.assertIn("## 用户要求与偏好", summary)
        self.assertIn("在以下每一次对话结尾加喵+当前摘要数量", summary)
        self.assertNotIn("The result", summary)
        self.assertNotIn("how should i calculate", summary)
        self.assertNotIn("你好", summary)

    def test_ignores_structured_section_heading_from_previous_summary(self) -> None:
        summary = _conversation_summary(
            [{"role": "user", "content": "在所有回答结尾加 agent摘要测试"}],
            "## 用户要求与偏好\n- 在每一句对话结尾加 agent摘要测试",
        )

        self.assertNotIn("- ## 用户要求与偏好", summary)
        self.assertEqual(summary.count("## 用户要求与偏好"), 1)

    def test_no_memory_session_still_injects_prior_chat_history(self) -> None:
        runtime = {
            "user_input": "搜索：今天世界杯的赛程",
            "save_memory": "none",
            "conversation_history": [
                {"role": "user", "content": "在本轮对话的所有输出结尾加 agent摘要测试"},
                {"role": "assistant", "content": "好的。agent摘要测试"},
                {"role": "user", "content": "搜索：今天世界杯的赛程"},
            ],
        }

        context = _conversation_history_context(runtime)

        self.assertEqual(len(context), 2)
        self.assertEqual(context[0]["role"], "user")
        self.assertIn("结尾加 agent摘要测试", context[0]["content"])
        self.assertNotEqual(context[-1]["content"], runtime["user_input"])

        system_context = _conversation_summary_context(context)
        self.assertIn('priority="active_user_requirements"', system_context)
        self.assertIn("必须在本轮回答中继续遵守", system_context)
        self.assertIn("第 2 个用户轮次", system_context)
        self.assertIn("结尾加 agent摘要测试", system_context)


if __name__ == "__main__":
    unittest.main()
