from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import b4_local_agent_llm as b4
from b3_tool_layer import get_tools_schema
from b4_local_agent_llm import _deterministic_plan_tool_calls


class PlanToolFastPathTests(unittest.TestCase):
    def test_builds_local_search_call_for_explicit_docs_step(self) -> None:
        schema = get_tools_schema(str(ROOT / "configs" / "tools.yaml"), "basic_tools")
        calls = _deterministic_plan_tool_calls(
            "步骤1：搜索docs目录中关于agent的学习资料文件（如教程、文档、课程列表等）。",
            schema,
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "local_file_search")
        self.assertEqual(calls[0]["args"]["root_dir"], "docs")
        self.assertIn("agent", calls[0]["args"]["query"].casefold())
        self.assertEqual(calls[0]["args"]["top_k"], 8)

    def test_builds_local_search_call_for_model_generated_step(self) -> None:
        schema = get_tools_schema(str(ROOT / "configs" / "tools.yaml"), "basic_tools")
        calls = _deterministic_plan_tool_calls(
            "搜索docs目录中关于agent的学习资料",
            schema,
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "local_file_search")
        self.assertEqual(calls[0]["args"]["root_dir"], "docs")
        self.assertIn("agent", calls[0]["args"]["query"].casefold())

    def test_plan_execute_continues_from_planning_into_docs_search(self) -> None:
        schema = get_tools_schema(str(ROOT / "configs" / "tools.yaml"), "basic_tools")
        generated = [
            {
                "status": "success",
                "ai_message": {
                    "role": "assistant",
                    "content": json.dumps({"plan": ["搜索docs目录中关于agent的学习资料"]}, ensure_ascii=False),
                    "tool_calls": [],
                },
            },
            {
                "status": "success",
                "ai_message": {"role": "assistant", "content": "STEP_DONE:1:已找到 Agent 入门资料", "tool_calls": []},
            },
            {
                "status": "success",
                "ai_message": {"role": "assistant", "content": "建议先学习模型、工具和执行循环。", "tool_calls": []},
            },
        ]
        executed_calls = []

        def execute_tool_calls(calls, _tools_config, _toolset, _tool_outdir):
            executed_calls.extend(calls)
            result = {
                "skill_name": "local_file_search",
                "status": "success",
                "input": calls[0]["args"],
                "output": {"results": [{"path": "docs/agent.md", "snippet": "Agent 由模型、工具和执行循环组成。"}]},
                "error": None,
            }
            return [
                {
                    "role": "tool",
                    "tool_call_id": calls[0]["id"],
                    "name": calls[0]["name"],
                    "status": "success",
                    "content": json.dumps(result, ensure_ascii=False),
                }
            ]

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            b4, "generate_ai_message", side_effect=generated
        ):
            result = b4._run_plan_execute_impl(
                str(ROOT / "configs" / "model.yaml"),
                [
                    {"role": "system", "content": "你是本地文件学习助手。"},
                    {"role": "user", "content": "读取docs目录关于agent的文件，回答我：我想学习agent，该如何入门"},
                ],
                schema,
                str(ROOT / "configs" / "tools.yaml"),
                "basic_tools",
                "prompt_json",
                temp_dir,
                runtime_bridge={"execute_tool_calls": execute_tool_calls},
            )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["final_answer"], "建议先学习模型、工具和执行循环。")
        self.assertEqual(len(executed_calls), 1)
        self.assertEqual(executed_calls[0]["name"], "local_file_search")
        self.assertEqual(executed_calls[0]["args"]["root_dir"], "docs")


if __name__ == "__main__":
    unittest.main()
