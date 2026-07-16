from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

from b3_tool_layer import execute_tool_calls, get_tools_schema
from skills.code_executor import SandboxExecutionError, SandboxViolation, code_executor


class CodeExecutorTests(unittest.TestCase):
    def test_basic_toolset_exposes_code_executor(self) -> None:
        schema = get_tools_schema(str(ROOT / "configs" / "tools.yaml"), "basic_tools")
        names = [item["function"]["name"] for item in schema]
        self.assertIn("code_executor", names)

    def test_sums_primes_up_to_300(self) -> None:
        result = code_executor(
            """
def is_prime(number):
    if number < 2:
        return False
    return all(number % divisor for divisor in range(2, int(number ** 0.5) + 1))

sum(number for number in range(2, 301) if is_prime(number))
"""
        )

        self.assertEqual(result["result"], 8275)

    def test_supports_multi_step_sequence_calculation(self) -> None:
        result = code_executor(
            "values = [value * value for value in range(1, 101)]\n"
            "sum(value for value in values if value % 3 == 0)"
        )

        self.assertEqual(result["result"], 112761)

    def test_still_blocks_file_and_system_access(self) -> None:
        with self.assertRaises(SandboxExecutionError):
            code_executor("open('forbidden.txt', 'w')")
        with self.assertRaises(SandboxViolation):
            code_executor("import os\nos.getcwd()", allowed_imports=["os"])

    def test_basic_toolset_executes_prime_sum_through_tool_layer(self) -> None:
        code = (
            "def prime(number):\n"
            "    return number > 1 and all(number % divisor for divisor in range(2, int(number ** 0.5) + 1))\n"
            "sum(number for number in range(2, 301) if prime(number))"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            messages = execute_tool_calls(
                [{"id": "prime_sum", "name": "code_executor", "args": {"code": code}}],
                str(ROOT / "configs" / "tools.yaml"),
                "basic_tools",
                temp_dir,
            )

        payload = json.loads(messages[0]["content"])
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["output"]["result"], 8275)


if __name__ == "__main__":
    unittest.main()
