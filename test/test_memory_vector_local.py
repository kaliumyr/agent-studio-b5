from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from b5_memory import VECTOR_DIMENSIONS, _search_by_vector


class LocalMemoryVectorTests(unittest.TestCase):
    def test_ranks_documents_without_embedding_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_root = Path(temp_dir)
            (memory_root / "agent.md").write_text(
                "Agent 使用工具调用读取文件并完成外部任务。",
                encoding="utf-8",
            )
            (memory_root / "weather.md").write_text(
                "今天阳光充足，适合户外活动。",
                encoding="utf-8",
            )
            index = {
                "agent": {
                    "memory_type": "global",
                    "title": "Agent 工具调用",
                    "summary": "模型使用外部工具完成任务",
                    "path": "agent.md",
                },
                "weather": {
                    "memory_type": "global",
                    "title": "天气记录",
                    "summary": "每日天气",
                    "path": "weather.md",
                },
            }

            results = _search_by_vector(
                index,
                "模型怎样使用外部工具完成任务",
                memory_root,
                top_k=2,
            )

        self.assertTrue(results)
        self.assertEqual(results[0]["memory_id"], "agent")
        self.assertGreaterEqual(results[0]["similarity"], 0.01)
        self.assertEqual(VECTOR_DIMENSIONS, 256)


if __name__ == "__main__":
    unittest.main()
