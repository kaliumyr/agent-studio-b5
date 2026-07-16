import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AI_WEB_DIR = PROJECT_ROOT / "ai_web"
if str(AI_WEB_DIR) not in sys.path:
    sys.path.insert(0, str(AI_WEB_DIR))

import server_1


class WebJobConcurrencyTests(unittest.TestCase):
    def test_second_job_initializes_while_first_job_is_running(self) -> None:
        first_started = threading.Event()
        release_first = threading.Event()
        second_started = threading.Event()

        def fake_run_agent(input_path, _tools, _memory, _model, outdir, _mode, _resume):
            if Path(input_path).stem == "first":
                first_started.set()
                release_first.wait(timeout=2)
            else:
                second_started.set()
            return {"outdir": outdir}

        with tempfile.TemporaryDirectory() as temp_dir_str, patch.dict(
            sys.modules,
            {"b1_agent_runtime_1": type("Runtime", (), {"run_agent": staticmethod(fake_run_agent)})},
        ):
            temp_dir = Path(temp_dir_str)
            first_dir = temp_dir / "first_out"
            second_dir = temp_dir / "second_out"
            first_dir.mkdir()
            second_dir.mkdir()
            first_input = temp_dir / "first.json"
            second_input = temp_dir / "second.json"
            first_input.write_text("{}", encoding="utf-8")
            second_input.write_text("{}", encoding="utf-8")

            server_1._start_background_job(first_dir, first_input, "mock", False, "first")
            self.assertTrue(first_started.wait(timeout=1))
            server_1._start_background_job(second_dir, second_input, "mock", False, "second")
            self.assertTrue(second_started.wait(timeout=1))
            release_first.set()

            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                with server_1.JOB_LOCK:
                    jobs_done = all(
                        server_1.JOB_REGISTRY[str(path.resolve())].get("status") == "finished"
                        for path in (first_dir, second_dir)
                    )
                if jobs_done:
                    break
                time.sleep(0.01)
            self.assertTrue(jobs_done)


if __name__ == "__main__":
    unittest.main()
