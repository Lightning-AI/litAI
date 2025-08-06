# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
import selectors
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Union

import psutil
import pytest
from tqdm import tqdm

from litai import LLM
from litai.llm import SDKLLM
from litai.tools import LitTool


class MockLLM(LLM):
    """A mock LLM to test the README code.

    We only want to check the correctness for README.
    We replace the LLM class with this mock class to avoid loading the real model.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        fallback_models: Optional[List[str]] = None,
        teamspace: Optional[str] = None,
        max_retries: int = 3,
        api_key: Optional[str] = None,
        enable_async: Optional[bool] = False,
        verbose: int = 0,
        full_response: Optional[bool] = None,
    ) -> None:
        # Skip parent initialization to avoid model loading
        self._model = model
        self._enable_async = enable_async

    def chat(  # noqa: D417
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 500,
        images: Optional[Union[List[str], str]] = None,
        conversation: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        stream: bool = False,
        tools: Optional[List[LitTool]] = None,
        **kwargs: Any,
    ) -> str:
        # Return a mock response for chat
        if self._enable_async:

            async def _chat():
                return "Hello, world!"

            return _chat()

        return "Hello, world!"

    def _model_call(
        self,
        model: SDKLLM,
        prompt: str,
        system_prompt: Optional[str],
        max_completion_tokens: int,
        images: Optional[Union[List[str], str]],
        conversation: Optional[str],
        metadata: Optional[Dict[str, str]],
        stream: bool,
        full_response: Optional[bool] = None,
        **kwargs: Any,
    ) -> str:
        if stream:
            yield from iter(["Hello, world!"])
        else:
            return "Hello, world!"

    def get_history(self, name: str, raw: bool = False) -> List[Dict[str, str]] | None:
        return []

    def list_conversations(self) -> List[str]:
        return []

    def reset_conversation(self, name: str) -> None:
        return None

    def call_tool(self, response: str, tools: Optional[List[LitTool]] = None) -> Optional[str]:
        return "Final response"


@pytest.fixture
def killall():
    def _run(process):
        parent = psutil.Process(process.pid)
        for child in parent.children(recursive=True):
            child.kill()
        process.kill()

    return _run


def _extract_code_blocks(lines: List[str]) -> List[str]:
    language = "python"
    regex = re.compile(
        r"(?P<start>^```(?P<block_language>(\w|-)+)\n)(?P<code>.*?\n)(?P<end>```)",
        re.DOTALL | re.MULTILINE,
    )
    blocks = [(match.group("block_language"), match.group("code")) for match in regex.finditer("".join(lines))]
    return [block for block_language, block in blocks if block_language == language]


def _get_code_blocks(file: str) -> List[str]:
    with open(file) as f:
        lines = list(f)
        return _extract_code_blocks(lines)


def _replace_llm_with_mockllm(code: str) -> str:
    """Replace LLM with MockLLM in the code block."""
    # Replace LLM class usage with MockLLM
    code = re.sub(r"\bLLM\b", "MockLLM", code)

    # Remove any existing litai imports that include LLM
    code = re.sub(r"from litai import.*?LLM.*?\n", "", code)
    code = re.sub(r"from litai import.*?LLM.*?$", "", code, flags=re.MULTILINE)

    # Add import for MockLLM
    if "from tests.test_readme import MockLLM" not in code:
        code = "from tests.test_readme import MockLLM\nfrom litai import tool, LitTool\n" + code
        code = "import sys\nsys.path.append('.')\n" + code

    return code


def _run_script_with_timeout(file, timeout, extra_time=0, killall=None):
    sel = selectors.DefaultSelector()
    try:
        process = subprocess.Popen(
            [sys.executable, str(file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,  # Line-buffered
            universal_newlines=True,  # Decode bytes to string
        )

        stdout_lines = []
        stderr_lines = []
        end_time = time.time() + timeout + extra_time

        sel.register(process.stdout, selectors.EVENT_READ)
        sel.register(process.stderr, selectors.EVENT_READ)

        while True:
            timeout_remaining = end_time - time.time()
            if timeout_remaining <= 0:
                killall(process)
                break

            events = sel.select(timeout=timeout_remaining)
            for key, _ in events:
                if key.fileobj is process.stdout:
                    line = process.stdout.readline()
                    if line:
                        stdout_lines.append(line)
                elif key.fileobj is process.stderr:
                    line = process.stderr.readline()
                    if line:
                        stderr_lines.append(line)

            if process.poll() is not None:
                break

        output = "".join(stdout_lines)
        errors = "".join(stderr_lines)

        # Get the return code of the process
        returncode = process.returncode

    except Exception as e:
        output = ""
        errors = str(e)
        returncode = -1  # Indicate failure in running the process

    return returncode, output, errors


def test_readme(tmp_path, killall):
    d = tmp_path / "readme_codes"
    d.mkdir(exist_ok=True)
    code_blocks = _get_code_blocks("README.md")
    assert len(code_blocks) > 0, "No code block found in README.md"

    for i, code in enumerate(tqdm(code_blocks)):
        # Replace LLM with MockLLM in the code block
        if "from litai import LLM" not in code:
            print(f"Skipping non LitAI code block: {i}")
            continue
        modified_code = _replace_llm_with_mockllm(code)

        file = d / f"{i}.py"
        file.write_text(modified_code)
        returncode, stdout, stderr = _run_script_with_timeout(file, timeout=5, extra_time=0.1, killall=killall)
        assert returncode == 0, (
            f"Code[{i}] exited with {returncode}.\nError: {stderr}\n"
            f"Please check the code for correctness:\n```\n{modified_code}\n```"
        )
