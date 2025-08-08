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
"""Script that gets the most up to date supported public models."""

from pathlib import Path

from lightning_sdk.llm.public_assistants import PUBLIC_MODELS

models = sorted(PUBLIC_MODELS.keys())

code = f"""# AUTO-GENERATED. Do not edit.
from typing import Final, Literal

MODELS: Final = ({", ".join(repr(m) for m in models)},)
ModelLiteral = Literal[{", ".join(repr(m) for m in models)}]
"""

Path("litai/_generated_models.py").write_text(code)
print("Wrote litai/_generated_models.py")
