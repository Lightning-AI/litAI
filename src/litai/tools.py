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
"""Tools for the LLM."""

import json
from inspect import signature
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, Field


class LitTool(BaseModel):
    """A tool is a function that can be used to interact with the world."""

    name: Optional[str] = Field(default="")
    description: Optional[str] = Field(default="")

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize the tool."""
        super().__init_subclass__(**kwargs)
        if cls.run.__doc__ is not None:
            cls.description = cls.run.__doc__.strip()

        cls.name = "".join(["_" + c.lower() if c.isupper() else c for c in cls.__name__]).lstrip("_")

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the tool."""
        raise NotImplementedError("Subclasses must implement this method")

    def _extract_parameters(self) -> Dict[str, Any]:
        sig = signature(self.run)
        return {
            "type": "object",
            "properties": {
                param.name: {"type": param.annotation.__name__ if param.annotation is not None else "string"}
                for param in sig.parameters.values()
            },
            "required": [param.name for param in sig.parameters.values() if param.default is param.empty],
        }

    def as_tool(self, json_mode: bool = False) -> Union[str, Dict[str, Any]]:
        """Returns the schema of the tool.
        If json_mode is True, returns the schema as a JSON string.
        Otherwise, returns the schema as a dictionary.
        """  # noqa: D205
        if json_mode:
            return json.dumps(self.as_tool(), indent=2)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._extract_parameters(),
        }
