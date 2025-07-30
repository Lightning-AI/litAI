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
from inspect import Signature, signature
from typing import Any, Callable, Dict, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class LitTool(BaseModel):
    """A tool is a function that can be used to interact with the world."""

    model_config = ConfigDict(extra="allow")

    name: Optional[str] = Field(default="")
    description: Optional[str] = Field(default="")

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the tool and call setup."""
        super().__init__(**kwargs)
        self.setup()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize the tool."""
        super().__init_subclass__(**kwargs)
        if cls.run.__doc__ is not None:
            cls.description = cls.run.__doc__.strip()

        cls.name = "".join(["_" + c.lower() if c.isupper() else c for c in cls.__name__]).lstrip("_")

    def setup(self) -> None:
        """Use this method to store states."""
        pass

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the tool."""
        raise NotImplementedError("Subclasses must implement this method")

    def _get_signature(self) -> Signature:
        """Get the signature to use for parameter extraction. Override this in subclasses."""
        return signature(self.run)

    def _extract_parameters(self) -> Dict[str, Any]:
        sig = self._get_signature()
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
        """
        if json_mode:
            return json.dumps(self.as_tool(), indent=2)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._extract_parameters(),
        }


def tool(func: Optional[Callable] = None) -> Union[LitTool, Callable]:
    """Decorator to convert a function into a LitTool instance.

    Can be used as @tool or @tool().

    Args:
        func: The function to convert into a tool.

    Returns:
        LitTool: An instance of LitTool that wraps the function.

    Example:
        @tool
        def get_weather(location: str) -> str:
            return f"The weather in {location} is sunny"

        # Or with parentheses:
        @tool()
        def get_weather(location: str) -> str:
            return f"The weather in {location} is sunny"
    """

    def _create_tool(f: Callable) -> LitTool:
        class FunctionTool(LitTool):
            def run(self, *args: Any, **kwargs: Any) -> Any:
                return f(*args, **kwargs)

            def _get_signature(self) -> Signature:
                """Override to return the signature of the wrapped function."""
                return signature(f)

        FunctionTool.__name__ = f.__name__
        tool_instance = FunctionTool()
        tool_instance.name = "".join(["_" + c.lower() if c.isupper() else c for c in f.__name__]).lstrip("_")
        if f.__doc__:
            tool_instance.description = f.__doc__.strip()

        return tool_instance

    if func is None:
        # Called as @tool()
        return _create_tool
    # Called as @tool
    return _create_tool(func)
