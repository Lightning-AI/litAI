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
"""Unit tests for tools module."""

import pytest
from langchain_core.tools import tool as langchain_tool

from litai import LitTool, tool


@pytest.fixture
def basic_tool_class():
    class TestTool(LitTool):
        def run(self, message: str) -> str:
            """Echo the message."""
            return f"Echo: {message}"

    return TestTool


@pytest.fixture
def weather_tool_class():
    class WeatherTool(LitTool):
        def run(self, location: str, units: str = "celsius") -> str:
            """Get weather for location."""
            return f"Weather in {location} ({units})"

    return WeatherTool


def test_basic_tool_creation(basic_tool_class):
    tool_instance = basic_tool_class()
    assert tool_instance.name == "test_tool"
    assert tool_instance.description == "Echo the message."


def test_tool_schema_generation(weather_tool_class):
    tool_instance = weather_tool_class()
    schema = tool_instance.as_tool()

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "weather_tool"
    assert schema["function"]["description"] == "Get weather for location."
    assert schema["function"]["parameters"]["type"] == "object"
    assert "location" in schema["function"]["parameters"]["properties"]
    assert "units" in schema["function"]["parameters"]["properties"]
    assert schema["function"]["parameters"]["required"] == ["location"]


def test_tool_execution():
    class CalculatorTool(LitTool):
        def run(self, a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

    tool_instance = CalculatorTool()
    result = tool_instance.run(5, 3)
    assert result == 8, f"Expected 8, got {result}"


def test_basic_decorator_usage():
    @tool
    def get_weather(
        location, country: str = "US", fetch_temperature: bool = False, latitude: float = 0.0, longitude: float = 0.0
    ) -> str:
        """Get weather for a location."""
        return f"Weather in {location} is sunny"

    assert isinstance(get_weather, LitTool)
    assert get_weather.name == "get_weather"
    assert get_weather.description == "Get weather for a location."

    # Check schema structure
    schema = get_weather.as_tool()
    assert schema["type"] == "function"
    assert schema["function"]["parameters"]["type"] == "object"
    assert schema["function"]["parameters"]["properties"]["location"]["type"] == "string"
    assert schema["function"]["parameters"]["properties"]["country"]["type"] == "string"
    assert schema["function"]["parameters"]["properties"]["fetch_temperature"]["type"] == "boolean"
    assert schema["function"]["parameters"]["properties"]["latitude"]["type"] == "number"
    assert schema["function"]["parameters"]["properties"]["longitude"]["type"] == "number"
    assert schema["function"]["parameters"]["required"] == ["location"]


def test_decorator_with_parameters():
    @tool
    def calculate(x: int, y: float, operation: str = "add") -> float:
        """Perform calculation on two numbers."""
        if operation == "add":
            return x + y
        if operation == "multiply":
            return x * y
        return 0.0

    schema = calculate.as_tool()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "calculate"
    assert schema["function"]["description"] == "Perform calculation on two numbers."
    assert schema["function"]["parameters"]["type"] == "object"

    # Check parameter properties
    props = schema["function"]["parameters"]["properties"]
    assert len(props) == 3
    assert props["x"]["type"] == "integer"
    assert props["y"]["type"] == "number"
    assert props["operation"]["type"] == "string"

    # Check required parameters
    assert schema["function"]["parameters"]["required"] == ["x", "y"]


def test_decorator_execution():
    @tool
    def greet(name: str, greeting: str = "Hello") -> str:
        """Greet someone."""
        return f"{greeting}, {name}!"

    result = greet.run(name="Alice")
    assert result == "Hello, Alice!"

    result = greet.run(name="Bob", greeting="Hi")
    assert result == "Hi, Bob!"


def test_decorator_without_docstring():
    @tool
    def simple_func(value: str) -> str:
        return value.upper()

    assert simple_func.name == "simple_func"
    assert simple_func.description == ""

    # Check parameter structure
    schema = simple_func.as_tool()
    assert schema["function"]["parameters"]["properties"]["value"]["type"] == "string"
    assert schema["function"]["parameters"]["required"] == ["value"]


def test_decorator_json_mode():
    @tool
    def test_func(param: str) -> str:
        """Test function."""
        return param

    json_schema = test_func.as_tool(json_mode=True)
    assert isinstance(json_schema, str)
    assert "test_func" in json_schema
    assert "param" in json_schema


def test_decorator_with_parentheses():
    @tool()
    def get_status(service: str, detailed: bool = False) -> str:
        """Get service status."""
        if detailed:
            return f"Service {service} is running with full details"
        return f"Service {service} is running"

    assert get_status.name == "get_status"
    assert get_status.description == "Get service status."

    result = get_status.run(service="api")
    assert result == "Service api is running"

    result_detailed = get_status.run(service="db", detailed=True)
    assert result_detailed == "Service db is running with full details"

    schema = get_status.as_tool()
    assert schema["function"]["parameters"]["properties"]["service"]["type"] == "string"
    assert schema["function"]["parameters"]["properties"]["detailed"]["type"] == "boolean"
    assert schema["function"]["parameters"]["required"] == ["service"]
    assert isinstance(get_status, LitTool)


def test_tool_setup():
    class TestTool(LitTool):
        def setup(self) -> None:
            self.state = 1

    tool_instance = TestTool()
    assert tool_instance.state == 1, "State initialized with 1"
    tool_instance.state += 1
    assert tool_instance.state == 2, "State not incremented. Should be 2"


def test_from_langchain():
    @langchain_tool
    def get_weather(city: str) -> str:
        """Get the weather of a given city."""
        return f"Weather in {city} is sunny."

    lit_tool = LitTool.from_langchain(get_weather)
    assert isinstance(lit_tool, LitTool)
    assert lit_tool.name == "get_weather"
    assert lit_tool.description == "Get the weather of a given city."
    assert lit_tool.as_tool() == {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather of a given city.",
            "parameters": get_weather.args_schema.model_json_schema(),
        },
    }


def test_convert_tools_empty():
    lit_tools = LitTool.convert_tools([])
    assert len(lit_tools) == 0


def test_convert_tools_unsupported_type():
    def get_weather(city: str) -> str:
        """Get the weather of a given city."""
        return f"Weather in {city} is sunny."

    with pytest.raises(TypeError, match="Unsupported tool type: <class 'function'>"):
        LitTool.convert_tools([get_weather])
