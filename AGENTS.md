# AGENTS.md

This project is a lightweight Python SDK for building LLM-powered “agents” using a single, unified interface.
The primary abstraction is an LLM client with optional tool/function-calling support. There are no background
workers or external queues required—everything runs in-process.

What this document covers:
- The “agent” abstraction in this SDK (LLM + Tools)
- How to configure models and API keys
- How to run, debug, and test locally
- Reliability and observability practices built into the SDK

---

## 1) Overview

- Language/runtime: Python 3.11.13
- Package manager: pip (regular)
- Core concepts:
  - LLM: a chat interface with state, retries, fallbacks, and async support
  - Tools: structured function calling with automatic JSON schema
- Dependencies of note (SDK usage level): requests for HTTP, pydantic for tool schemas, pytest for tests
- External services: None required (only outbound HTTPS to your model endpoints)

Design principles:
- Simple and composable APIs
- OpenAI-compatible chat completions wire format
- Tool-first function calling with clear parameter schemas
- Practical reliability (retries, fallbacks, context-length awareness)
- Helpful errors and structured debug capabilities

---

## 2) Agent Abstraction

In this SDK, an “agent” is a configured LLM instance that can:
- Chat with a list of messages (system, user, assistant)
- Maintain conversation state (history)
- Invoke tools/functions with typed parameters
- Run synchronously or asynchronously
- Apply retries and fallbacks across multiple models

Key classes:
- LLM: main chat client and stateful agent wrapper
- LitTool: base class for defining tools
- @tool: decorator to turn a Python function into a tool easily
- LightningLLM: deprecated alias; use LLM instead

---

## 3) Model Configuration

You can select and configure models in two ways:

- Use a pre-registered model (via Models):
  - Provide the model key (e.g., "lightning/llama-4").
  - Ensure the environment variable required by that model is set for authentication.

- Provide your own configuration source (e.g., a config object or file you load):
  - Each model needs: name, URL (chat completions endpoint), an environment variable name for the API key, and optionally the model_name used by the server.

Typical usage:

- Pre-registered model (recommended to start):
  ```python
  from litai import LLM

  # Requires: export LLAMA_API_KEY="<your-api-key>"
  llm = LLM(model="lightning/llama-4")
  print(llm.chat("Hello!"))
  ```

---

## 4) Basic Usage

- Single-turn chat:
  ```python
  from litai import LLM
  llm = LLM(model="lightning/llama-4")
  answer = llm.chat("Who are you?")
  print(answer)
  ```

- Multi-turn chat with message roles:
  ```python
  from litai import LLM
  llm = LLM(model="lightning/llama-4")

  messages = [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Summarize the benefits of unit tests."},
  ]
  reply = llm.chat(messages)
  print(reply)
  ```

- Conversation state:
  ```python
  llm.chat("Hi!")                          # turn 1
  llm.chat("Remind me what I said?")       # turn 2
  history = llm.get_history()              # inspect messages so far
  llm.reset_conversation()                 # clear only this conversation
  conversations = llm.list_conversations() # list all conversation ids
  ```


Notes:
- The chat API accepts either a single string (converted to a user message) or a list of message dicts with roles: system, user, assistant.
- Responses are returned as strings by default; when tools are enabled, the model may propose a tool call instead of plain text (see Tools below).

---

## 5) Tools

Tools let the model call structured functions with typed parameters.

- Quick tools via decorator:
  ```python
  from litai import LLM, tool

  @tool
  def get_weather(location: str):
      "Return a fake weather string for the given location."
      return f"The weather in {location} is sunny"

  llm = LLM(model="lightning/llama-4")

  # Option A: automatic tool execution (fast prototyping)
  result = llm.chat("What's the weather in Tokyo?", tools=[get_weather], auto_call_tools=True)
  print(result)  # "The weather in Tokyo is sunny"

  # Option B: manual tool execution (full control, recommended for production)
  chosen_tool = llm.chat("What's the weather in Tokyo?", tools=[get_weather])
  result = llm.call_tool(chosen_tool, tools=[get_weather])
  print(result)
  ```

- Custom tool classes:
  ```python
  from litai import LLM, LitTool

  class Add(LitTool):
      def run(self, a: int, b: int) -> int:
          "Add two integers."
          return a + b

  llm = LLM(model="lightning/llama-4")
  tool_call = llm.chat("Add 2 and 3", tools=[Add()])
  print(llm.call_tool(tool_call, tools=[Add()]))  # 5
  ```

Tips:
- Automatic tool calling is convenient but can hide when/why a tool executed. Manual calling makes logging, testing, and auditing simpler.
- Tool parameter schemas are generated automatically from function signatures or LitTool.run().

---

## 6) Reliability (Retries and Fallbacks)

Configure retries and fallbacks directly on the LLM:

```python
from litai import LLM

# Configure with 3 retries and a fallback model
llm = LLM(
    model="lightning/llama-4",
    max_retries=3,
    fallback_models=["openai/gpt-3.5-turbo"]
)

# If the primary model fails (e.g., network error, rate limit), the SDK will retry up to 3 times.
# If all retries fail, it will automatically try the fallback model(s) in order.

response = llm.chat("Tell me a joke.")
print(response)
```