<div align='center'>

<h1> ⚡ LitAI </h1>

**Easiest way to access any AI model with a single subscription using Python.**

&#160;

</div>

Every AI model is better at some tasks than others, and we have to switch between them. This requires subscriptions to multiple LLM providers and is costly. LitAI lets you use any LLM provider (both proprietary and open-source) under a single subscription.

Easily switch between any AI model, save costs, and track usage through a unified dashboard.

&#160;

<div align='center'>
<pre>
✅ Access any AI model      ✅ Usage dashboard            ✅ Single subscription        
✅ Bring your own model     ✅ Easily switch across LLMs  ✅ 20+ public models          
✅ Track LLM token usage    ✅ Easy setup                 ✅ No MLOps glue code         
</pre>
</div>  

<div align='center'>

[![PyPI Downloads](https://static.pepy.tech/badge/litai)](https://pepy.tech/projects/litai)
[![Discord](https://img.shields.io/discord/1077906959069626439?label=Get%20help%20on%20Discord)](https://discord.gg/WajDThKAur)
![cpu-tests](https://github.com/Lightning-AI/litai/actions/workflows/ci-testing.yml/badge.svg)
[![codecov](https://codecov.io/gh/Lightning-AI/litai/graph/badge.svg?token=SmzX8mnKlA)](https://codecov.io/gh/Lightning-AI/litai)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Lightning-AI/litai/blob/main/LICENSE)

</div>

<p align="center">
  <a href="https://lightning.ai/">Lightning AI</a> •
  <a href="https://lightning.ai/docs/litai">Docs</a> •
  <a href="#quick-start">Quick start</a>
</p>

______________________________________________________________________

# Quick Start

Install LitAI via pip ([more options](https://lightning.ai/docs/litai/home/install)):

```bash
pip install litai
```

## Run on a Studio

When running inside Lightning Studio, you can use any available LLM out of the box — no extra setup required.

```python
from litai import LLM

llm = LLM(model="openai/gpt-4")
print(llm.chat("who are you?"))
# I'm an AI by OpenAI
```

## Run locally (outside Studio)

To use LitAI outside of Lightning Studio, you'll need to explicitly provide your teamspace name.

The teamspace input format is: `"owner-name/teamspace-name"` (e.g. `"username/my-team"` or `"org-name/team-name"`)

```python
from litai import LLM

llm = LLM(model="openai/gpt-4", teamspace="owner-name/teamspace-name")
print(llm.chat("who are you?"))
# I'm an AI by OpenAI
```

# Key benefits

A few key benefits:

- Supports 20+ public models
- Bring your own model
- Keeps chat logs
- Optional guardrails
- Usage dashboard

# Features

✅ [Concurrency with async](https://lightning.ai/docs/litai/features/async-litai/)\
✅ [Fallback and retry](https://lightning.ai/docs/litai/features/fallback-retry/)\
✅ [Switch models](https://lightning.ai/docs/litai/features/models/)\
✅ [Multi-turn conversation logs](https://lightning.ai/docs/litai/features/multi-turn-conversation/)\
✅ [Streaming](https://lightning.ai/docs/litai/features/streaming/)

# Advanced features

## Concurrency with async

LitAI supports asynchronous execution, allowing you to handle multiple requests concurrently without blocking. This is especially useful in high-throughput applications like chatbots, APIs, or agent loops.

To enable async behavior, set `enable_async=True` when initializing the `LLM` class. Then use `await llm.chat(...)` inside an `async` function.

```python
import asyncio
from litai import LLM


async def main():
    llm = LLM(model="openai/gpt-4", teamspace="lightning-ai/litai", enable_async=True)
    print(await llm.chat("who are you?"))


if __name__ == "__main__":
    asyncio.run(main())
```

## Streaming

Stream the model response as it's being generated.

```python
from litai import LLM

llm = LLM(model="openai/gpt-4")
for chunk in llm.chat("hello", stream=True):
    print(chunk, end="", flush=True)
```

## Conversations

Keep chat history across multiple turns so the model remembers context.
This is useful for assistants, summarizers, or research tools that need multi-turn chat history.

Each conversation is identified by a unique name. LitAI stores conversation history separately for each name.

```python
from litai import LLM

llm = LLM(model="openai/gpt-4")

# Continue a conversation across multiple turns
llm.chat("What is Lightning AI?", conversation="intro")
llm.chat("What can it do?", conversation="intro")

print(llm.get_history("intro"))  # View all messages from the 'intro' thread
llm.reset_conversation("intro")  # Clear conversation history
```

Create multiple named conversations for different tasks.

```python
from litai import LLM

llm = LLM(model="openai/gpt-4")

llm.chat("Summarize this text", conversation="summarizer")
llm.chat("What's a RAG pipeline?", conversation="research")

print(llm.list_conversations())
```

## Switch models

Use the best model for each task.
LitAI lets us dynamically switch models at request time.

We set a default model when initializing `LLM` and override it with the `model` parameter only when needed.

```python
from litai import LLM

llm = LLM(model="openai/gpt-4")

# Uses the default model (openai/gpt-4)
print(llm.chat("Who created you?"))
# >> I am a large language model, trained by OpenAI.

# Override the default model for this request
print(llm.chat("Who created you?", model="google/gemini-2.5-flash"))
# >> I am a large language model, trained by Google.

# Uses the default model again
print(llm.chat("Who created you?"))
# >> I am a large language model, trained by OpenAI.
```

## Fallbacks and retries

Ensure reliable responses even if a model is unavailable.\
LitAI automatically retries requests and switches to fallback models in order.

- Fallback models are tried in the order provided.
- Each model gets up to `max_retries` attempts independently.
- The first successful response is returned immediately.
- If all models fail after their retry limits, LitAI raises an error.

```python
from litai import LLM

llm = LLM(
    model="openai/gpt-4",
    fallback_models=["google/gemini-2.5-flash", "anthropic/claude-3-5-sonnet-20240620"],
    max_retries=4,
)

print(llm.chat("How do I fine-tune an LLM?"))
```
