<div align='center'>

<h2>
  The easiest way to use any AI model.
  <br/>
  No subscription. Better uptime. Unified billing.
</h2>    

<img alt="Lightning" src="https://github.com/user-attachments/assets/0d0b40a7-d7b9-4b59-a0b6-51ba865e5211" width="800px" style="max-width: 100%;">

&#160;

</div>

Using multiple AI models is painful - different APIs, multiple subscriptions, downtime, and runaway costs. LitAI gives you one interface for any model - OpenAI, Anthropic, open-source, or your own - with automatic fallback, usage logging, and usage monitoring built into a single platform.

&#160;

<div align='center'>
<pre>
✅ Use any AI model         ✅ Unified usage dashboard  ✅ No subscription   
✅ Bring your own model     ✅ Smart model fallback     ✅ 20+ public models 
✅ Deploy dedicated models  ✅ Start instantly          ✅ No MLOps glue code
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
  <a href="#quick-start">Quick start</a> •
  <a href="#key-features">Features</a> •
  <a href="https://lightning.ai/">Lightning AI</a> •
  <a href="#performance">Performance</a> •
  <a href="https://lightning.ai/docs/litai">Docs</a>
</p>

______________________________________________________________________

# Quick Start

Install LitAI via pip ([more options](https://lightning.ai/docs/litai/home/install)):

```bash
pip install litai
```

## Run
Add AI to any Python program in 3 lines:   

```python
from litai import LLM

llm = LLM(model="openai/gpt-4")
print(llm.chat("who are you?"))
# I'm an AI by OpenAI
```

# Key features
Monitor usage and manage spend via the model dashboard on [Lightning AI](https://lightning.ai/).   

<div align='center'>
<img alt="Lightning" src="https://github.com/user-attachments/assets/702bb8aa-7948-4602-b9a3-8f55224eb116" width="800px" style="max-width: 100%;">
</div>

✅ [Use over 20+ models (ChatGPT, Claude, etc...)](https://lightning.ai/)    
✅ [Monitor all usage in one place](https://lightning.ai/model-apis)    
✅ [Async support](https://lightning.ai/docs/litai/features/async-litai/)     
✅ [Auto retries on failure](https://lightning.ai/docs/litai/features/fallback-retry/)    
✅ [Auto model switch on failure](https://lightning.ai/docs/litai/features/fallback-retry/)    
✅ [Switch models](https://lightning.ai/docs/litai/features/models/)    
✅ [Multi-turn conversation logs](https://lightning.ai/docs/litai/features/multi-turn-conversation/)    
✅ [Streaming](https://lightning.ai/docs/litai/features/streaming/)    
✅ Bring your own model (connect your API keys, coming soon...)    
✅ Chat logs (coming soon...)    

<br/>

# Advanced features

### Auto fallbacks and retries

Model APIs can flake or can have outages. LitAI LitAI automatically retries in case of failures. After multiple failures it can automatically fallback to other models in case the provider is down.

```python
from litai import LLM

llm = LLM(
    model="openai/gpt-4",
    fallback_models=["google/gemini-2.5-flash", "anthropic/claude-3-5-sonnet-20240620"],
    max_retries=4,
)

print(llm.chat("How do I fine-tune an LLM?"))
```

Details:  
- Fallback models are tried in the order provided.
- Each model gets up to `max_retries` attempts independently.
- The first successful response is returned immediately.
- If all models fail after their retry limits, LitAI raises an error.


<details>
  <summary>Streaming</summary>

Real-time chat applications benefit from showing words as they generate which gives the illusion of faster speed to the user.  Streaming
is the mechanism that allows you to do this.

```python
from litai import LLM

llm = LLM(model="openai/gpt-4")
for chunk in llm.chat("hello", stream=True):
    print(chunk, end="", flush=True)
````
</details>

<details>
  <summary>Concurrency with async</summary>

Advanced Python programs that process multiple requests at once rely on "async" to do this. LitAI can work with async libraries without blocking calls. This is especially useful in high-throughput applications like chatbots, APIs, or agent loops.   

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

</details>


<details>
  <summary>Multi-turn conversations</summary>

Models only know the message that was sent to them. To enable them to respond with memory of all the messages sent to it so far, track the related
message under the same conversation.  This is useful for assistants, summarizers, or research tools that need multi-turn chat history.

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
</details>

<details>
  <summary>Switch models on each call</summary>

In certain applications you may want to call ChatGPT in one message and Anthropic in another so you can use the best model for each task. 
LitAI lets you dynamically switch models at request time.

Set a default model when initializing `LLM` and override it with the `model` parameter only when needed.

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
</details>

<br/>

# Performance
LitAI does smart routing across a global network of servers. As a result, it only adds 25ms of overhead for an API call.   

