<div align='center'>

<h2>
  Chat with any AI model with one line of Python.
  <br/>
  Build agents, chatbots, and apps that just work with no downtime.
</h2>    

<img alt="Lightning" src="https://github.com/user-attachments/assets/0d0b40a7-d7b9-4b59-a0b6-51ba865e5211" width="800px" style="max-width: 100%;">

&#160;

</div>

LitAI is the easiest way to chat with any model (ChatGPT, Anthropic, etc) in one line of Python. LitAI handles retries, fallback, billing, and logging - so you can build agents, chatbots, or apps without managing flaky APIs or writing wrapper code.

&#160;

<div align='center'>
<pre>
✅ Use any AI model (OpenAI, etc.) ✅ 20+ public models  ✅ Bring your model API keys
✅ Unified usage dashboard         ✅ No subscription    ✅ Auto retries and fallback
✅ Deploy dedicated models on-prem ✅ Start instantly    ✅ No MLOps glue code       
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
  <a href="#examples">Examples</a> •
  <a href="#performance">Performance</a> •
  <a href="#faq">FAQ</a> •
  <a href="https://lightning.ai/docs/litai">Docs</a>
</p>

______________________________________________________________________

# Quick Start

Install LitAI via pip ([more options](https://lightning.ai/docs/litai/home/install)):

```bash
pip install litai
```
Add AI to any Python program in 3 lines:   

```python
from litai import LLM

llm = LLM(model="openai/gpt-4")
answer = llm.chat("who are you?")
print(answer)

# I'm an AI by OpenAI
```

# Examples
What we ***love*** about LitAI is that you can build agents, chatbots and apps in plain Python without heavy frameworks or magical abstractions. Agents can be just simple Python programs with a few decisions made by a model.   

### Agent
Here's a simple agent that tells you the latest news

```python
import re, requests
from litai import LLM

llm = LLM(model="openai/gpt-4o")

website_url = "https://text.npr.org/"
website_text = re.sub(r'<[^>]+>', ' ', requests.get(website_url).text)

response = llm.chat(f"Based on this, what is the latest: {website_text}")
print(response)
```

### Agentic if statement
We believe the best way to build agents is to use normal Python programs with "agentic IF statements". This way, 90% of the flow is still deterministic
with a few decisions made by a model. This avoids complex abstractions that agentic frameworks introduce with make agents unreliable and hard to debug.

```python
from litai import LLM

llm = LLM(model="openai/gpt-4o")

product_review = "give me a short, bad product review about a tv"
response = llm.chat(f"Is this review good or bad? Reply only with 'good' or 'bad': {product_review}").strip().lower()

if response == "good":
    print("✅ good review")
else:
    print("❌ bad review")
```

# Key features
Monitor usage and manage spend via the model dashboard on [Lightning AI](https://lightning.ai/).   

<div align='center'>
<img alt="Lightning" src="https://github.com/user-attachments/assets/b1e7049c-c7b0-42f3-a43c-c1e156929f50" width="800px" style="max-width: 100%;">
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

Model APIs can flake or can have outages. LitAI automatically retries in case of failures. After multiple failures it can automatically fallback to other models in case the provider is down.

```python
from litai import LLM

llm = LLM(
    model="openai/gpt-4",
    fallback_models=["google/gemini-2.5-flash", "anthropic/claude-3-5-sonnet-20240620"],
    max_retries=4,
)

print(llm.chat("How do I fine-tune an LLM?"))
```

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
  <summary>Use your own client (Like OpenAI)</summary>
  
If you have your own client (Like OpenAI), simply switch the URL to Lightning AI.
```python
from openai import OpenAI

client = OpenAI(
  base_url="https://lightning.ai/api/v1",
  api_key="LIGHTNING_API_KEY",
)

completion = client.chat.completions.create(
  model="openai/gpt-4o",
  messages=[
    {
      "role": "user",
      "content": "What is the meaning of life?"
    }
  ]
)

print(completion.choices[0].message.content)
```
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

<br/>

# FAQ



<details>
  <summary>Do I need a subscription to use LitAI? (Nope) </summary>
   
Nope. You can start instantly without a subscription. LitAI is pay-as-you-go and lets you use your own model API keys (like OpenAI, Anthropic, etc.).
</details>

<details>
  <summary>Do I need an OpenAI account?  (Nope)</summary>

Nope. You get access to all models and all model providers without a subscription.   
</details>

<details>
  <summary>What happens if a model API fails or goes down? </summary>

LitAI automatically retries the same model and can fall back to other models you specify. You’ll get the best chance of getting a response, even during outages.
</details>

<details>
  <summary>Can I bring my own API keys for OpenAI, Anthropic, etc.? (Yes)</summary>

Yes! You can plug in your own keys to any OpenAI compatible API 
</details>

<details>
  <summary>Can I connect private models? (Yes)</summary>

Yes! You can connect any endpoint that supports the OpenAI spec.   
</details>

<details>
  <summary>Can you deploy a dedicated, private model like Llama for me? (Yes)</summary>

Yes! We can deploy dedicated models on any cloud (Lambda, AWS, etc).
</details>

<details>
  <summary>Can you deploy models on-prem? (Yes)</summary>

Yes! We can deploy on any dedicated VPC on the cloud or your own physical data center.
</details>

<details>
  <summary>Do deployed models support Kubernetes? (Yes)</summary>

Yes! We can use the Lightning AI orchestrator custom built for AI or Kubernetes, whatever you want!
</details>

<details>
  <summary>How do I pay for the model APIs?</summary>

Buy Lightning AI credits on Lightning to pay for the APIs.
</details>

<details>
  <summary>Do you add fees?</summary>

At this moment we don't add fees on top of the API calls, but that might change in the future!
</details>

<details>
  <summary>Are you SOC2, HIPAA compliant? (Yes)</summary>

LitAI is built by Lightning AI. Our enterprise AI platform powers teams all the way from Fortune 100 to startups. Our platform is fully SOC2, HIPAA compliant.   
</details>


# Community
LitAI is a [community project accepting contributions](https://lightning.ai/docs/litai/community) - Let's make the world's most advanced AI routing engine.

💬 [Get help on Discord](https://discord.com/invite/XncpTy7DSt)    
📋 [License: Apache 2.0](https://github.com/Lightning-AI/litAI/blob/main/LICENSE)     

