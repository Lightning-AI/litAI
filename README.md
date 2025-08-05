<div align='center'>

<h2>
  Chat with any AI model with one line of Python.
  <br/>
  Build agents and apps in pure Python with full control and zero magic.
</h2>    

<img alt="Lightning" src="https://github.com/user-attachments/assets/0d0b40a7-d7b9-4b59-a0b6-51ba865e5211" width="800px" style="max-width: 100%;">

&#160;

</div>

LitAI is the easiest way to chat with any model (ChatGPT, Anthropic, etc) in one line of Python. It handles retries, fallback, billing, and logging - so you can build agents, chatbots, or apps without managing flaky APIs or heavy, opaque frameworks. Tool use is built in, with no magic - just clean, testable, raw Python code.

&#160;

<div align='center'>
<pre>
✅ Use any AI model (OpenAI, etc.) ✅ Create & share tools ✅ 20+ public models      
✅ Bring your model API keys       ✅ No subscription      ✅ Unified usage dashboard
✅ Auto retries and fallback       ✅ No MLOps glue code   ✅ Start instantly        
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
  <a href="#tools-docs">Tools</a> •
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
Chat with any AI model in 3 lines:

```python
from litai import LLM

llm = LLM(model="openai/gpt-4")
answer = llm.chat("who are you?")
print(answer)

# I'm an AI by OpenAI
```
<br/>

# Why LitAI for agents?
Agent frameworks are great for demos - but they’re hard to learn, full of custom syntax, and often unpredictable. They get you up and running fast, but at the cost of heavy abstractions, hidden logic, and behavior that's hard to debug. Even simple things like tool calls or prompt formatting get rewritten behind the scenes. Teams eventually hit a wall and rebuild everything in plain Python just to make it work.

With LitAI there’s nothing new to learn - just write raw Python and sprinkle in `.chat()`, `.if_()`, or `.classify()` as needed to make the program smart and agentic. LitAI gives you lightweight, minimal building blocks you’d end up building yourself: model calls, retries, fallbacks, tool use, memory, streaming - all in clean, testable code. You promps stay pure without YAML, wrappers or surprises. Just code that works the way you expect.

[Compare vs agent frameworks](https://lightning.ai/docs/litai/home/why-litai#comparisons)   

<br/>

# Examples
What we ***love*** about LitAI is that if you know Python, you already know LitAI - just sprinkle a few "smart" agent decisions.

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
We believe the best way to build agents is with normal Python programs and simple **“agentic if statements.”** 
That way, 90% of the logic stays deterministic, and the model only steps in when needed. No complex abstractions, no framework magic - just code you can trust and debug.

```python
from litai import LLM

llm = LLM(model="openai/gpt-3.5-turbo")

product_review = "This TV is terrible."
response = llm.chat(f"Is this review good or bad? Reply only with 'good' or 'bad': {product_review}").strip().lower()

if response == "good":
    print("good review")
else:
    print("bad review")
```

### Shortcuts
Agentic workflows mostly come down to agentic-if statements or classification decisions. While you can use `llm.chat` yourself to do it,
we provide 2 simple shortcuts

```python
from litai import LLM
llm = LLM()

# shortcut for agentic if statement (can do this yourself with llm.chat if needed)
product_review = "This TV is terrible."
if llm.if_(product_review, "is this a positive review?"):
    print("good review")
else:
    print("bad review")

# shortcut for agentic classification (can do this yourself with llm.chat if needed)
sentiment = llm.classify("This movie was awful.", ["positive", "negative"])
print("Sentiment:", sentiment)
```

### Tools ([docs](https://lightning.ai/docs/litai/features/tools))
Tools allow models to get real-world data or take actions. In LitAI, there is no magic with tool use, agents can decide to call tools (`auto_call_tools=True`), or you can manually call a tool with `llm.call_tool(...)` for full control. Zero magic, just plain Python.

```python
from litai import LLM, tool

@tool
def get_weather(location: str):
    return f"The weather in {location} is sunny"

llm = LLM(model="openai/gpt-4")

result = llm.chat("What's the weather in Tokyo?", tools=[get_weather], auto_call_tools=True)
# The weather in Tokyo is sunny

chosen_tool = llm.chat("What's the weather in Tokyo?", tools=[get_weather])
result = llm.call_tool(chosen_tool, tools=[get_weather])
# The weather in London is sunny
```

Choose automatic or manual tool calling based on production needs. `auto_call_tools=True` is great for quick demos, but can obscure when and why a tool runs which can lead to surprises in production. `llm.call_tool(...)` gives you full control to decide when tools execute, making it easier to log, debug, test, and audit. This clarity is critical for reliability, safety, and trust in real-world systems.

<br/>

# Key features
Track usage and spending in your [Lightning AI](https://lightning.ai/) dashboard. Model calls are paid for with Lightning AI credits.

<div align='center'>
<pre>
✅ No subscription     ✅ 15 free credits (~37M tokens)      ✅ Pay as you go for more credits
</pre>
</div>  


<div align='center'>
<img alt="Lightning" src="https://github.com/user-attachments/assets/b1e7049c-c7b0-42f3-a43c-c1e156929f50" width="800px" style="max-width: 100%;">
</div>

<br/>

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

### Auto fallbacks and retries ([docs](https://lightning.ai/docs/litai/features/fallback-retry))

Model APIs can flake or can have outages. LitAI automatically retries in case of failures. After multiple failures it can automatically fallback to other models in case the provider is down.

```python
from litai import LLM

llm = LLM(
    model="openai/gpt-4",
    fallback_models=["google/gemini-2.5-flash", "anthropic/claude-3-5-sonnet-20240620"],
    max_retries=4,
)

print(llm.chat("What is a fun fact about space?"))
```

<br/>

<details>
  <summary>Tools</summary>

<br/>
  
Models can only reply with text, but tool calling lets them get real-world data or act, like checking calendars or sending messages, which allows AI apps to actually do things, not just talk. There are 2 ways to create tools in LitAI.

`@tool`: Turn any function into a tool with `litai.tool` decorator - useful when you just need a quick, simple tool.   

```python
from litai import LLM, tool

@tool
def get_weather(location: str):
    return f"The weather in {location} is sunny"

llm = LLM(model="openai/gpt-4")

chosen_tool = llm.chat("What's the weather in Tokyo?", tools=[get_weather])

result = llm.call_tool(chosen_tool, tools=[get_weather])
# The weather in London is sunny
```
  
`LitTool`: For more production-ready tools that encapsulate more logic, maintain state and can be shared across programs, use `LitTool`: 

```python
from litai import LLM, LitTool

class FAQTool(LitTool):
    def setup(self):
        self.faq = {
            "pricing": "You can view our pricing plans on the website.",
            "support": "Our support team is available 24/7 via chat.",
            "refund": "Refunds are available within 30 days of purchase."
        }

    def run(self, question: str) -> str:
        keyword = question.lower()
        for topic, answer in self.faq.items():
            if topic in keyword:
                return answer
        return "Sorry, I couldn't find an answer for that."

tool = FAQTool()

llm = LLM(model="openai/gpt-4")
response = llm.chat("How do I get a refund?", tools=[tool])
result = llm.call_tool(response, tools=[tool])

print(result)  # → "Refunds are available within 30 days of purchase."
```

##### Note: LitAI also supports any tool that is a pydantic BaseModel.
</details>

<details>
  <summary>Streaming</summary>

<br/>

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
  <summary>Use your own client (like OpenAI)</summary>

<br/>

For those who already have their own SDK to call LLMs (like the OpenAI sdk), you can still use LitAI via the `https://lightning.ai/api/v1` endpoint,
which will track usage, billing, etc...

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
      "content": "What is a fun fact about space?"
    }
  ]
)

print(completion.choices[0].message.content)
```
</details>

<details>
  <summary>Concurrency with async</summary>

<br/>

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

<br/>

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

<br/>

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

<details>
  <summary>Multiple models, same conversation</summary>

<br/>

One application of LitAI is to reduce costs of chats by using separate models for the same conversation. For example, use a cheap model to answer
the first question and a more expensive model for something that requires more intelligence.

```python
from litai import LLM

llm = LLM(model="openai/gpt-4")

# use a cheap model for this question
llm.chat("Is this a number or word: '5'", model="google/gemini-2.5-flash", conversation="story")

# go back to the expensive model
llm.chat("Create a story about that number like Lord of the Rings", conversation="story")

print(llm.get_history("story"))  # View all messages from the 'story' thread
```

</details>

<br/>

# Performance
LitAI does smart routing across a global network of servers - it only adds 25ms of overhead for an API call.   

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

Yes. You can plug in your own keys to any OpenAI compatible API 
</details>

<details>
  <summary>Can I connect private models? (Yes)</summary>

Yes. You can connect any endpoint that supports the OpenAI spec.   
</details>

<details>
  <summary>Can you deploy a dedicated, private model like Llama for me? (Yes)</summary>

Yes. We can deploy dedicated models on any cloud (Lambda, AWS, etc).
</details>

<details>
  <summary>Can you deploy models on-prem? (Yes)</summary>

Yes. We can deploy on any dedicated VPC on the cloud or your own physical data center.
</details>

<details>
  <summary>Do deployed models support Kubernetes? (Yes)</summary>

Yes. We can use the Lightning AI orchestrator custom built for AI or Kubernetes, whatever you want!
</details>

<details>
  <summary>How do I pay for the model APIs?</summary>

Buy Lightning AI credits on Lightning to pay for the APIs.
</details>

<details>
  <summary>Do you add fees?</summary>

At this moment we don't add fees on top of the API calls, but that might change in the future.
</details>

<details>
  <summary>Are you SOC2, HIPAA compliant? (Yes)</summary>

LitAI is built by Lightning AI. Our enterprise AI platform powers teams all the way from Fortune 100 to startups. Our platform is fully SOC2, HIPAA compliant.   
</details>

<br/>

# Community
LitAI is a [community project accepting contributions](https://lightning.ai/docs/litai/community) - Let's make the world's most advanced AI routing engine.

💬 [Get help on Discord](https://discord.com/invite/XncpTy7DSt)    
📋 [License: Apache 2.0](https://github.com/Lightning-AI/litAI/blob/main/LICENSE)     

