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
"""LLM client class."""

import datetime
import json
import logging
import os
import threading
import warnings
from typing import Any, Dict, List, Optional, Union

import requests
from lightning_sdk.lightning_cloud import login
from lightning_sdk.llm import LLM as SDKLLM

from litai.tools import LitTool
from litai.utils import handle_http_error, verbose_http_error_log, verbose_sdk_error_log

CLOUDY_MODELS = {
    "openai/gpt-4o",
    "openai/gpt-4",
    "openai/o3-mini",
    "anthropic/claude-3-5-sonnet-20240620",
    "google/gemini-2.5-pro",
    "google/gemini-2.5-flash",
}

logger = logging.getLogger(__name__)


class LLM:
    """⚡ LLM.

    Developer-first LLM client.
    - Simple for beginners.
    - Flexible for pros.

    Attributes:
        model (str):
            The main model to use for processing requests.

        fallback_models (List[str]):
            List of fallback models to use if the main model fails. Models are attempted in the order they are listed.

        teamspace (Optional[str]):
            The teamspace used for billing. If not provided, it will be resolved using the following methods:
            1. `.lightning/credentials.json` - Attempts to retrieve the teamspace from the local credentials file.
            2. Environment Variables - Checks for `LIGHTNING_*` environment variables.
            3. User Authentication - Redirects the user to the login page if teamspace information is not found.

        max_retries (int):
            The maximum number of retries for API requests in case of failure.


    Usage:
        llm = LLM(model="openai/gpt-4", fallback_models=["mistral/mixtral"], max_retries=6, debug=True)
        llm.chat("What is AI?")  # Stateless
        llm.chat("What is Lightning AI?", conversation="research")  # With history
    """

    _sdkllm_cache: Dict[str, SDKLLM] = {}

    def __init__(
        self,
        model: Optional[str] = None,
        fallback_models: Optional[List[str]] = None,
        teamspace: Optional[str] = None,
        max_retries: int = 3,
        lightning_api_key: Optional[str] = None,
        lightning_user_id: Optional[str] = None,
        enable_async: Optional[bool] = False,
        verbose: int = 0,
        full_response: Optional[bool] = None,
    ) -> None:
        """Initializes the LLM client.

        Args:
            model (Optional[str]): The main model to use. Defaults to openai/gpt-4o.
            fallback_models (Optional[List[str]]): A list of fallback models to use
                                                   if the main model fails. Defaults to None.
            teamspace (Optional[List[str]]): Teamspace used for billing.
            max_retries (int): The maximum number of retries for API requests. Defaults to 3.
            lightning_api_key (Optional[str]): The API key for Lightning AI. Defaults to None.
            lightning_user_id (Optional[str]): The user ID for Lightning AI. Defaults to None.
            enable_async (Optional[bool]): Enable async requests. Defaults to True.
            verbose (int): Verbosity level for logging. Defaults to 0. Must be 0, 1, or 2.
            full_response (bool): Whether the entire response should be returned from the chat
        """
        if (lightning_api_key is None) != (lightning_user_id is None):
            missing_param = "lightning_api_key" if lightning_api_key is None else "lightning_user_id"
            raise ValueError(
                f"Missing required parameter: '{missing_param}'. "
                "Both 'lightning_api_key' and 'lightning_user_id' must be provided together. "
                "Either provide both or none.\n"
                "To find the API key and user ID, go to the Global Settings page in your Lightning account."
            )
        if lightning_api_key is not None:
            os.environ["LIGHTNING_API_KEY"] = lightning_api_key

        if lightning_user_id is not None:
            os.environ["LIGHTNING_USER_ID"] = lightning_user_id

        if verbose not in [0, 1, 2]:
            raise ValueError("Verbose must be 0, 1, or 2.")
        self._verbose = verbose

        if not model:
            model = "openai/gpt-4o"
            warnings.warn(f"No model was provided, defaulting to {model}", UserWarning, stacklevel=2)

        self._model = model
        self._fallback_models = fallback_models or []
        self._teamspace = teamspace
        self._enable_async = enable_async
        self._verbose = verbose
        self.max_retries = max_retries
        self._full_response = full_response

        self._llm: Optional[SDKLLM] = None
        self._fallback_llm: List[SDKLLM] = []
        self._load_event = threading.Event()
        self._load_exception: Optional[BaseException] = None

        threading.Thread(target=self._load_models, daemon=True).start()

    def _authenticate(self, lightning_api_key: Optional[str], lightning_user_id: Optional[str]) -> None:
        if not (lightning_api_key and lightning_user_id):
            return
        auth = login.Auth()
        try:
            auth.authenticate()
            user_api_key = auth.api_key
            user_id = auth.user_id
            os.environ["LIGHTNING_API_KEY"] = user_api_key
            os.environ["LIGHTNING_USER_ID"] = user_id
        except ConnectionError as e:
            raise e

    @property
    def model(self) -> str:
        """Returns the main model name."""
        return self._model

    @property
    def fallback_models(self) -> List[str]:
        """Returns the list of fallback models."""
        return self._fallback_models

    def _load_models(self) -> None:
        """Background loader for SDKLLM and fallback models."""
        try:
            key = f"{self._model}::{self._teamspace}::{self._enable_async}"
            if key not in self._sdkllm_cache:
                self._sdkllm_cache[key] = SDKLLM(
                    name=self._model, teamspace=self._teamspace, enable_async=self._enable_async
                )
            self._llm = self._sdkllm_cache[key]

            for fallback in self._fallback_models:
                fb_key = f"{fallback}::{self._teamspace}::{self._enable_async}"
                if fb_key not in self._sdkllm_cache:
                    self._sdkllm_cache[fb_key] = SDKLLM(
                        name=fallback, teamspace=self._teamspace, enable_async=self._enable_async
                    )
                self._fallback_llm.append(self._sdkllm_cache[fb_key])
            self.models = [self._llm] + self._fallback_llm

            # load cloudy models if not loaded already
            for cur_enable_async in [True, False]:
                for cloudy_model in CLOUDY_MODELS:
                    preload_key = f"{cloudy_model}::{self._teamspace}::{cur_enable_async}"
                    if preload_key in self._sdkllm_cache:
                        continue
                    try:
                        self._sdkllm_cache[preload_key] = SDKLLM(
                            name=cloudy_model, teamspace=self._teamspace, enable_async=cur_enable_async
                        )
                    except Exception:
                        if self._model == cloudy_model:
                            logger.warning(f"Failed to preload cloudy model {cloudy_model}.")

        except BaseException as e:
            self._load_exception = e
        finally:
            self._load_event.set()

    def _wait_for_model(self) -> None:
        """Waits for background model loading to finish."""
        self._load_event.wait()
        if self._load_exception:
            e = self._load_exception
            error_msg = f"failed to load model '{self._model}': {str(e)}"
            if self._verbose == 0:
                raise type(e)(f"failed to load model '{self._model}': {str(e)}")
            raise type(e)(error_msg) from e

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
        """Handles the model call and logs appropriate messages."""
        try:
            if self._verbose == 2:
                print(f"⚡️ Using model: {model.name} (Provider: {model.provider})")

            full_response = (
                (False if self._full_response is None else self._full_response)
                if full_response is None
                else full_response
            )

            return model.chat(
                prompt=prompt,
                system_prompt=system_prompt,
                max_completion_tokens=max_completion_tokens,
                images=images,
                conversation=conversation,
                metadata=metadata,
                stream=stream,
                full_response=full_response,
                **kwargs,
            )
        except requests.exceptions.HTTPError as e:
            print(f"❌ Model '{model.name}' (Provider: {model.provider}) failed.")
            error = handle_http_error(e, model.name)
            if self._verbose:
                print(verbose_http_error_log(error, verbose=self._verbose))
            raise e
        except Exception as e:
            print(
                f"LitAI ran into an error while processing the request to {model.name}. "
                "Please check the error trace for more details."
            )
            if self._verbose:
                print(verbose_sdk_error_log(e, verbose=self._verbose))
            raise e

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
        """Sends a message to the LLM and retrieves a response.

        Args:
            prompt (str): The message to send to the LLM.
            system_prompt (str): The system prompt to set the context. Defaults to assistant's default system prompt.
            model (Optional[str]): The model to override. If provided, this model will be prioritized and used.
            max_tokens (int): The maximum number of tokens for the response. Defaults to 500.
            images (Optional[Union[List[str], str]]): List of local or public image paths to pass to the model.
            conversation (Optional[str]): The conversation ID for maintaining context. Defaults to None.
            metadata (Optional[dict[str, str]]): Dictionary for storing additional information of the request.
            stream (bool): Whether to stream the response. Defaults to False.
            upload_local_images (bool): Whether to upload local images to Teamspace drive.
            internal_conversation (bool): Whether to mark the conversation as internal.
            debug (bool): Enables debug mode for logging detailed information during execution.
            conversation_history (Dict[str, List[Dict[str, Any]]]): A dictionary to store conversation history,
            categorized by conversation ID.
            full_response (bool): Whether the entire response should be returned from the chat.
            **kwargs (Any): Additional keyword arguments

        Returns:
            str: The response from the LLM.
        """
        self._wait_for_model()
        tool_schema = [tool.as_tool() for tool in tools] if tools else None
        if tool_schema:
            tool_context = (
                f"# Available tools:\n{json.dumps(tool_schema, indent=2)}\n\n"
                "Just return the result of the tool call, do not include any other text."
            )
            if system_prompt is None:
                system_prompt = f"Use the following tools to answer the question:\n\n{tool_context}"
            else:
                system_prompt = f"{system_prompt}\n\n{tool_context}"
        if model:
            try:
                model_key = f"{model}::{self._teamspace}::{self._enable_async}"
                if model_key not in self._sdkllm_cache:
                    self._sdkllm_cache[model_key] = SDKLLM(
                        name=model, teamspace=self._teamspace, enable_async=self._enable_async
                    )
                sdk_model = self._sdkllm_cache[model_key]
                return self._model_call(
                    model=sdk_model,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_completion_tokens=max_tokens,
                    images=images,
                    conversation=conversation,
                    metadata=metadata,
                    stream=stream,
                    **kwargs,
                )
            except Exception:
                print(f"💥 Failed to override with model '{model}'")

        # Retry with fallback models
        for model in self.models:
            for attempt in range(self.max_retries):
                try:
                    return self._model_call(
                        model=model,
                        prompt=prompt,
                        system_prompt=system_prompt,
                        max_completion_tokens=max_tokens,
                        images=images,
                        conversation=conversation,
                        metadata=metadata,
                        stream=stream,
                        **kwargs,
                    )
                except Exception:
                    print(f"🔁 Attempt {attempt}/{self.max_retries} failed. Retrying...")

        raise RuntimeError(f"💥 [LLM call failed after {self.max_retries} attempts]")

    @staticmethod
    def call_tool(response: str, tools: Optional[List[LitTool]] = None) -> Optional[str]:
        """Calls a tool with the given response."""
        if tools is None:
            raise ValueError("No tools provided")

        parsed = json.loads(response)
        tool_name = parsed["tool"]
        tool_args = parsed["parameters"]
        for tool in tools:
            if tool.name == tool_name:
                return tool.run(**tool_args)
        return None

    def _dump_debug(
        self,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        exception: Exception,
        response: Optional[requests.Response] = None,
    ) -> None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = "llm_debug_logs"
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, f"llm_error_{timestamp}.log")

        with open(path, "w", encoding="utf-8") as f:
            f.write("❌ LLM CALL DEBUG INFO\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Model: {self.model}\n\n")
            f.write("📬 Headers:\n")
            for k, v in headers.items():
                redacted = v[:10] + "..." if k.lower() == "authorization" else v
                f.write(f"  {k}: {redacted}\n")
            f.write("\n📤 Payload:\n")
            from pprint import pformat

            f.write(pformat(payload))
            f.write("\n\n")
            if response is not None:
                f.write(f"📥 Response status: {response.status_code}\n")
                f.write("📥 Response body:\n")
                f.write(response.text + "\n\n")
            f.write("📛 Exception:\n")
            f.write(f"{repr(exception)}\n")

        print(f"📄 Debug details written to: {path}")

    def reset_conversation(self, name: str) -> None:
        """Resets the conversation history for a given name."""
        self._wait_for_model()
        if self._llm is None:
            raise ValueError("No model loaded")
        self._llm.reset_conversation(name)

    def get_history(self, name: str, raw: bool = False) -> Optional[List[Dict[str, str]]]:
        """Retrieves the conversation history for a given name."""
        self._wait_for_model()
        if self._llm is None:
            raise ValueError("No model loaded")

        history = self._llm.get_history(name)

        if raw:
            return history

        print(f"\n🧠 Conversation: '{name}'")
        for msg in history:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            role_label = "🟦 You" if msg["role"] == "user" else f"🟨 {self.model}"
            print(f"\n[{timestamp}] {role_label}:\n{msg['content']}")
        print("\n--- End of conversation ---\n")
        return None

    def list_conversations(self) -> List[str]:
        """Lists all conversation names."""
        self._wait_for_model()
        if self._llm is None:
            raise ValueError("No model loaded")
        return self._llm.list_conversations()

    def __repr__(self) -> str:
        """Returns a string representation of the LLM instance.

        Returns:
            str: A string representation of the instance.
        """
        return f"<LLM model={self.model} fallback_models={self.fallback_models} max_retries={self.max_retries}"


class LightningLLM(LLM):
    """Alias for the LightningLLM class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the LightningLLM client."""
        super().__init__(*args, **kwargs)
        warnings.warn(
            "The LightningLLM class is deprecated. Use the LLM class instead.",
            DeprecationWarning,
            stacklevel=2,
        )
