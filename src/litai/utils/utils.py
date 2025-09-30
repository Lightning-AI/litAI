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
"""Utility functions for handling model call errors and logging."""

import re
import traceback
from typing import Optional

import requests
from lightning_sdk.llm import LLM as SDKLLM
from requests import HTTPError


class ModelCallError(Exception):
    """LitAI exception for model call failures."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        message: Optional[str] = None,
        status_code: Optional[int] = None,
        next_steps: Optional[str] = None,
        original_exception: Optional[Exception] = None,
    ):
        """Initializes a ModelCallError with detailed context."""
        super().__init__(message)
        self.status_code = status_code
        self.original_exception = original_exception
        self.reason = message
        self.next_steps = next_steps
        self.model_name = model_name
        self.message = message


class ErrorLogger:
    """Centralized error logging with verbose control."""

    def __init__(self, verbose: int):
        """Initializes ErrorLogger with verbosity level."""
        self.verbose = verbose

    def log_http_error(self, error: ModelCallError) -> None:
        """Log HTTP error with appropriate verbosity."""
        if not self.verbose:
            return

        parts = []

        if error.status_code:
            parts.append(f"[Status code {error.status_code}]:")
        else:
            parts.append("[Status code unknown]:")

        if error.reason:
            parts.append(error.reason)

        if error.next_steps:
            parts.append(error.next_steps)

        if self.verbose >= 2 and error.original_exception:
            parts.append(f"\n[Original exception: {error.original_exception}]")

        print(" ".join(parts))

    def log_sdk_error(self, e: Exception) -> None:
        """Log SDK error with appropriate verbosity."""
        parts = []
        error_type = type(e).__name__

        # Basic error type at verbose >= 1
        if self.verbose >= 1:
            parts.append(f"[Error type] {error_type}")

        # Detailed info at verbose >= 2
        if self.verbose >= 2:
            parts.append(f"[Error message] {str(e)}")
            parts.append(f"[Full traceback]\n{traceback.format_exc()}")

        if self.verbose == 0:
            parts.append(
                "LitAI ran into an error while processing the request to the model. "
                "Please check the error trace for more details."
            )

        print("\n".join(parts))


def extract_token_counts(response_body: str) -> Optional[tuple[int, int]]:
    """Extracts token counts from the error response body."""
    match = re.search(r"maximum context length is (\d+) tokens.*?your messages resulted in (\d+) tokens", response_body)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def handle_http_error(error: HTTPError, model_name: str) -> ModelCallError:
    """Handles HTTPError and returns a ModelCallError with detailed context."""
    status_code = error.response.status_code if error.response is not None else None

    next_steps = "Retry the request or check your network connection."
    if status_code is not None and status_code == 499:
        message = "Client closed connection before response."
    elif status_code is not None and 500 <= status_code < 600:
        message = "Server error occurred. "
        if error.response.text:
            token_counts = extract_token_counts(error.response.text)
            if token_counts:
                message += f"{model_name} max context length is {token_counts[0]}. You sent {token_counts[1]} tokens."
                next_steps = "Reduce the tokens and try again."
            else:
                message += "Please try again later."
    else:
        message = "Unexpected HTTP error occurred."

    return ModelCallError(
        model_name=model_name, message=message, status_code=status_code, next_steps=next_steps, original_exception=error
    )


def verbose_http_error_log(e: ModelCallError, verbose: int) -> str:
    """Formats the ModelCallError for verbose logging."""
    out = []
    if e.status_code:
        out.append(f"[Status code {e.status_code}]:")
    else:
        out.append("[Status code unknown]:")

    if e.reason:
        out.append(e.reason)

    if e.next_steps:
        out.append(e.next_steps)

    if verbose > 1 and e.original_exception:
        out.append(f"\n[Original exception: {e.original_exception}]")
    return " ".join(out)


def verbose_sdk_error_log(e: Exception, verbose: int) -> str:
    """Formats a generic SDK error for verbose logging."""
    import traceback

    error_type = type(e).__name__
    message = str(e)
    full_traceback = traceback.format_exc()

    out = []

    if verbose >= 1:
        out.append(f"[Error type] {error_type}")
    if verbose >= 2:
        out.append(f"[Error message] {message}")
        out.append(f"[Full traceback]\n{full_traceback}")

    return "\n".join(out)


def handle_model_error(e: Exception, model: SDKLLM, attempt: int, max_retries: int, verbose: int) -> None:
    """Centralized error handling and logging for model calls."""
    logger = ErrorLogger(verbose)
    # Log error for every attempt
    if isinstance(e, requests.exceptions.HTTPError):
        error = handle_http_error(e, model.name)
        logger.log_http_error(error)
    else:
        logger.log_sdk_error(e)

    if attempt < max_retries - 1:
        print(f"🔁 Attempt {attempt + 1}/{max_retries} failed. Retrying...")
    else:
        print("-" * 50)
        print(f"❌ All {max_retries} attempts failed for model {model.name}")
        print("-" * 50)


def handle_empty_response(model: SDKLLM, attempt: int, max_retries: int) -> None:
    """Handles empty responses from model calls."""
    if attempt < max_retries - 1:
        print(f"🔁 Received empty response. Attempt {attempt + 1}/{max_retries} failed. Retrying...")
    else:
        print("-" * 50)
        print(f"❌ All {max_retries} attempts received empty responses for model {model.name}.")
        print("-" * 50)
