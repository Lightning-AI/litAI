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
"""LLM config class."""

import os
from typing import Dict


class LLMConfig:
    """Configuration class for a Language Learning Model (LLM).

    Attributes:
        name (str): The name of the LLM.
        url (str): The URL endpoint for the LLM API.
        api_key (str): The API key for authenticating requests to the LLM.
        model_name (str): The name of the specific model to use (optional).
    """

    def __init__(self, name: str, url: str, api_key_env_var: str, model_name: str = "") -> None:
        """Initializes an LLMConfig instance.

        Args:
            name (str): The name of the LLM.
            url (str): The URL endpoint for the LLM API.
            api_key_env_var (str): The environment variable name that stores the API key.
            model_name (str, optional): The name of the specific model to use. Defaults to an empty string.
        """
        self.name: str = name
        self.url: str = url
        self.api_key: str = os.getenv(api_key_env_var, "")
        self.model_name: str = model_name

    def __repr__(self) -> str:
        """Returns a string representation of the LLMConfig instance.

        Returns:
            str: A string representation of the instance.
        """
        return f"<LLMConfig name={self.name} url={self.url} model={self.model_name}>"


Models: Dict[str, LLMConfig] = {
    "lightning/llama-4": LLMConfig(
        name="lightning/llama-4",
        url="https://8000-dep-01jr8da15depc6k5xveb5jp9aw-d.cloudspaces.litng.ai/v1/chat/completions",
        api_key_env_var="LLAMA_API_KEY",
        model_name="llama-4",
    )
}
