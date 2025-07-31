"""LitAI main tests."""

import json
import os
import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litai import LLM, tool


def test_initialization_with_config_file(monkeypatch):
    """Test LitAI config."""
    mock_llm_instance = MagicMock()
    monkeypatch.setattr("litai.llm.SDKLLM", mock_llm_instance)
    LLM(model="openai/gpt-4", lightning_api_key="my-key", lightning_user_id="my-user-id")
    assert os.getenv("LIGHTNING_API_KEY") == "my-key"
    assert os.getenv("LIGHTNING_USER_ID") == "my-user-id"


@patch("litai.llm.SDKLLM")
def test_invalid_model(mock_llm_class):
    """Test invalid model name."""
    dummy_model_name = "dummy-model"
    mock_llm_class.side_effect = ValueError(
        f"Failed to load model '{dummy_model_name}': Model '{dummy_model_name}' not found. "
    )
    llm = LLM(model=dummy_model_name)
    with pytest.raises(ValueError, match="not found"):
        llm._wait_for_model()


def test_default_model(monkeypatch):
    """Test default model name."""
    mock_llm_instance = MagicMock()
    monkeypatch.setattr("litai.llm.SDKLLM", mock_llm_instance)
    warning_message = "No model was provided, defaulting to openai/gpt-4o"
    with pytest.warns(UserWarning, match=re.escape(warning_message)):
        llm = LLM()
        assert len(llm.fallback_models) == 0
        assert llm.model == "openai/gpt-4o"


@patch("litai.llm.SDKLLM")
def test_cloudy_models_preload(mock_sdkllm):
    """Test that CLOUDY_MODELS are preloaded during LLM initialization."""
    cloudy_models = {
        "openai/gpt-4o",
        "openai/gpt-4",
        "openai/o3-mini",
        "anthropic/claude-3-5-sonnet-20240620",
        "google/gemini-2.5-pro",
        "google/gemini-2.5-flash",
    }
    from litai.llm import LLM as LLMCLIENT

    LLMCLIENT._sdkllm_cache.clear()
    llm = LLM()
    llm._wait_for_model()

    expected_calls = len(cloudy_models) * 2  # for both async and sync
    assert mock_sdkllm.call_count == expected_calls, (
        f"Expected {expected_calls} calls to SDKLLM, but got {mock_sdkllm.call_count}"
    )

    enable_async_param = {call.kwargs["enable_async"] for call in mock_sdkllm.call_args_list}
    assert set(enable_async_param) == {True, False}


@patch("litai.llm.SDKLLM")
def test_llm_context_length(mock_llm_class):
    """Test LigtningLLM context length."""
    from litai.llm import LLM as LLMCLIENT

    LLMCLIENT._sdkllm_cache.clear()
    mock_llm_instance = MagicMock()
    mock_llm_instance.context_length.return_value = 8000

    mock_llm_class.return_value = mock_llm_instance

    llm = LLM(model="openai/gpt-4")

    assert llm.context_length() == 8000


@patch("litai.llm.SDKLLM")
def test_llm_chat(mock_llm_class):
    """Test LigtningLLM chat."""
    from litai.llm import LLM as LLMCLIENT

    LLMCLIENT._sdkllm_cache.clear()
    mock_llm_instance = MagicMock()
    mock_llm_instance.chat.return_value = "Hello! I am a helpful assistant."

    mock_llm_class.return_value = mock_llm_instance

    llm = LLM(model="openai/gpt-4")

    response = llm.chat(
        "Hello, who are you?",
        system_prompt="You are a helpful assistant.",
        metadata={"user_api": "123456"},
        my_kwarg="test-kwarg",
    )

    assert isinstance(response, str)
    assert "helpful" in response.lower()
    mock_llm_instance.chat.assert_called_once_with(
        prompt="Hello, who are you?",
        system_prompt="You are a helpful assistant.",
        max_completion_tokens=500,
        images=None,
        conversation=None,
        metadata={"user_api": "123456"},
        stream=False,
        full_response=False,
        my_kwarg="test-kwarg",
    )
    test_kwargs = mock_llm_instance.chat.call_args.kwargs
    assert test_kwargs.get("my_kwarg") == "test-kwarg"

    llm.reset_conversation("test")
    mock_llm_instance.reset_conversation.assert_called_once()


def test_model_override(monkeypatch):
    """Test override model logic when main model fails."""
    mock_llm = MagicMock()
    mock_llm.name = "default-model"
    mock_llm.enable_async = False
    mock_fallback_model = MagicMock()
    mock_fallback_model.name = "fallback-model"
    mock_fallback_model.enable_async = False

    mock_override = MagicMock()
    mock_override.name = "override-model"
    mock_override.chat.return_value = "Override response"
    mock_override.enable_async = False

    def mock_llm_constructor(name, teamspace="default-teamspace", **kwargs):
        if name == "default-model":
            return mock_llm
        if name == "fallback-model":
            return mock_fallback_model
        if name == "override-model":
            return mock_override
        raise ValueError(f"Unknown model: {name}")

    monkeypatch.setattr("litai.llm.SDKLLM", mock_llm_constructor)

    llm = LLM(
        model="default-model",
        fallback_models=["fallback-model"],
        max_retries=3,
        full_response=True,
    )

    llm.chat(prompt="Hello", model="override-model")

    assert mock_override.chat.call_count == 1
    assert mock_fallback_model.chat.call_count == 0
    assert mock_llm.chat.call_count == 0

    mock_override.chat.assert_called_once_with(
        prompt="Hello",
        system_prompt=None,
        max_completion_tokens=500,
        images=None,
        conversation=None,
        metadata=None,
        stream=False,
        full_response=True,
    )


def test_fallback_models(monkeypatch):
    """Test fallback model logic when main model fails."""
    from litai.llm import LLM as LLMCLIENT

    LLMCLIENT._sdkllm_cache.clear()
    mock_main_model = MagicMock()
    mock_main_model.name = "main-model"
    mock_fallback_model = MagicMock()
    mock_fallback_model.name = "fallback-model"

    mock_main_model.chat.side_effect = Exception("Primary model error")
    mock_fallback_model.chat.side_effect = [
        Exception("Fallback error 1"),
        Exception("Fallback error 2"),
        "Fallback response",
    ]

    def mock_llm_constructor(name, teamspace="default-teamspace", **kwargs):
        if name == "main-model":
            return mock_main_model
        if name == "fallback-model":
            return mock_fallback_model
        raise ValueError(f"Unknown model: {name}")

    monkeypatch.setattr("litai.llm.SDKLLM", mock_llm_constructor)

    llm = LLM(
        model="main-model",
        fallback_models=["fallback-model"],
        max_retries=3,
    )

    llm.chat(prompt="Hello")

    assert mock_main_model.chat.call_count == 3
    assert mock_fallback_model.chat.call_count == 3

    mock_fallback_model.chat.assert_called_with(
        prompt="Hello",
        system_prompt=None,
        max_completion_tokens=500,
        images=None,
        conversation=None,
        metadata=None,
        stream=False,
        full_response=False,
    )


@pytest.mark.asyncio
async def test_llm_async_chat(monkeypatch):
    """Test async requests."""
    mock_sdkllm = MagicMock()
    mock_sdkllm.name = "mock-model"
    mock_sdkllm.chat = AsyncMock(return_value="Hello, async world!")

    monkeypatch.setattr("litai.llm.SDKLLM", lambda *args, **kwargs: mock_sdkllm)

    llm = LLM(model="mock-model", enable_async=True)
    result = await llm.chat("Hi there", conversation="async-test")
    assert result == "Hello, async world!"
    mock_sdkllm.chat.assert_called_once()


def test_get_history(monkeypatch, capsys):
    """Test get history."""
    mock_sdkllm = MagicMock()
    mock_sdkllm.name = "mock-model"
    mock_sdkllm.get_history = MagicMock(
        return_value=[
            {"role": "user", "content": "Hello, world!", "model": "mock-model"},
            {"role": "assistant", "content": "I am a mock model!", "model": "mock-model"},
        ]
    )

    monkeypatch.setattr("litai.llm.SDKLLM", lambda *args, **kwargs: mock_sdkllm)

    llm = LLM(model="mock-model")

    # Test default behavior (prints to stdout)
    result = llm.get_history("async-test")
    assert result is None  # get_history returns None when raw=False

    # Capture the printed output
    captured = capsys.readouterr()
    assert "🧠 Conversation: 'async-test'" in captured.out
    assert "🟦 You" in captured.out
    assert "🟨 mock-model" in captured.out
    assert "Hello, world!" in captured.out
    assert "I am a mock model!" in captured.out
    assert "--- End of conversation ---" in captured.out

    # Test raw=True behavior (returns data instead of printing)
    result = llm.get_history("async-test", raw=True)
    assert result == [
        {"role": "user", "content": "Hello, world!", "model": "mock-model"},
        {"role": "assistant", "content": "I am a mock model!", "model": "mock-model"},
    ]


def test_authenticate_method(monkeypatch):
    # Mock the login.Auth class
    mock_auth = MagicMock()
    mock_auth.api_key = "test-api-key"
    mock_auth.user_id = "test-user-id"

    def mock_auth_constructor():
        return mock_auth

    monkeypatch.setattr("litai.llm.login.Auth", mock_auth_constructor)

    # Test case 1: Both api_key and user_id provided
    LLM(model="openai/gpt-4", lightning_api_key="my-key", lightning_user_id="my-user-id")

    # Verify that the authentication was not called
    mock_auth.authenticate.assert_not_called()

    # Verify that environment variables were set
    assert os.getenv("LIGHTNING_API_KEY") == "my-key"
    assert os.getenv("LIGHTNING_USER_ID") == "my-user-id"

    # Test case 2: Neither api_key nor user_id provided
    mock_auth.reset_mock()
    os.environ.pop("LIGHTNING_API_KEY", None)
    os.environ.pop("LIGHTNING_USER_ID", None)

    LLM(model="openai/gpt-4")

    # Verify that authentication was called
    mock_auth.authenticate.assert_called_once()


@patch("litai.llm.SDKLLM")
def test_llm_if_method(mock_sdkllm_class):
    """Test the LLM if_ method."""
    from litai.llm import LLM as LLMCLIENT

    LLMCLIENT._sdkllm_cache.clear()

    # Instantiate LLM first
    llm = LLM(model="openai/gpt-4")

    # Get the actual mock instance used by llm
    mock_sdkllm_instance = mock_sdkllm_class.return_value

    # Test case where the condition is true
    mock_sdkllm_instance.chat.side_effect = ["yes", "no", " Yes "]  # Use side_effect for multiple calls
    assert llm.if_("this review is great", "is this a positive review?") is True

    # Test case where the condition is false
    assert llm.if_("this review is terrible", "is this a positive review?") is False

    # Test case with different capitalization/spacing
    assert llm.if_("the product is amazing", "is it a positive response?") is True


@patch("litai.llm.SDKLLM")
def test_llm_classify_method(mock_sdkllm_class):
    """Test the LLM classify method."""
    from litai.llm import LLM as LLMCLIENT

    LLMCLIENT._sdkllm_cache.clear()

    llm = LLM(model="openai/gpt-4")

    # Get the actual mock instance used by llm
    mock_sdkllm_instance = mock_sdkllm_class.return_value

    # Use side_effect to return different values for sequential calls
    mock_sdkllm_instance.chat.side_effect = ["positive", "negative", "neutral"]

    # Test simple classification
    result = llm.classify("this movie was great!", ["positive", "negative"])
    assert result == "positive"

    # Test another classification
    result = llm.classify("this movie was awful.", ["positive", "negative"])
    assert result == "negative"

    # Test with multiple classes
    result = llm.classify("it was okay.", ["positive", "negative", "neutral"])
    assert result == "neutral"


def test_llm_call_tool():
    """Test the LLM call_tool method."""
    response = json.dumps({"tool": "test_tool", "parameters": {"message": "How do I get a refund?"}})

    @tool
    def test_tool(message: str) -> str:
        return f"Tool received: {message}"

    llm = LLM(model="openai/gpt-4")

    with patch("litai.llm.SDKLLM.chat", return_value=response):
        result = llm.call_tool(response, tools=[test_tool])
    assert result == "Tool received: How do I get a refund?"


@patch("builtins.open", new_callable=MagicMock)
@patch("os.makedirs")
def test_dump_debug(mock_makedirs, mock_open):
    """Test the LLM dump_debug method."""
    mock_file = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_file

    logs = []
    mock_file.write = lambda x: logs.append(x)

    llm = LLM(model="openai/gpt-4")
    llm._dump_debug(
        payload={"prompt": "Hello, world!"},
        headers={"Authorization": "Bearer test-token"},
        exception=Exception("Test exception"),
        response=MagicMock(status_code=200, text="Test response"),
    )

    mock_makedirs.assert_called_once_with("llm_debug_logs", exist_ok=True)

    mock_open.assert_called_once()
    call_args = mock_open.call_args
    assert call_args[0][1] == "w", "write mode"
    assert call_args[1]["encoding"] == "utf-8"

    assert len(logs) > 0, "content must be written"
    written_content = "".join(logs)
    assert "❌ LLM CALL DEBUG INFO" in written_content
    assert "Model: openai/gpt-4" in written_content
    assert "📬 Headers:" in written_content
    assert "Authorization: Bearer tes..." in written_content, "Authorization header should be redacted"
    assert "📤 Payload:" in written_content
    assert "Hello, world!" in written_content
    assert "📥 Response status: 200" in written_content
    assert "Test response" in written_content
    assert "📛 Exception:" in written_content
    assert "Test exception" in written_content
