"""Test cases for error handling in litai.utils."""

from requests.exceptions import HTTPError
from requests.models import Response

from litai.utils import ModelCallError, handle_http_error, verbose_http_error_log, verbose_sdk_error_log


def mock_http_error(
    status_code=500,
    reason="Internal Server Error",
):
    """Creates a fake HTTPError with a populated response."""
    body = {
        "error": {
            "code": 2,
            "message": (
                "All attempts fail:\n"
                "#1: error, status code: 400, status: 400 Bad Request, message: "
                "This model's maximum context length is 8192 tokens. "
                "However, your messages resulted in 8204 tokens. "
                "Please reduce the length of the messages.\n"
                "#2: error, status code: 400, status: 400 Bad Request, "
                "message: This model's maximum context length is 8192 tokens. "
                "However, your messages resulted in 8204 tokens. "
                "Please reduce the length of the messages.\n"
                "#3: error, status code: 400, status: 400 Bad Request, "
                "message: This model's maximum context length is 8192 tokens. "
                "However, your messages resulted in 8204 tokens. Please reduce the length of the messages."
            ),
            "details": [],
        }
    }
    response = Response()
    response.status_code = status_code
    response.reason = reason
    response._content = str.encode(str(body).replace("'", '"'))
    response.url = "https://fake.api"
    return HTTPError("Simulated error", response=response)


def test_handle_http_error_returns_model_call_error():
    """Test that handle_http_error returns a ModelCallError with correct attributes."""
    http_err = mock_http_error()

    model_err = handle_http_error(http_err, model_name="test-model")

    assert isinstance(model_err, ModelCallError)
    assert model_err.status_code == 500
    assert "Server error" in str(model_err)
    assert model_err.message == "Server error occurred. test-model max context length is 8192. You sent 8204 tokens."
    assert model_err.next_steps == "Reduce the tokens and try again."


def test_verbose_http_error_log_prints_output(capfd):
    """Test that verbose_http_error_log formats the ModelCallError correctly."""
    http_err = mock_http_error()
    model_err = handle_http_error(http_err, model_name="test-model")

    out = verbose_http_error_log(model_err, verbose=1)

    assert (
        out == "[Status code 500]: Server error occurred. test-model max context length is 8192. "
        "You sent 8204 tokens. Reduce the tokens and try again."
    )


def test_verbose_sdk_error_log_levels():
    """Test verbose_sdk_error_log formatting for different verbosity levels."""
    e = AttributeError("object has no attribute '_teamspace_id'")

    assert verbose_sdk_error_log(e, verbose=0) == ""
    assert verbose_sdk_error_log(e, verbose=1) == "[SDK Error] AttributeError"
    assert verbose_sdk_error_log(e, verbose=2) == "[SDK Error] AttributeError: object has no attribute '_teamspace_id'"
