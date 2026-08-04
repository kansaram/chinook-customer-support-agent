import pytest
from chinook_agent.tools.invoice_tools import customer_lookup

_TOOL_CALL_ID = "test-call-id"


def _invoke(payload: dict) -> str:
    """Invoke the tool and extract the ToolMessage content from the returned Command."""
    cmd = customer_lookup.invoke({"input": payload, "tool_call_id": _TOOL_CALL_ID})
    return cmd.update["messages"][0].content


def test_customer_lookup_by_email():
    result = _invoke({"email": "luisg@embraer.com.br"})
    assert "luisg@embraer.com.br" in result


def test_customer_lookup_by_phone():
    result = _invoke({"phone": "+55 11 3033-5446"})
    assert "Customer ID" in result


def test_customer_lookup_not_found():
    result = _invoke({"email": "ghost@nowhere.com"})
    assert "No customer found" in result


def test_customer_lookup_case_insensitive_email():
    result = _invoke({"email": "LUISG@EMBRAER.COM.BR"})
    assert "luisg@embraer.com.br" in result


def test_customer_lookup_missing_both():
    result = _invoke({})
    assert "Please provide either an email" in result