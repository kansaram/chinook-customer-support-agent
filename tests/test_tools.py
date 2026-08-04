import pytest
from tools.invoice_tools import customer_lookup  # rename import path if you move the tool


def test_customer_lookup_by_email():
    result = customer_lookup.invoke({"email": "luisg@embraer.com.br"})
    assert "Luis" in result


def test_customer_lookup_by_phone():
    result = customer_lookup.invoke({"phone": "+55 11 3033-5446"})
    assert "Customer ID" in result


def test_customer_lookup_not_found():
    result = customer_lookup.invoke({"email": "ghost@nowhere.com"})
    assert "No customer found" in result


def test_customer_lookup_case_insensitive_email():
    result = customer_lookup.invoke({"email": "LUISG@EMBRAER.COM.BR"})
    assert "Luis" in result


def test_customer_lookup_missing_both():
    result = customer_lookup.invoke({})
    assert "Please provide either an email" in result