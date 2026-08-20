import json
import logging
import os
import sys
from typing import Optional, Annotated
from pydantic import BaseModel, Field
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from ..database.repository import get_customer_by_email as _get_customer_by_email
from ..database.repository import get_customer_by_phone as _get_customer_by_phone
from ..database.repository import get_customer_by_id as _get_customer_by_id
from ..database.repository import get_invoices_for_customer as _get_invoices_for_customer
from ..database.repository import get_tracks_for_invoices_for_customer as _get_tracks_for_invoices_for_customer
from ..database.repository import get_tracks_for_invoice_for_customer as _get_tracks_for_invoice_for_customer
from ..database.repository import get_support_rep_for_customer_by_invoiceId as _get_support_rep_for_customer_by_invoiceId
from ..config.logging import get_logger

logger = get_logger(__name__)


class CustomerLookupInput(BaseModel):
    email: Optional[str] = Field(default=None, description="The customer's email address to look up")
    phone: Optional[str] = Field(default=None, description="The customer's phone number to look up")
    customer_id: Optional[int] = Field(default=None, description="The customer's ID to look up")


class InvoiceIdInput(BaseModel):
    invoice_id: int = Field(description="The invoice ID to retrieve details for")


class NoInput(BaseModel):
    """For tools that need nothing beyond what's already in state."""
    pass


def _tool_message(payload: dict, tool_call_id: str, **extra_updates) -> Command:
    """Build a Command whose ToolMessage.content is a JSON string."""
    return Command(update={"messages": [ToolMessage(content=json.dumps(payload), tool_call_id=tool_call_id)], **extra_updates})


def _resolve_customer(email: Optional[str], phone: Optional[str], customer_id: Optional[int]):
    customer = None
    if email:
        customer = _get_customer_by_email(email)
    if customer is None and phone:
        customer = _get_customer_by_phone(phone)
    if customer is None and customer_id:
        customer = _get_customer_by_id(customer_id)
    return customer


@tool("customer_lookup", description="Look up a customer by email, phone, or customer ID.")
def customer_lookup(
    input: CustomerLookupInput,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Look up a customer and persist their customer_id into graph state for later tool calls."""
    if not input.email and not input.phone and not input.customer_id:
        payload = {
            "status": "error",
            "message": "Please provide either an email address, phone number, or customer ID to look up a customer.",
        }
        return _tool_message(payload, tool_call_id)

    customer = _resolve_customer(input.email, input.phone, input.customer_id)

    if customer is None:
        payload = {"status": "not_found", "message": "No customer found with the information provided."}
        return _tool_message(payload, tool_call_id)

    payload = {
        "status": "ok",
        "customer": {
            "customer_id": customer.customer_id,
            "first_name": customer.first_name,
            "last_name": customer.last_name,
            "email": customer.email,
            "phone": customer.phone,
            "company": customer.company,
            "city": customer.city,
            "country": customer.country,
        },
    }

    # This is the key part: update BOTH the message history AND customer_id in state
    return _tool_message(
        payload,
        tool_call_id,
        customer_id=customer.customer_id,
        authenticated=True,
        customer_email=customer.email,
    )


@tool("get_invoice_history", description="Retrieve a customer's invoice history. No arguments needed — uses the customer already identified in this conversation.")
def get_invoice_history(
    input: NoInput,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
) -> Command:
    """Retrieve a customer's invoice history using their customer_id from state."""
    customer_id = state.get("customer_id")
    if not customer_id:
        payload = {"status": "error", "message": "Customer ID not found in state. Please look up the customer first."}
        return _tool_message(payload, tool_call_id)

    invoices = _get_invoices_for_customer(customer_id)
    if not invoices:
        payload = {"status": "not_found", "message": "No invoices found for this customer.", "invoices": []}
        return _tool_message(payload, tool_call_id)

    payload = {
        "status": "ok",
        "invoices": [
            {"invoice_id": invoice["invoice_id"], "date": invoice["invoice_date"], "total": invoice["total"]}
            for invoice in invoices
        ],
    }
    return _tool_message(payload, tool_call_id)


@tool("get_tracks_for_invoices_for_customer", description="Retrieve track details across all of a customer's invoices. No arguments needed — uses the customer already identified in this conversation.")
def get_tracks_for_invoices_for_customer(
    input: NoInput,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
) -> Command:
    """Retrieve track details for all of a customer's invoices using their customer_id from state."""
    customer_id = state.get("customer_id")
    if not customer_id:
        payload = {"status": "error", "message": "Customer ID not found in state. Please look up the customer first."}
        return _tool_message(payload, tool_call_id)

    tracks = _get_tracks_for_invoices_for_customer(customer_id)
    if not tracks:
        payload = {"status": "not_found", "message": "No tracks found for this customer's invoices.", "tracks": []}
        return _tool_message(payload, tool_call_id)

    payload = {
        "status": "ok",
        "tracks": [
            {
                "track_id": track["track_id"],
                "track_name": track["track_name"],
                "composer": track["composer"],
                "milliseconds": track["milliseconds"],
                "bytes": track["bytes"],
                "unit_price": track["unit_price"],
            }
            for track in tracks
        ],
    }
    return _tool_message(payload, tool_call_id)


@tool("get_support_rep_for_customer_by_invoiceId", description="Retrieve the support representative details for a specific invoice, for the customer already identified in this conversation.")
def get_support_rep_for_customer_by_invoiceId(
    input: InvoiceIdInput,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
) -> Command:
    """Retrieve the support representative details for a specific customer and invoice ID using their customer_id from state."""
    customer_id = state.get("customer_id")
    if not customer_id:
        payload = {"status": "error", "message": "Customer ID not found in state. Please look up the customer first."}
        return _tool_message(payload, tool_call_id)

    invoice_id = input.invoice_id
    if not invoice_id:
        payload = {"status": "error", "message": "Invoice ID is required to retrieve the support representative details."}
        return _tool_message(payload, tool_call_id)

    support_rep = _get_support_rep_for_customer_by_invoiceId(customer_id, invoice_id)
    if not support_rep:
        payload = {"status": "not_found", "message": "No support representative found for this customer and invoice ID."}
        return _tool_message(payload, tool_call_id)

    payload = {
        "status": "ok",
        "support_rep": {
            "employee_id": support_rep["employee_id"],
            "first_name": support_rep["first_name"],
            "last_name": support_rep["last_name"],
            "title": support_rep["title"],
            "email": support_rep.get("email"),
        },
    }
    return _tool_message(payload, tool_call_id)


@tool("get_tracks_for_invoice_for_customer", description="Retrieve track details for one specific invoice, for the customer already identified in this conversation.")
def get_tracks_for_invoice_for_customer(
    input: InvoiceIdInput,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
) -> Command:
    """Retrieve track details for a specific invoice and customer using their customer_id from state."""
    customer_id = state.get("customer_id")
    if not customer_id:
        payload = {"status": "error", "message": "Customer ID not found in state. Please look up the customer first."}
        return _tool_message(payload, tool_call_id)

    invoice_id = input.invoice_id
    if not invoice_id:
        payload = {"status": "error", "message": "Invoice ID is required to retrieve the track details."}
        return _tool_message(payload, tool_call_id)

    tracks = _get_tracks_for_invoice_for_customer(invoice_id, customer_id)
    if not tracks:
        payload = {"status": "not_found", "message": "No tracks found for this invoice and customer.", "tracks": []}
        return _tool_message(payload, tool_call_id)

    payload = {
        "status": "ok",
        "invoice_id": invoice_id,
        "tracks": [
            {
                "track_id": track["track_id"],
                "track_name": track["track_name"],
                "composer": track["composer"],
                "milliseconds": track["milliseconds"],
                "bytes": track["bytes"],
                "unit_price": track["unit_price"],
            }
            for track in tracks
        ],
    }
    return _tool_message(payload, tool_call_id)