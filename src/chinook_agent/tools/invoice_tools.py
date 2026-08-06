import logging
import os
import sys
from typing import Optional, Annotated
from pydantic import BaseModel, Field
from langchain_core.tools import tool
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
    tool_call_id: Annotated[str, "tool_call_id"],
) -> Command:
    """Look up a customer and persist their customer_id into graph state for later tool calls."""
    if not input.email and not input.phone and not input.customer_id:
        message = "Please provide either an email address, phone number, or customer ID to look up a customer."
        return Command(update={"messages": [ToolMessage(content=message, tool_call_id=tool_call_id)]})

    customer = _resolve_customer(input.email, input.phone, input.customer_id)

    if customer is None:
        message = "No customer found with the information provided."
        return Command(update={"messages": [ToolMessage(content=message, tool_call_id=tool_call_id)]})

    summary = (
        f"Customer ID: {customer.customer_id}\n"
        f"Name: {customer.first_name} {customer.last_name}\n"
        f"Email: {customer.email}\n"
        f"Phone: {customer.phone}\n"
        f"Company: {customer.company}\n"
        f"City: {customer.city}\n"
        f"Country: {customer.country}"
    )

    # This is the key part: update BOTH the message history AND customer_id in state
    return Command(
        update={
            "customer_id": customer.customer_id,
            "authenticated": True,
            "customer_email": customer.email,
            "messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)],
        }
    )


@tool("get_invoice_history", description="Retrieve a customer's invoice history by customer ID.")
def get_invoice_history(
    input: NoInput,
    tool_call_id: Annotated[str, "tool_call_id"],
    state: InjectedState,
) -> Command:
    """Retrieve a customer's invoice history using their customer_id from state."""
    customer_id = state.get("customer_id")
    if not customer_id:
        message = "Customer ID not found in state. Please look up the customer first."
        return Command(update={"messages": [ToolMessage(content=message, tool_call_id=tool_call_id)]})

    invoices = _get_invoices_for_customer(customer_id)
    if not invoices:
        message = "No invoices found for this customer."
        return Command(update={"messages": [ToolMessage(content=message, tool_call_id=tool_call_id)]})

    invoice_summary = "\n".join(
        f"Invoice ID: {invoice['invoice_id']}, Date: {invoice['invoice_date']}, Total: ${invoice['total']:.2f}"
        for invoice in invoices
    )

    return Command(update={"messages": [ToolMessage(content=invoice_summary, tool_call_id=tool_call_id)]})


@tool("get_tracks_for_invoices_for_customer", description="Retrieve track details across all of a customer's invoices.")
def get_tracks_for_invoices_for_customer(
    input: NoInput,
    tool_call_id: Annotated[str, "tool_call_id"],
    state: InjectedState,
) -> Command:
    """Retrieve track details for all of a customer's invoices using their customer_id from state."""
    customer_id = state.get("customer_id")
    if not customer_id:
        message = "Customer ID not found in state. Please look up the customer first."
        return Command(update={"messages": [ToolMessage(content=message, tool_call_id=tool_call_id)]})

    tracks = _get_tracks_for_invoices_for_customer(customer_id)
    if not tracks:
        message = "No tracks found for this customer's invoices."
        return Command(update={"messages": [ToolMessage(content=message, tool_call_id=tool_call_id)]})

    track_summary = "\n".join(
        f"Track ID: {track['track_id']}, Name: {track['track_name']}, Composer: {track['composer']}, "
        f"Duration (ms): {track['milliseconds']}, Size (bytes): {track['bytes']}, Price: ${track['unit_price']:.2f}"
        for track in tracks
    )

    return Command(update={"messages": [ToolMessage(content=track_summary, tool_call_id=tool_call_id)]})


@tool("get_support_rep_for_customer_by_invoiceId", description="Retrieve the support representative details for a specific customer and invoice ID.")
def get_support_rep_for_customer_by_invoiceId(
    input: InvoiceIdInput,
    tool_call_id: Annotated[str, "tool_call_id"],
    state: InjectedState,
) -> Command:
    """Retrieve the support representative details for a specific customer and invoice ID using their customer_id from state."""
    customer_id = state.get("customer_id")
    if not customer_id:
        message = "Customer ID not found in state. Please look up the customer first."
        return Command(update={"messages": [ToolMessage(content=message, tool_call_id=tool_call_id)]})

    invoice_id = input.invoice_id
    if not invoice_id:
        message = "Invoice ID is required to retrieve the support representative details."
        return Command(update={"messages": [ToolMessage(content=message, tool_call_id=tool_call_id)]})

    support_rep = _get_support_rep_for_customer_by_invoiceId(customer_id, invoice_id)
    if not support_rep:
        message = "No support representative found for this customer and invoice ID."
        return Command(update={"messages": [ToolMessage(content=message, tool_call_id=tool_call_id)]})

    support_rep_summary = (
        f"Support Rep ID: {support_rep['employee_id']}\n"
        f"Name: {support_rep['first_name']} {support_rep['last_name']}\n"
        f"Title: {support_rep['title']}\n"
        f"Email: {support_rep.get('email', 'N/A')}"
    )

    return Command(update={"messages": [ToolMessage(content=support_rep_summary, tool_call_id=tool_call_id)]})


@tool("get_tracks_for_invoice_for_customer", description="Retrieve track details for a specific invoice and customer.")
def get_tracks_for_invoice_for_customer(
    input: InvoiceIdInput,
    tool_call_id: Annotated[str, "tool_call_id"],
    state: InjectedState,
) -> Command:
    """Retrieve track details for a specific invoice and customer using their customer_id from state."""
    customer_id = state.get("customer_id")
    if not customer_id:
        message = "Customer ID not found in state. Please look up the customer first."
        return Command(update={"messages": [ToolMessage(content=message, tool_call_id=tool_call_id)]})

    invoice_id = input.invoice_id
    if not invoice_id:
        message = "Invoice ID is required to retrieve the track details."
        return Command(update={"messages": [ToolMessage(content=message, tool_call_id=tool_call_id)]})

    tracks = _get_tracks_for_invoice_for_customer(invoice_id, customer_id)
    if not tracks:
        message = "No tracks found for this invoice and customer."
        return Command(update={"messages": [ToolMessage(content=message, tool_call_id=tool_call_id)]})

    track_summary = "\n".join(
        f"Track ID: {track['track_id']}, Name: {track['track_name']}, Composer: {track['composer']}, "
        f"Duration (ms): {track['milliseconds']}, Size (bytes): {track['bytes']}, Price: ${track['unit_price']:.2f}"
        for track in tracks
    )

    return Command(update={"messages": [ToolMessage(content=track_summary, tool_call_id=tool_call_id)]})