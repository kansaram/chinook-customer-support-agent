from typing import Optional, Annotated
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from ..database.repository import get_customer_by_email as _get_customer_by_email
from ..database.repository import get_customer_by_phone as _get_customer_by_phone
from ..database.repository import get_customer_by_id as _get_customer_by_id


class CustomerLookupInput(BaseModel):
    email: Optional[str] = Field(default=None, description="The customer's email address to look up")
    phone: Optional[str] = Field(default=None, description="The customer's phone number to look up")
    customer_id: Optional[int] = Field(default=None, description="The customer's ID to look up")


def _resolve_customer(email: Optional[str], phone: Optional[str], customer_id: Optional[int]):
    customer = None
    if email:
        customer = _get_customer_by_email(email)
    if customer is None and phone:
        customer = _get_customer_by_phone(phone)
    if customer is None and customer_id:
        customer = _get_customer_by_id(customer_id)
    return customer


@tool(name="customer_lookup", description="Look up a customer by email, phone, or customer ID.")
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
            "messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)],
        }
    )