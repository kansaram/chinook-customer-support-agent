import sqlite3
import re
from typing import Optional
from .connection import get_connection
from .models import Customer

def _normalize_phone(phone: Optional[str]) -> str:
    """Strip everything except digits, so +1 (555) 123-4567 and 15551234567 compare equal."""
    if phone is None:
        return ""
    digits = re.sub(r"\D", "", phone)
    return digits[-10:] if len(digits) >= 10 else digits

def get_customer_by_email(email: str) -> Optional[Customer]:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM Customer WHERE LOWER(Email) = LOWER(?)", (email,)).fetchone()
        if row is None:
            return None
        return Customer(
            customer_id=row["CustomerId"],
            first_name=row["FirstName"],
            last_name=row["LastName"],
            email=row["Email"],
            phone=row["Phone"],
            company=row["Company"],
            city=row["City"],
            country=row["Country"],
            support_rep_id=row["SupportRepId"],
        )
    finally:
        conn.close()

def get_customer_by_phone(phone: str) -> Optional[Customer]:
    conn = get_connection()
    conn.create_function("NORMALIZE_PHONE", 1, _normalize_phone)
    try:
        normalized_input = _normalize_phone(phone)
        row = conn.execute(
            "SELECT CustomerId, FirstName, LastName, Email, Phone, Company, City, Country, SupportRepId "
            "FROM Customer WHERE NORMALIZE_PHONE(Phone) = ?",
            (normalized_input,),
        ).fetchone()
        if row is None:
            return None
        return Customer(
            customer_id=row["CustomerId"],
            first_name=row["FirstName"],
            last_name=row["LastName"],
            email=row["Email"],
            phone=row["Phone"],
            company=row["Company"],
            city=row["City"],
            country=row["Country"],
            support_rep_id=row["SupportRepId"],
        )
    finally:
        conn.close()

def get_customer_by_id(customer_id: int) -> Optional[Customer]:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM Customer WHERE CustomerId = ?", (customer_id,)).fetchone()
        if row is None:
            return None
        return Customer(
            customer_id=row["CustomerId"],
            first_name=row["FirstName"],
            last_name=row["LastName"],
            email=row["Email"],
            phone=row["Phone"],
            company=row["Company"],
            city=row["City"],
            country=row["Country"],
            support_rep_id=row["SupportRepId"],
        )
    finally:
        conn.close()