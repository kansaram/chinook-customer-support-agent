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

def get_invoices_for_customer(customer_id: int) -> list[dict]:
    """Return a list of invoices for the given customer ID."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT InvoiceId, InvoiceDate, BillingAddress, BillingCity, BillingState, BillingCountry, Total "
            "FROM Invoice WHERE CustomerId = ? ORDER BY InvoiceDate DESC",
            (customer_id,),
        ).fetchall()
        return [
            {
                "invoice_id": row["InvoiceId"],
                "invoice_date": row["InvoiceDate"],
                "billing_address": row["BillingAddress"],
                "billing_city": row["BillingCity"],
                "billing_state": row["BillingState"],
                "billing_country": row["BillingCountry"],
                "total": row["Total"],
            }
            for row in rows
        ]
    finally:
        conn.close()

def get_tracks_for_invoices_for_customer(customer_id: int) -> list[dict]:
    """Return a list of tracks for the given invoice ID and customer ID, including track details."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT il.InvoiceLineId, il.TrackId, t.Name AS TrackName, t.Composer, t.Milliseconds, t.Bytes, t.UnitPrice "
            "FROM InvoiceLine il "
            "JOIN Track t ON il.TrackId = t.TrackId "
            "JOIN Invoice i ON il.InvoiceId = i.InvoiceId "
            "WHERE  i.CustomerId = ? ORDER BY t.UnitPrice DESC",
            (customer_id,),
        ).fetchall()
        return [
            {
                "invoice_line_id": row["InvoiceLineId"],
                "track_id": row["TrackId"],
                "track_name": row["TrackName"],
                "composer": row["Composer"],
                "milliseconds": row["Milliseconds"],
                "bytes": row["Bytes"],
                "unit_price": row["UnitPrice"],
            }
            for row in rows
        ]
    finally:
        conn.close()

def get_support_rep_for_customer_by_invoiceId(customer_id: int, invoice_id: int) -> Optional[dict]:
        """Return the support rep details for the given customer ID."""
        conn = get_connection()
        try:
            row = conn.execute(
                "SELECT e.EmployeeId, e.FirstName, e.LastName, e.Title, e.Email "
                "FROM Employee e "
                "JOIN Customer c ON e.EmployeeId = c.SupportRepId "
                "JOIN Invoice i ON c.CustomerId = i.CustomerId "
                "WHERE i.InvoiceId = ? AND c.CustomerId = ?",
                (invoice_id, customer_id),
            ).fetchone()
            if row is None:
                return None
            return {
                "employee_id": row["EmployeeId"],
                "first_name": row["FirstName"],
                "last_name": row["LastName"],
                "title": row["Title"],
                "email": row["Email"],
            }
        finally:
            conn.close()

def get_tracks_for_invoice_for_customer(invoice_id: int, customer_id: int) -> list[int]:
    """Return a list of track IDs for the given invoice ID."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT il.TrackId, t.Name AS TrackName, t.Composer, t.Milliseconds, t.Bytes, t.UnitPrice "
            "FROM InvoiceLine il "
            "JOIN Track t ON il.TrackId = t.TrackId "
            "JOIN Invoice i ON il.InvoiceId = i.InvoiceId "
            "WHERE il.InvoiceId = ? AND i.CustomerId = ?",
            (invoice_id, customer_id),
        ).fetchall()
        return [
            {
                "track_id": row["TrackId"],
                "track_name": row["TrackName"],
                "composer": row["Composer"],
                "milliseconds": row["Milliseconds"],
                "bytes": row["Bytes"],
                "unit_price": row["UnitPrice"],
            }
            for row in rows
        ]
    finally:
        conn.close()
