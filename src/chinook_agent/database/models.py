from dataclasses import dataclass
from typing import Optional


@dataclass
class Customer:
    customer_id: int
    first_name: str
    last_name: str
    email: str
    phone: Optional[str]
    company: Optional[str]
    city: Optional[str]
    country: Optional[str]
    support_rep_id: Optional[int]