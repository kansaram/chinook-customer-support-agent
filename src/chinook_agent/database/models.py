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

@dataclass
class Invoice:
    invoice_id: int
    customer_id: int
    invoice_date: str
    billing_address: Optional[str]
    billing_city: Optional[str]
    billing_state: Optional[str]
    billing_country: Optional[str]
    billing_postal_code: Optional[str]
    total: float

@dataclass
class InvoiceLine:
    invoice_line_id: int
    invoice_id: int
    track_id: int
    unit_price: float
    quantity: int

@dataclass
class Track:
    track_id: int
    name: str
    album_id: Optional[int]
    media_type_id: int
    genre_id: Optional[int]
    composer: Optional[str]
    milliseconds: int
    bytes: Optional[int]
    unit_price: float

@dataclass
class Employee:
    employee_id: int
    last_name: str
    first_name: str
    title: Optional[str]
    reports_to: Optional[int]
    birth_date: Optional[str]
    hire_date: Optional[str]
    address: Optional[str]
    city: Optional[str]
    state: Optional[str]
    country: Optional[str]
    postal_code: Optional[str]
    phone: Optional[str]
    fax: Optional[str]
    email: Optional[str]


@dataclass
class SupportRep:
    employee_id: int
    first_name: str
    last_name: str
    title: Optional[str]
    email: Optional[str]

@dataclass
class Album:
    album_id: int
    title: str
    artist_id: int

@dataclass
class Artist:
    artist_id: int
    name: str

@dataclass
class Genre:
    genre_id: int
    name: Optional[str]

@dataclass
class MediaType:
    media_type_id: int
    name: Optional[str]

@dataclass
class Playlist: 
    playlist_id: int
    name: str

@dataclass
class PlaylistTrack:
    playlist_id: int
    track_id: int