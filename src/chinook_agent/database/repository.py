import sqlite3
import re
from typing import Optional
from .connection import get_connection
from .models import Customer
from rapidfuzz import process, fuzz

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

    
def find_artist_id_by_name(artist_name: str, threshold: int = 60) -> Optional[dict]:
    """Fuzzy-match an artist name to the closest Artist record. Returns None if no good match."""
    conn = get_connection()
    try:
        rows = conn.execute("SELECT ArtistId, Name FROM Artist").fetchall()
    finally:
        conn.close()

    names = {row["Name"]: row["ArtistId"] for row in rows}
    if not names:
        return None

    match = process.extractOne(artist_name, names.keys(), scorer=fuzz.WRatio)
    if match is None or match[1] < threshold:
        return None

    matched_name, score, _ = match

    # Guard against WRatio's partial-ratio fallback matching a short
    # candidate against a substring of a much longer query (e.g. "Kiss"
    # matching inside "kishor kumar"). Require a high score when the
    # two strings differ a lot in length.
    len_ratio = min(len(matched_name), len(artist_name)) / max(len(matched_name), len(artist_name))
    if len_ratio < 0.5 and score < 90:
        return None

    return {"artist_id": names[matched_name], "matched_name": matched_name, "score": score}


def find_genre_id_by_name(genre_name: str, threshold: int = 60) -> Optional[dict]:
    """Fuzzy-match a genre name to the closest Genre record. Returns None if no good match."""
    conn = get_connection()
    try:
        rows = conn.execute("SELECT GenreId, Name FROM Genre WHERE Name IS NOT NULL").fetchall()
    finally:
        conn.close()

    names = {row["Name"]: row["GenreId"] for row in rows}
    if not names:
        return None

    match = process.extractOne(genre_name, names.keys(), scorer=fuzz.WRatio)
    if match is None or match[1] < threshold:
        return None

    matched_name, score, _ = match

    # Same guard as find_artist_id_by_name: prevents a short real genre
    # name (e.g. "Pop") from matching a much longer, unrelated query
    # (e.g. "Zzzqplorp-fusion") off a coincidental substring overlap.
    len_ratio = min(len(matched_name), len(genre_name)) / max(len(matched_name), len(genre_name))
    if len_ratio < 0.5 and score < 90:
        return None

    return {"genre_id": names[matched_name], "matched_name": matched_name, "score": score}

def get_albums_for_artist(artist_id: int) -> list[dict]:
    """Return a list of albums for the given artist ID."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT a.Title, t.Name AS ArtistName "
            "FROM Artist t "
            "JOIN Album a ON a.ArtistId = t.ArtistId "
            "WHERE t.ArtistId = ?",
            (artist_id,),
        ).fetchall()
        return [{"artist_name": row["ArtistName"], "title": row["Title"]} for row in rows]
    finally:
        conn.close()

def search_tracks_by_artist(artist_id: int, sample_size: int = 10) -> dict:
    """Return total track count and a sample of tracks for the given artist ID."""
    conn = get_connection()
    try:
        total_row = conn.execute(
            "SELECT COUNT(*) as total "
            "FROM Track t "
            "JOIN Album a ON t.AlbumId = a.AlbumId "
            "JOIN Artist ar ON a.ArtistId = ar.ArtistId "
            "WHERE ar.ArtistId = ?",
            (artist_id,),
        ).fetchone()
        total = total_row["total"]

        sample_rows = conn.execute(
            "SELECT t.Name AS TrackName, a.Title AS AlbumTitle "
            "FROM Track t "
            "JOIN Album a ON t.AlbumId = a.AlbumId "
            "JOIN Artist ar ON a.ArtistId = ar.ArtistId "
            "WHERE ar.ArtistId = ? "
            "LIMIT ?",
            (artist_id, sample_size),
        ).fetchall()

        sample = [{"track_name": row["TrackName"], "album_title": row["AlbumTitle"]} for row in sample_rows]

        return {"total": total, "sample": sample}
    finally:
        conn.close()


def browse_songs_by_genre(genre_name: str, sample_size: int = 12, per_artist_cap: int = 2) -> Optional[dict]:
    """Return genre totals and a representative sample spread across different artists."""
    if sample_size <= 0:
        return None

    if per_artist_cap <= 0:
        per_artist_cap = 1

    match = find_genre_id_by_name(genre_name)
    if match is None:
        return None

    conn = get_connection()
    try:
        totals_row = conn.execute(
            "SELECT COUNT(*) AS total_tracks, COUNT(DISTINCT ar.ArtistId) AS total_artists "
            "FROM Track t "
            "JOIN Album a ON t.AlbumId = a.AlbumId "
            "JOIN Artist ar ON a.ArtistId = ar.ArtistId "
            "WHERE t.GenreId = ?",
            (match["genre_id"],),
        ).fetchone()

        total_tracks = totals_row["total_tracks"]
        total_artists = totals_row["total_artists"]

        if total_tracks == 0:
            return {
                "genre_name": match["matched_name"],
                "total_tracks": 0,
                "total_artists": 0,
                "sample": [],
            }

        sample_rows = conn.execute(
            "WITH ranked AS ("
            "    SELECT "
            "        t.Name AS track_name, "
            "        a.Title AS album_title, "
            "        ar.Name AS artist_name, "
            "        ROW_NUMBER() OVER (PARTITION BY ar.ArtistId ORDER BY t.Name) AS artist_rank "
            "    FROM Track t "
            "    JOIN Album a ON t.AlbumId = a.AlbumId "
            "    JOIN Artist ar ON a.ArtistId = ar.ArtistId "
            "    WHERE t.GenreId = ?"
            "), interleaved AS ("
            "    SELECT "
            "        track_name, album_title, artist_name, artist_rank, "
            "        ROW_NUMBER() OVER (ORDER BY artist_rank, artist_name, track_name) AS global_rank "
            "    FROM ranked "
            "    WHERE artist_rank <= ?"
            ") "
            "SELECT track_name, album_title, artist_name "
            "FROM interleaved "
            "WHERE global_rank <= ? "
            "ORDER BY global_rank",
            (match["genre_id"], per_artist_cap, sample_size),
        ).fetchall()

        sample = [
            {
                "track_name": row["track_name"],
                "album_title": row["album_title"],
                "artist_name": row["artist_name"],
            }
            for row in sample_rows
        ]

        return {
            "genre_name": match["matched_name"],
            "total_tracks": total_tracks,
            "total_artists": total_artists,
            "sample": sample,
        }
    finally:
        conn.close()


def search_song_by_title_fuzzy(song_title: str, limit: int = 10, threshold: int = 60) -> list[dict]:
    """Fuzzy-search tracks by title and return detailed matches with artist and album."""
    if not song_title.strip() or limit <= 0:
        return []

    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT t.TrackId, t.Name AS TrackName, a.Title AS AlbumTitle, ar.Name AS ArtistName "
            "FROM Track t "
            "JOIN Album a ON t.AlbumId = a.AlbumId "
            "JOIN Artist ar ON a.ArtistId = ar.ArtistId"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return []

    name_to_rows: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        name_to_rows.setdefault(row["TrackName"], []).append(row)

    matches = process.extract(
        song_title,
        name_to_rows.keys(),
        scorer=fuzz.WRatio,
        limit=limit,
        score_cutoff=threshold,
    )

    results: list[dict] = []
    for matched_name, score, _ in matches:
        for row in name_to_rows[matched_name]:
            results.append(
                {
                    "track_id": row["TrackId"],
                    "track_name": row["TrackName"],
                    "album_title": row["AlbumTitle"],
                    "artist_name": row["ArtistName"],
                    "score": score,
                }
            )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:limit]


def search_tracks_by_composer(composer_name: str, sample_size: int = 10, threshold: int = 60) -> dict:
    """Search tracks by composer text and return a total count plus a representative sample."""
    if not composer_name.strip() or sample_size <= 0:
        return {"total": 0, "sample": []}

    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT t.TrackId, t.Name AS TrackName, a.Title AS AlbumTitle, t.Composer "
            "FROM Track t "
            "JOIN Album a ON t.AlbumId = a.AlbumId "
            "WHERE t.Composer IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return {"total": 0, "sample": []}

    composer_lower = composer_name.strip().lower()
    exact_matches = [
        row for row in rows
        if composer_lower in str(row["Composer"]).lower()
    ]

    if exact_matches:
        sample = [
            {
                "track_id": row["TrackId"],
                "track_name": row["TrackName"],
                "album_title": row["AlbumTitle"],
                "composer": row["Composer"],
                "score": 100,
            }
            for row in exact_matches[:sample_size]
        ]
        return {"total": len(exact_matches), "sample": sample}

    composer_to_rows: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        composer_to_rows.setdefault(row["Composer"], []).append(row)

    matches = process.extract(
        composer_name,
        composer_to_rows.keys(),
        scorer=fuzz.WRatio,
        limit=sample_size,
        score_cutoff=threshold,
    )

    sample: list[dict] = []
    for matched_composer, score, _ in matches:
        for row in composer_to_rows[matched_composer]:
            sample.append(
                {
                    "track_id": row["TrackId"],
                    "track_name": row["TrackName"],
                    "album_title": row["AlbumTitle"],
                    "composer": row["Composer"],
                    "score": score,
                }
            )

    sample.sort(key=lambda item: item["score"], reverse=True)
    return {"total": len(sample), "sample": sample[:sample_size]}


def get_track_details_by_id(track_id: int) -> Optional[dict]:
    """Return complete details for a specific track by its ID."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT "
            "    t.TrackId, "
            "    t.Name AS TrackName, "
            "    a.AlbumId, "
            "    a.Title AS AlbumTitle, "
            "    ar.ArtistId, "
            "    ar.Name AS ArtistName, "
            "    g.GenreId, "
            "    g.Name AS GenreName, "
            "    mt.MediaTypeId, "
            "    mt.Name AS MediaTypeName, "
            "    t.Composer, "
            "    t.Milliseconds, "
            "    t.Bytes, "
            "    t.UnitPrice "
            "FROM Track t "
            "LEFT JOIN Album a ON t.AlbumId = a.AlbumId "
            "LEFT JOIN Artist ar ON a.ArtistId = ar.ArtistId "
            "LEFT JOIN Genre g ON t.GenreId = g.GenreId "
            "LEFT JOIN MediaType mt ON t.MediaTypeId = mt.MediaTypeId "
            "WHERE t.TrackId = ?",
            (track_id,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return None

    return {
        "track_id": row["TrackId"],
        "track_name": row["TrackName"],
        "album_id": row["AlbumId"],
        "album_title": row["AlbumTitle"],
        "artist_id": row["ArtistId"],
        "artist_name": row["ArtistName"],
        "genre_id": row["GenreId"],
        "genre_name": row["GenreName"],
        "media_type_id": row["MediaTypeId"],
        "media_type_name": row["MediaTypeName"],
        "composer": row["Composer"],
        "milliseconds": row["Milliseconds"],
        "bytes": row["Bytes"],
        "unit_price": row["UnitPrice"],
    }
