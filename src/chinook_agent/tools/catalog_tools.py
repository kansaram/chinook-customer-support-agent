import json
import re
from typing import Annotated
from pydantic import BaseModel, Field
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from ..database.repository import (
    find_artist_id_by_name,
    get_albums_for_artist,
    search_tracks_by_artist,
    browse_songs_by_genre,
    search_song_by_title_fuzzy,
    search_tracks_by_composer,
    get_track_details_by_id,
)
from ..database.memory_repository import load_preferences_list
from .preference_tools import get_preferences, save_preference, resolve_identifier


class AlbumsByArtistInput(BaseModel):
    artist_name: str = Field(description="The artist's name, even if potentially misspelled")


def _tool_message(payload: dict, tool_call_id: str, **extra_updates) -> Command:
    """Build a Command whose ToolMessage.content is a JSON string."""
    return Command(update={"messages": [ToolMessage(content=json.dumps(payload), tool_call_id=tool_call_id)], **extra_updates})


@tool("get_albums_for_artist", description="Find albums by an artist name, using fuzzy matching to tolerate typos.")
def get_albums_for_artist_tool(
    input: AlbumsByArtistInput,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    match = find_artist_id_by_name(input.artist_name)

    if match is None:
        payload = {"status": "not_found", "message": f"No artist found matching '{input.artist_name}'."}
        return _tool_message(payload, tool_call_id)

    albums = get_albums_for_artist(match["artist_id"])

    if not albums:
        payload = {
            "status": "not_found",
            "message": f"Found artist '{match['matched_name']}' but no albums are listed for them.",
            "artist_name": match["matched_name"],
        }
        return _tool_message(payload, tool_call_id)

    matched_from = input.artist_name if match["matched_name"].lower() != input.artist_name.lower() else None
    payload = {
        "status": "ok",
        "artist_name": match["matched_name"],
        "matched_from": matched_from,
        "albums": [{"title": a["title"]} for a in albums],
    }
    return _tool_message(payload, tool_call_id)


class TracksByArtistInput(BaseModel):
    artist_name: str = Field(description="The artist's name, even if potentially misspelled")


class BrowseSongsByGenreInput(BaseModel):
    genre_name: str = Field(description="Genre name, even if potentially misspelled")
    sample_size: int = Field(default=12, ge=1, le=50, description="How many songs to sample")
    per_artist_cap: int = Field(default=2, ge=1, le=5, description="Max songs per artist in the sample")


class SongTitleSearchInput(BaseModel):
    song_title: str = Field(description="Song title to search for, even if potentially misspelled")
    limit: int = Field(default=10, ge=1, le=25, description="Maximum number of matching songs to return")
    threshold: int = Field(default=60, ge=0, le=100, description="Minimum fuzzy match score")


class TrackIdInput(BaseModel):
    track_id: int = Field(description="Track ID to retrieve complete details for")


class ComposerSearchInput(BaseModel):
    composer_name: str = Field(description="Composer name to search for, even if it appears inside a longer composer field")
    sample_size: int = Field(default=10, ge=1, le=25, description="Maximum number of tracks to return")
    threshold: int = Field(default=60, ge=0, le=100, description="Minimum fuzzy match score for fallback matching")


class PreferenceSuggestionInput(BaseModel):
    sample_size: int = Field(default=6, ge=1, le=20, description="How many songs to sample per suggestion")
    per_artist_cap: int = Field(default=1, ge=1, le=3, description="Max songs per artist for genre-based suggestions")
    max_preference_matches: int = Field(default=3, ge=1, le=10, description="Maximum preference entries to use")


def _preference_candidates(preference: str) -> list[str]:
    fragments = [fragment.strip() for fragment in re.split(r"\s*;\s*|\n+", preference) if fragment.strip()]
    candidates: list[str] = []
    prefixes = [
        "i like ",
        "you like ",
        "i love ",
        "you love ",
        "i prefer ",
        "you prefer ",
        "my favorite genre is ",
        "my favourite genre is ",
        "favorite genre is ",
        "favourite genre is ",
        "prefer ",
        "like ",
        "love ",
        "dislike ",
        "you dislike ",
    ]
    for raw in fragments or [preference.strip()]:
        lowered = raw.lower()
        for prefix in prefixes:
            if lowered.startswith(prefix):
                cleaned = raw[len(prefix):].strip(" .,!?")
                if cleaned:
                    candidates.extend([cleaned, raw])
                break
        else:
            candidates.append(raw)

    deduped: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        key = item.lower()
        if key not in seen:
            deduped.append(item)
            seen.add(key)
    return deduped


@tool("search_tracks_by_artist", description="Find tracks/songs by an artist name, with fuzzy matching, total count, and a sample of results.")
def search_tracks_by_artist_tool(
    input: TracksByArtistInput,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    match = find_artist_id_by_name(input.artist_name)

    if match is None:
        payload = {"status": "not_found", "message": f"No artist found matching '{input.artist_name}'."}
        return _tool_message(payload, tool_call_id)

    result = search_tracks_by_artist(match["artist_id"])

    if result["total"] == 0:
        payload = {
            "status": "not_found",
            "message": f"Found artist '{match['matched_name']}' but no tracks are listed for them.",
            "artist_name": match["matched_name"],
        }
        return _tool_message(payload, tool_call_id)

    sample = result["sample"]
    payload = {
        "status": "ok",
        "artist_name": match["matched_name"],
        "total": result["total"],
        "showing": len(sample),
        "more_exist": result["total"] > len(sample),
        "sample": [{"track_name": t["track_name"], "album_title": t["album_title"]} for t in sample],
    }
    return _tool_message(payload, tool_call_id)


@tool(
    "browse_songs_by_genre",
    description="Browse songs by genre with a representative sample across different artists.",
)
def browse_songs_by_genre_tool(
    input: BrowseSongsByGenreInput,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    result = browse_songs_by_genre(
        genre_name=input.genre_name,
        sample_size=input.sample_size,
        per_artist_cap=input.per_artist_cap,
    )

    if result is None:
        payload = {"status": "not_found", "message": f"No genre found matching '{input.genre_name}'."}
        return _tool_message(payload, tool_call_id)

    if result["total_tracks"] == 0:
        payload = {
            "status": "not_found",
            "message": f"Genre '{result['genre_name']}' has no tracks listed.",
            "genre_name": result["genre_name"],
        }
        return _tool_message(payload, tool_call_id)

    sample = result["sample"]
    payload = {
        "status": "ok",
        "genre_name": result["genre_name"],
        "total_tracks": result["total_tracks"],
        "total_artists": result["total_artists"],
        "showing": len(sample),
        "more_exist": result["total_tracks"] > len(sample),
        "sample": [
            {"track_name": t["track_name"], "artist_name": t["artist_name"], "album_title": t["album_title"]}
            for t in sample
        ],
    }
    return _tool_message(payload, tool_call_id)


@tool(
    "search_song_by_title_fuzzy",
    description="Search for a specific song by title using fuzzy matching.",
)
def search_song_by_title_fuzzy_tool(
    input: SongTitleSearchInput,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    result = search_song_by_title_fuzzy(
        song_title=input.song_title,
        limit=input.limit,
        threshold=input.threshold,
    )

    if result["total"] == 0:
        payload = {"status": "not_found", "message": f"No songs found matching '{input.song_title}'."}
        return _tool_message(payload, tool_call_id)

    sample = result["sample"]
    payload = {
        "status": "ok",
        "query": input.song_title,
        "total": result["total"],
        "showing": len(sample),
        "more_exist": result["total"] > len(sample),
        "sample": [
            {
                "track_name": row["track_name"],
                "artist_name": row["artist_name"],
                "album_title": row["album_title"],
                "score": row["score"],
            }
            for row in sample
        ],
    }
    return _tool_message(payload, tool_call_id)


@tool(
    "search_tracks_by_composer",
    description="Search tracks by composer text, including composer names embedded in track metadata.",
)
def search_tracks_by_composer_tool(
    input: ComposerSearchInput,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    result = search_tracks_by_composer(
        composer_name=input.composer_name,
        sample_size=input.sample_size,
        threshold=input.threshold,
    )

    if result["total"] == 0:
        payload = {"status": "not_found", "message": f"No tracks found for composer '{input.composer_name}'."}
        return _tool_message(payload, tool_call_id)

    sample = result["sample"]
    payload = {
        "status": "ok",
        "composer_query": input.composer_name,
        "total": result["total"],
        "showing": len(sample),
        "more_exist": result["total"] > len(sample),
        "sample": [
            {
                "track_name": row["track_name"],
                "composer": row["composer"],
                "album_title": row["album_title"],
                "score": row["score"],
            }
            for row in sample
        ],
    }
    return _tool_message(payload, tool_call_id)


@tool(
    "get_track_details_by_id",
    description="Get complete details for a specific track by its ID.",
)
def get_track_details_by_id_tool(
    input: TrackIdInput,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    details = get_track_details_by_id(input.track_id)
    if details is None:
        payload = {"status": "not_found", "message": f"No track found with ID {input.track_id}."}
        return _tool_message(payload, tool_call_id)

    payload = {
        "status": "ok",
        "track": {
            "track_id": details["track_id"],
            "track_name": details["track_name"],
            "artist_name": details.get("artist_name"),
            "album_title": details.get("album_title"),
            "genre_name": details.get("genre_name"),
            "media_type_name": details.get("media_type_name"),
            "composer": details.get("composer"),
            "milliseconds": details["milliseconds"],
            "bytes": details.get("bytes"),
            "unit_price": details["unit_price"],
        },
    }
    return _tool_message(payload, tool_call_id)


@tool(
    "suggest_catalog_from_preferences",
    description="Suggest tracks using saved customer preferences (genre/artist).",
)
def suggest_catalog_from_preferences_tool(
    input: PreferenceSuggestionInput,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
) -> Command:
    preferences = list(state.get("preferences", []))
    if not preferences:
        identifier = resolve_identifier(
            customer_id=state.get("customer_id"),
            email=state.get("customer_email"),
            phone=state.get("customer_phone"),
        )
        if identifier:
            preferences = load_preferences_list(identifier)

    if not preferences:
        payload = {
            "status": "no_preferences",
            "message": "No saved preferences are available yet. Ask the customer to share a preference, then use save_preference first.",
        }
        return _tool_message(payload, tool_call_id)

    suggestions: list[dict] = []
    used = 0

    for preference in preferences:
        if used >= input.max_preference_matches:
            break

        for candidate in _preference_candidates(preference):
            genre_result = browse_songs_by_genre(
                genre_name=candidate,
                sample_size=input.sample_size,
                per_artist_cap=input.per_artist_cap,
            )
            if genre_result and genre_result["sample"]:
                tracks = [
                    {"track_name": row["track_name"], "artist_name": row["artist_name"]}
                    for row in genre_result["sample"][: input.sample_size]
                ]
                suggestions.append(
                    {
                        "preference": preference,
                        "matched_genre": genre_result["genre_name"],
                        "matched_artist": None,
                        "tracks": tracks,
                    }
                )
                used += 1
                break

            artist_match = find_artist_id_by_name(candidate)
            if artist_match:
                tracks_result = search_tracks_by_artist(artist_match["artist_id"], sample_size=input.sample_size)
                if tracks_result["sample"]:
                    tracks = [
                        {"track_name": row["track_name"], "album_title": row["album_title"]}
                        for row in tracks_result["sample"][: input.sample_size]
                    ]
                    suggestions.append(
                        {
                            "preference": preference,
                            "matched_genre": None,
                            "matched_artist": artist_match["matched_name"],
                            "tracks": tracks,
                        }
                    )
                    used += 1
                    break

    if not suggestions:
        payload = {
            "status": "no_match",
            "message": "Saved preferences were found, but none could be mapped to a known genre or artist in the catalog.",
        }
        return _tool_message(payload, tool_call_id)

    payload = {"status": "ok", "suggestions": suggestions}
    return _tool_message(payload, tool_call_id)