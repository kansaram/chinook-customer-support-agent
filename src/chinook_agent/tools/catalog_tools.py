from typing import Annotated
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from ..database.repository import (
    find_artist_id_by_name,
    get_albums_for_artist,
    search_tracks_by_artist,
    browse_songs_by_genre,
    search_song_by_title_fuzzy,
    get_track_details_by_id,
)
from ..database.memory_repository import load_preferences_list
from .preference_tools import get_preferences, save_preference, resolve_identifier


class AlbumsByArtistInput(BaseModel):
    artist_name: str = Field(description="The artist's name, even if potentially misspelled")


@tool("get_albums_for_artist", description="Find albums by an artist name, using fuzzy matching to tolerate typos.")
def get_albums_for_artist_tool(
    input: AlbumsByArtistInput,
    tool_call_id: Annotated[str, "tool_call_id"],
) -> Command:
    match = find_artist_id_by_name(input.artist_name)

    if match is None:
        message = f"No artist found matching '{input.artist_name}'."
        return Command(update={"messages": [ToolMessage(content=message, tool_call_id=tool_call_id)]})

    albums = get_albums_for_artist(match["artist_id"])

    if not albums:
        message = f"Found artist '{match['matched_name']}' but no albums are listed for them."
        return Command(update={"messages": [ToolMessage(content=message, tool_call_id=tool_call_id)]})

    note = f" (matched from '{input.artist_name}')" if match["matched_name"].lower() != input.artist_name.lower() else ""
    album_list = "\n".join(f"- {a['title']}" for a in albums)
    summary = f"Albums by {match['matched_name']}{note}:\n{album_list}"

    return Command(update={"messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)]})

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


class PreferenceSuggestionInput(BaseModel):
    sample_size: int = Field(default=6, ge=1, le=20, description="How many songs to sample per suggestion")
    per_artist_cap: int = Field(default=1, ge=1, le=3, description="Max songs per artist for genre-based suggestions")
    max_preference_matches: int = Field(default=3, ge=1, le=10, description="Maximum preference entries to use")


def _preference_candidates(preference: str) -> list[str]:
    raw = preference.strip()
    lowered = raw.lower()
    prefixes = [
        "i like ",
        "i love ",
        "i prefer ",
        "my favorite genre is ",
        "my favourite genre is ",
        "favorite genre is ",
        "favourite genre is ",
        "prefer ",
        "like ",
        "love ",
    ]
    for prefix in prefixes:
        if lowered.startswith(prefix):
            cleaned = raw[len(prefix):].strip(" .,!?")
            if cleaned:
                return [cleaned, raw]
    return [raw]


@tool("search_tracks_by_artist", description="Find tracks/songs by an artist name, with fuzzy matching, total count, and a sample of results.")
def search_tracks_by_artist_tool(
    input: TracksByArtistInput,
    tool_call_id: Annotated[str, "tool_call_id"],
) -> Command:
    match = find_artist_id_by_name(input.artist_name)

    if match is None:
        message = f"No artist found matching '{input.artist_name}'."
        return Command(update={"messages": [ToolMessage(content=message, tool_call_id=tool_call_id)]})

    result = search_tracks_by_artist(match["artist_id"])

    if result["total"] == 0:
        message = f"Found artist '{match['matched_name']}' but no tracks are listed for them."
        return Command(update={"messages": [ToolMessage(content=message, tool_call_id=tool_call_id)]})

    sample_lines = "\n".join(f"- {t['track_name']} (from {t['album_title']})" for t in result["sample"])
    truncation_note = (
        f"\n\n(Showing {len(result['sample'])} of {result['total']} total tracks — more results exist.)"
        if result["total"] > len(result["sample"])
        else ""
    )
    summary = f"Tracks by {match['matched_name']}:\n{sample_lines}{truncation_note}"

    return Command(update={"messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)]})


@tool(
    "browse_songs_by_genre",
    description="Browse songs by genre with a representative sample across different artists.",
)
def browse_songs_by_genre_tool(
    input: BrowseSongsByGenreInput,
    tool_call_id: Annotated[str, "tool_call_id"],
) -> Command:
    result = browse_songs_by_genre(
        genre_name=input.genre_name,
        sample_size=input.sample_size,
        per_artist_cap=input.per_artist_cap,
    )

    if result is None:
        message = f"No genre found matching '{input.genre_name}'."
        return Command(update={"messages": [ToolMessage(content=message, tool_call_id=tool_call_id)]})

    if result["total_tracks"] == 0:
        message = f"Genre '{result['genre_name']}' has no tracks listed."
        return Command(update={"messages": [ToolMessage(content=message, tool_call_id=tool_call_id)]})

    sample_lines = "\n".join(
        f"- {t['track_name']} by {t['artist_name']} (from {t['album_title']})" for t in result["sample"]
    )
    truncation_note = (
        f"\n\n(Showing {len(result['sample'])} of {result['total_tracks']} total tracks "
        f"across {result['total_artists']} artists — more results exist.)"
        if result["total_tracks"] > len(result["sample"])
        else f"\n\n(Showing all {len(result['sample'])} tracks across {result['total_artists']} artists.)"
    )
    summary = f"Songs in genre {result['genre_name']}:\n{sample_lines}{truncation_note}"

    return Command(update={"messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)]})


@tool(
    "search_song_by_title_fuzzy",
    description="Search for a specific song by title using fuzzy matching.",
)
def search_song_by_title_fuzzy_tool(
    input: SongTitleSearchInput,
    tool_call_id: Annotated[str, "tool_call_id"],
) -> Command:
    matches = search_song_by_title_fuzzy(
        song_title=input.song_title,
        limit=input.limit,
        threshold=input.threshold,
    )

    if not matches:
        message = f"No songs found matching '{input.song_title}'."
        return Command(update={"messages": [ToolMessage(content=message, tool_call_id=tool_call_id)]})

    lines = "\n".join(
        f"- {row['track_name']} by {row['artist_name']} (from {row['album_title']}, score: {row['score']:.1f})"
        for row in matches
    )
    summary = (
        f"Song matches for '{input.song_title}':\n"
        f"{lines}\n\n"
        f"(Showing {len(matches)} result{'s' if len(matches) != 1 else ''}.)"
    )
    return Command(update={"messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)]})


@tool(
    "get_track_details_by_id",
    description="Get complete details for a specific track by its ID.",
)
def get_track_details_by_id_tool(
    input: TrackIdInput,
    tool_call_id: Annotated[str, "tool_call_id"],
) -> Command:
    details = get_track_details_by_id(input.track_id)
    if details is None:
        message = f"No track found with ID {input.track_id}."
        return Command(update={"messages": [ToolMessage(content=message, tool_call_id=tool_call_id)]})

    summary = (
        f"Track ID: {details['track_id']}\n"
        f"Name: {details['track_name']}\n"
        f"Artist: {details.get('artist_name') or 'N/A'}\n"
        f"Album: {details.get('album_title') or 'N/A'}\n"
        f"Genre: {details.get('genre_name') or 'N/A'}\n"
        f"Media Type: {details.get('media_type_name') or 'N/A'}\n"
        f"Composer: {details.get('composer') or 'N/A'}\n"
        f"Duration (ms): {details['milliseconds']}\n"
        f"Size (bytes): {details.get('bytes') if details.get('bytes') is not None else 'N/A'}\n"
        f"Unit Price: ${details['unit_price']:.2f}"
    )
    return Command(update={"messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)]})


@tool(
    "suggest_catalog_from_preferences",
    description="Suggest tracks using saved customer preferences (genre/artist).",
)
def suggest_catalog_from_preferences_tool(
    input: PreferenceSuggestionInput,
    tool_call_id: Annotated[str, "tool_call_id"],
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
        message = (
            "No saved preferences are available yet. "
            "Ask the customer to share a preference, then use save_preference first."
        )
        return Command(update={"messages": [ToolMessage(content=message, tool_call_id=tool_call_id)]})

    suggestion_blocks: list[str] = []
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
                lines = "\n".join(
                    f"  - {row['track_name']} by {row['artist_name']}"
                    for row in genre_result["sample"][: input.sample_size]
                )
                suggestion_blocks.append(
                    f"Based on preference '{preference}' (matched genre: {genre_result['genre_name']}):\n{lines}"
                )
                used += 1
                break

            artist_match = find_artist_id_by_name(candidate)
            if artist_match:
                tracks = search_tracks_by_artist(artist_match["artist_id"], sample_size=input.sample_size)
                if tracks["sample"]:
                    lines = "\n".join(
                        f"  - {row['track_name']} (from {row['album_title']})"
                        for row in tracks["sample"][: input.sample_size]
                    )
                    suggestion_blocks.append(
                        f"Based on preference '{preference}' (matched artist: {artist_match['matched_name']}):\n{lines}"
                    )
                    used += 1
                    break

    if not suggestion_blocks:
        message = (
            "Saved preferences were found, but none could be mapped to a known genre or artist in the catalog."
        )
        return Command(update={"messages": [ToolMessage(content=message, tool_call_id=tool_call_id)]})

    summary = "Preference-based suggestions:\n" + "\n\n".join(suggestion_blocks)
    return Command(update={"messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)]})