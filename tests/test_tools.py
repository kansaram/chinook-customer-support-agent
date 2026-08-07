import pytest
from uuid import uuid4
from chinook_agent.tools.invoice_tools import customer_lookup
from chinook_agent.tools.catalog_tools import browse_songs_by_genre_tool
from chinook_agent.tools.catalog_tools import search_song_by_title_fuzzy_tool
from chinook_agent.tools.catalog_tools import search_tracks_by_composer_tool
from chinook_agent.tools.catalog_tools import get_track_details_by_id_tool
from chinook_agent.tools.catalog_tools import suggest_catalog_from_preferences_tool
from chinook_agent.tools.preference_tools import save_preference, apply_preference_update

_TOOL_CALL_ID = "test-call-id"


def _tool_call(name: str, args: dict) -> dict:
    return {"type": "tool_call", "name": name, "id": _TOOL_CALL_ID, "args": args}


def _invoke(payload: dict) -> str:
    """Invoke the tool and extract the ToolMessage content from the returned Command."""
    cmd = customer_lookup.invoke(_tool_call("customer_lookup", {"input": payload}))
    return cmd.update["messages"][0].content


def _invoke_genre(payload: dict) -> str:
    """Invoke the genre tool and extract the ToolMessage content from the returned Command."""
    cmd = browse_songs_by_genre_tool.invoke(_tool_call("browse_songs_by_genre", {"input": payload}))
    return cmd.update["messages"][0].content


def _invoke_song_search(payload: dict) -> str:
    """Invoke the fuzzy song title tool and extract the ToolMessage content from the returned Command."""
    cmd = search_song_by_title_fuzzy_tool.invoke(_tool_call("search_song_by_title_fuzzy", {"input": payload}))
    return cmd.update["messages"][0].content


def _invoke_composer_search(payload: dict) -> str:
    """Invoke the composer search tool and extract the ToolMessage content from the returned Command."""
    cmd = search_tracks_by_composer_tool.invoke(_tool_call("search_tracks_by_composer", {"input": payload}))
    return cmd.update["messages"][0].content


def _invoke_track_details(payload: dict) -> str:
    """Invoke the track details tool and extract the ToolMessage content from the returned Command."""
    cmd = get_track_details_by_id_tool.invoke(_tool_call("get_track_details_by_id", {"input": payload}))
    return cmd.update["messages"][0].content


def _invoke_preference_suggestions(payload: dict, state: dict) -> str:
    """Invoke preference-based catalog suggestion tool with explicit state."""
    cmd = suggest_catalog_from_preferences_tool.invoke(_tool_call("suggest_catalog_from_preferences", {"input": payload, "state": state}))
    return cmd.update["messages"][0].content


def test_customer_lookup_by_email():
    result = _invoke({"email": "luisg@embraer.com.br"})
    assert "luisg@embraer.com.br" in result


def test_customer_lookup_by_phone():
    result = _invoke({"phone": "+55 11 3033-5446"})
    assert "Customer ID" in result


def test_customer_lookup_not_found():
    result = _invoke({"email": "ghost@nowhere.com"})
    assert "No customer found" in result


def test_customer_lookup_case_insensitive_email():
    result = _invoke({"email": "LUISG@EMBRAER.COM.BR"})
    assert "luisg@embraer.com.br" in result


def test_customer_lookup_missing_both():
    result = _invoke({})
    assert "Please provide either an email" in result


def test_browse_songs_by_genre_tool_returns_sample():
    result = _invoke_genre({"genre_name": "Rock", "sample_size": 6, "per_artist_cap": 1})
    assert "Songs in genre Rock" in result
    assert "across" in result
    assert "artists" in result


def test_browse_songs_by_genre_tool_unknown_genre():
    result = _invoke_genre({"genre_name": "no-such-genre-xyz"})
    assert "No genre found" in result


def test_search_song_by_title_fuzzy_tool_returns_results():
    result = _invoke_song_search({"song_title": "Balls to the Wall", "limit": 5})
    assert "Song matches for 'Balls to the Wall'" in result
    assert "by" in result
    assert "score:" in result


def test_search_song_by_title_fuzzy_tool_handles_unknown_song():
    result = _invoke_song_search({"song_title": "no-such-song-xyz", "threshold": 90})
    assert "No songs found matching" in result


def test_search_tracks_by_composer_tool_finds_embedded_composer_text():
    result = _invoke_composer_search({"composer_name": "Dave Grohl", "sample_size": 5})
    assert "Tracks by composer 'Dave Grohl'" in result
    assert "Foo Fighters" in result or "Dave Grohl" in result


def test_search_tracks_by_composer_tool_handles_unknown_composer():
    result = _invoke_composer_search({"composer_name": "no-such-composer-xyz", "sample_size": 5, "threshold": 90})
    assert "No tracks found for composer" in result


def test_get_track_details_by_id_tool_returns_details():
    result = _invoke_track_details({"track_id": 1})
    assert "Track ID: 1" in result
    assert "Name:" in result
    assert "Artist:" in result
    assert "Unit Price:" in result


def test_get_track_details_by_id_tool_handles_unknown_track():
    result = _invoke_track_details({"track_id": 999999})
    assert "No track found with ID" in result


def test_suggest_catalog_from_preferences_tool_uses_state_preferences():
    state = {
        "customer_id": None,
        "customer_email": None,
        "customer_phone": None,
        "preferences": ["I prefer Rock"],
        "pending_preferences": [],
    }

    result = _invoke_preference_suggestions({"sample_size": 3, "per_artist_cap": 1}, state)
    assert "Preference-based suggestions:" in result
    assert "Based on preference" in result


def test_suggest_catalog_from_preferences_tool_loads_saved_preferences_from_memory_repo():
    email = f"pref-test-{uuid4().hex[:8]}@example.com"
    state = {
        "customer_id": None,
        "customer_email": email,
        "customer_phone": None,
        "preferences": [],
        "pending_preferences": [],
    }

    save_preference.invoke(
        _tool_call("save_preference", {"input": {"preference": "I like Jazz"}, "state": state})
    )

    result = _invoke_preference_suggestions({"sample_size": 2, "per_artist_cap": 1}, state)
    assert "Preference-based suggestions:" in result
    assert "Based on preference" in result


def test_save_preference_without_identifier_queues_preference():
    cmd = save_preference.invoke(
        _tool_call(
            "save_preference",
            {
                "input": {"preference": "I like Jazz"},
                "state": {
                    "customer_id": None,
                    "customer_email": None,
                    "customer_phone": None,
                    "preferences": [],
                    "pending_preferences": [],
                },
            },
        )
    )

    message = cmd.update["messages"][0].content
    assert "email, phone, or customer ID" in message
    assert cmd.update["pending_preferences"] == ["I like Jazz"]


def test_apply_preference_update_removes_existing_subject_entries_on_contradiction():
    existing = [
        "You like Pop and catchy music.",
        "Pop",
        "You love Jazz.",
        "Jazz",
    ]

    updated, normalized, is_negative, _ = apply_preference_update(
        existing,
        "You dislike Pop now, unfortunately.",
    )

    assert is_negative is True
    assert normalized == "You dislike Pop now, unfortunately"
    assert "Pop" not in updated
    assert all("pop" not in item.lower() or item.lower() == normalized.lower() for item in updated)
    assert "You love Jazz." in updated
    assert "Jazz" in updated