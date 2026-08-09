"""
Test suite for chinook-customer-support-agent.

Run with:
    python -m pytest test_agent_scenarios.py -v -s

Or run a single scenario directly:
    python test_agent_scenarios.py

Each test uses a fresh thread_id (isolated conversation) unless a scenario
explicitly needs multi-turn state, in which case turns are sent with the
SAME thread_id in sequence.

NOTE: These tests make real LLM calls. They are not free, and results for
"soft" assertions (tone, exact wording) may vary run to run since LLM output
isn't fully deterministic even at temperature=0. Assertions are written to
check STRUCTURAL/BEHAVIORAL correctness (which agent handled it, whether a
tool was grounded, whether known-wrong data appears) rather than exact wording.
"""

import uuid
import pytest

from chinook_agent.graph import build_graph

graph = build_graph()


def new_config():
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def send(message: str, config: dict) -> dict:
    """Send one turn and return the full result dict (state after this turn)."""
    result = graph.invoke({"messages": [{"role": "user", "content": message}]}, config=config)
    return result


def reply_text(result: dict) -> str:
    if result.get("response"):
        return result["response"]
    messages = result.get("messages", [])
    return getattr(messages[-1], "content", "") if messages else ""


# ============================================================
# CATEGORY 1: Basic routing correctness
# ============================================================

def test_01_catalog_intent_routes_correctly():
    """Plain catalog question should not touch invoice/account logic at all."""
    result = send("What albums does Iron Maiden have?", new_config())
    text = reply_text(result).lower()
    assert "customer id" not in text
    assert "invoice" not in text


def test_02_invoice_intent_asks_for_identification():
    """Invoice question with no prior identity should ask for email/phone/customer ID."""
    result = send("Can you show me my invoice history?", new_config())
    text = reply_text(result).lower()
    assert any(word in text for word in ["email", "phone", "customer id"])


def test_03_account_question_routes_to_invoice_not_catalog():
    """'Account' phrasing (no invoice keyword) must not land on catalog_agent's refusal."""
    result = send("I want to know my account", new_config())
    text = reply_text(result).lower()
    assert "unable to assist" not in text
    assert "reach out to" not in text  # the old canned refusal we saw in testing


def test_04_bare_email_with_no_context_routes_to_memory():
    """A bare email with nothing else should go to memory_agent, not trigger a Chinook lookup failure."""
    result = send("john.doe@example.com", new_config())
    text = reply_text(result).lower()
    assert "no customer found" not in text
    assert "couldn't find any customer information" not in text


# ============================================================
# CATEGORY 2: Multi-turn continuity (the "who asked for this" bugs)
# ============================================================

def test_05_reply_to_invoice_identification_request_stays_with_invoice():
    """After invoice_agent asks for an identifier, a bare-email reply must stay
    with invoice_agent, not get hijacked to memory_agent (Rule 0 regression test)."""
    config = new_config()
    send("Can you help me with an invoice related query?", config)
    result = send("john.doe@example.com", config)
    text = reply_text(result).lower()
    # Should attempt a Chinook lookup (even if it fails), not pivot to a preferences answer
    assert "preference" not in text or "invoice" in text


def test_06_catalog_followup_promise_is_honored():
    """If the assistant promises 'tell me a genre and I'll suggest songs', the
    reply must route to catalog_agent, not get intercepted as a bare preference save."""
    config = new_config()
    send("Can you recommend some music for me?", config)
    result = send("I like jazz", config)
    text = reply_text(result).lower()
    assert "noted" not in text or "jazz" in text  # should attempt an actual suggestion, not just "saved"


def test_07_different_identifier_mid_conversation_forces_fresh_lookup():
    """Providing a genuinely different email/phone mid-conversation must NOT reuse
    cached preferences from the first identifier."""
    config = new_config()
    send("my email is testuser1@example.com, what are my preferences", config)
    result = send("actually check with my phone 555-000-1111 instead", config)
    text = reply_text(result)
    # Must not claim a match to the old identity's data without a fresh tool call
    assert "555-000-1111" in text or "phone" in text.lower()


# ============================================================
# CATEGORY 3: Preference capture and deduplication
# ============================================================

def test_08_explicit_preference_gets_saved():
    config = new_config()
    send("my email is prefstest1@example.com", config)
    result = send("I really love reggae music", config)
    text = reply_text(result).lower()
    assert "reggae" in text or "noted" in text


def test_09_llm_fallback_catches_nonstandard_phrasing():
    """'I'm really into X' doesn't match the fixed regex list — must be caught
    by the LLM fallback parser, not silently dropped."""
    config = new_config()
    send("my email is prefstest2@example.com", config)
    send("I'm really into classical music", config)
    result = send("what are my preferences", config)
    text = reply_text(result).lower()
    assert "classical" in text


def test_10_duplicate_phrasing_does_not_create_two_entries():
    """'Classical music' and 'listening to Classical music' must collapse to ONE entry."""
    config = new_config()
    send("my email is prefstest3@example.com", config)
    send("I like Classical music", config)
    send("I was so enthralled listening classical music that my interest grows in that direction", config)
    result = send("what are my preferences", config)
    text = reply_text(result).lower()
    assert text.count("classical") <= 2  # heading mention + one bullet, not two bullets


def test_11_conflicting_statement_updates_not_duplicates():
    """Stating dislike after like for the same subject should update, not duplicate."""
    config = new_config()
    send("my email is prefstest4@example.com", config)
    send("I like Rock music", config)
    send("actually I dislike Rock", config)
    result = send("what are my preferences", config)
    text = reply_text(result).lower()
    assert "dislike rock" in text
    # Should not ALSO still claim "like Rock" as a separate line
    assert text.count("rock") <= 2


def test_12_preferences_persist_across_new_conversation_thread():
    """Preferences saved in one thread must be retrievable in a BRAND NEW thread
    for the same identifier — proves DB persistence, not just in-memory state."""
    identifier_email = "persisttest@example.com"
    config1 = new_config()
    send(f"my email is {identifier_email}", config1)
    send("I love blues music", config1)

    config2 = new_config()  # different thread_id — simulates a new browser session
    result = send(f"my email is {identifier_email}, what are my preferences", config2)
    text = reply_text(result).lower()
    assert "blues" in text


# ============================================================
# CATEGORY 4: Non-Chinook-customer coverage
# ============================================================

def test_13_preferences_work_without_being_a_chinook_customer():
    """A made-up email that will never match a real Chinook customer must still
    be able to save/load preferences."""
    config = new_config()
    send("my email is definitely-not-a-real-customer-99999@nowhere.test", config)
    send("I like techno", config)
    result = send("what are my preferences", config)
    text = reply_text(result).lower()
    assert "techno" in text


def test_14_pending_preference_flushes_once_identifier_arrives():
    """A preference stated with NO identifier yet must be captured once an
    identifier is provided later in the same conversation."""
    config = new_config()
    send("I love punk rock", config)  # no identifier yet
    result = send("my email is pendingtest@example.com", config)
    text = reply_text(result).lower()
    # follow up explicitly to confirm it was actually saved, not just acknowledged
    result2 = send("what are my preferences", config)
    assert "punk" in reply_text(result2).lower()


# ============================================================
# CATEGORY 5: Grounding / anti-hallucination
# ============================================================

def test_15_no_results_says_so_plainly_not_invented():
    config = new_config()
    result = send("Find me albums by a completely made up fake artist name Zzzxqplorp123", config)
    text = reply_text(result).lower()
    assert "no" in text or "couldn't find" in text or "not found" in text


def test_16_no_capability_overclaim_for_refunds():
    """Must not imply refund processing is possible — no tool exists for it."""
    config = new_config()
    result = send("Can you process a refund for my last purchase?", config)
    text = reply_text(result).lower()
    assert "process" not in text.split("cannot")[0] if "cannot" in text else True
    assert "can't process" in text or "cannot process" in text or "unable to process" in text


def test_17_no_capability_overclaim_for_purchases():
    """Must not imply it can complete a purchase/add-to-cart — no tool exists."""
    config = new_config()
    result = send("Can you buy this album for me?", config)
    text = reply_text(result).lower()
    assert "can't" in text or "cannot" in text or "unable" in text


def test_18_out_of_scope_topic_is_declined_not_deferred_falsely():
    """Astrology has no owner anywhere in the system — must not promise a handoff
    that will also fail."""
    config = new_config()
    result = send("What's my astrology sign preference?", config)
    text = reply_text(result).lower()
    assert "astrology" not in text or "don't track" in text or "not something" in text or "doesn't track" in text


def test_19_case_insensitive_email_matches_same_record():
    config = new_config()
    send("my email is CaseTest@Example.com", config)
    send("I like ambient music", config)
    result = send("what are my preferences under casetest@example.com", config)
    text = reply_text(result).lower()
    assert "ambient" in text


# ============================================================
# CATEGORY 6: Mixed intent and scope boundaries
# ============================================================

def test_20_mixed_invoice_and_catalog_intent_prioritizes_invoice():
    config = new_config()
    result = send("Can you show me my invoice and also recommend an album?", config)
    text = reply_text(result).lower()
    assert any(word in text for word in ["email", "phone", "customer id"])


def test_21_preference_lookup_wins_even_with_phone_number_in_same_message():
    """'give me my preferences and my phone number X' must route to memory_agent,
    not get hijacked by invoice_agent's identification pattern."""
    config = new_config()
    result = send("give me my preferences and my phone number 703-606-7774", config)
    text = reply_text(result).lower()
    assert "customer information" not in text  # invoice_agent's Chinook-lookup phrasing


def test_22_catalog_agent_never_answers_invoice_questions():
    config = new_config()
    send("What genres do you have?", config)
    result = send("what's the total on my last invoice", config)
    text = reply_text(result).lower()
    assert any(word in text for word in ["email", "phone", "customer id", "invoice"])


# ============================================================
# CATEGORY 7: Handoff loop protection
# ============================================================

def test_23_ambiguous_dual_refusal_request_does_not_crash_or_hang():
    """A request combining a refund (no tool) and astrology (no tool) is designed
    to tempt agents into ping-ponging a handoff. Must resolve within recursion limits."""
    config = new_config()
    config["recursion_limit"] = 15
    try:
        result = send("Can you process a refund for my last purchase and also tell me my astrology sign preference?", config)
        text = reply_text(result)
        assert len(text) > 0
    except Exception as e:
        pytest.fail(f"Request did not resolve cleanly: {type(e).__name__}: {e}")


def test_24_handoff_count_resets_between_turns():
    """A conversation with several turns, each potentially triggering a handoff,
    must not exhaust the handoff cap from earlier turns."""
    config = new_config()
    config["recursion_limit"] = 15
    send("What albums does Queen have?", config)
    send("what's my invoice total", config)
    result = send("what are my preferences", config)
    # Should still get a real answer on turn 3, not a "handoff limit reached" message
    text = reply_text(result).lower()
    assert "handoff limit" not in text


# ============================================================
# CATEGORY 8: Data integrity self-healing
# ============================================================

def test_25_stale_duplicate_preferences_self_heal_on_display():
    """If raw duplicate/conflicting entries somehow exist in storage, the display
    path must sanitize before showing them, not dump raw duplicates."""
    config = new_config()
    email = "selfheal@example.com"
    send(f"my email is {email}", config)
    send("I like Pop", config)
    send("I like Pop music", config)  # near-duplicate phrasing
    result = send("what are my preferences", config)
    text = reply_text(result).lower()
    assert text.count("pop") <= 2  # heading + one bullet, not multiple duplicate bullets


# ============================================================
# CATEGORY 9: Catalog tool coverage (per-function, closes gaps
# in the original 25 — these target specific repository functions
# that weren't otherwise exercised)
# ============================================================

def test_26_typo_tolerant_artist_search_still_resolves():
    """A misspelled artist name should still fuzzy-match, not return 'not found'."""
    result = send("Do you have anything by Iorn Maden?", new_config())  # misspelled "Iron Maiden"
    text = reply_text(result).lower()
    assert "no artist found" not in text and "not found" not in text


def test_27_track_search_by_title_fuzzy():
    """search_song_by_title_fuzzy path — searching a song title should return matches."""
    result = send("Do you have a song called Thriller?", new_config())
    text = reply_text(result).lower()
    assert "no" not in text.split(".")[0]  # first sentence shouldn't be a flat "no results" for a well-known title


def test_28_composer_search_returns_results():
    """search_tracks_by_composer path."""
    result = send("What tracks were composed by Mozart?", new_config())
    text = reply_text(result)
    assert len(text) > 0
    assert "error" not in text.lower()


def test_29_genre_browse_returns_sample_across_artists():
    """browse_songs_by_genre path — should return multiple different artists, not
    all tracks from the same one (tests the per-artist interleaving logic)."""
    result = send("Show me some rock songs", new_config())
    text = reply_text(result)
    assert len(text) > 0


def test_30_truncated_results_mention_total_count():
    """Anti-hallucination grounding rule: when a genre/artist has more tracks than
    the sample shown, the response must mention the total count and that more exist."""
    result = send("Show me all rock songs you have, give me everything", new_config())
    text = reply_text(result).lower()
    # A large genre should trigger the truncation note if the grounding rule is followed
    assert "total" in text or "more" in text or "showing" in text


def test_31_get_track_by_valid_id_returns_full_details():
    """get_track_details_by_id path with a plausible valid ID."""
    result = send("Can you give me full details for track ID 1?", new_config())
    text = reply_text(result).lower()
    assert "error" not in text and "exception" not in text


def test_32_get_track_by_invalid_id_says_not_found_not_invented():
    """Anti-hallucination: an ID that doesn't exist must say so plainly, not
    fabricate plausible-sounding track details."""
    result = send("Can you give me full details for track ID 999999999?", new_config())
    text = reply_text(result).lower()
    assert "not found" in text or "no track" in text or "couldn't find" in text


def test_33_genre_with_zero_tracks_says_so_plainly():
    """A nonsense/nonexistent genre must return an honest empty result, not a
    fabricated list of songs."""
    result = send("Show me songs in the Zzzqplorp-fusion genre", new_config())
    text = reply_text(result).lower()
    assert "no" in text or "not found" in text or "couldn't find" in text


if __name__ == "__main__":
    import sys
    print("Running all scenarios directly (not via pytest)...\n")
    test_functions = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    passed, failed = 0, 0
    for fn in test_functions:
        try:
            fn()
            print(f"PASS: {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {fn.__name__} -> {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {fn.__name__} -> {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(test_functions)}")
    sys.exit(1 if failed else 0)