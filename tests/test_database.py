from collections import Counter

from chinook_agent.database.repository import browse_songs_by_genre
from chinook_agent.database.repository import search_song_by_title_fuzzy
from chinook_agent.database.repository import get_track_details_by_id


def test_browse_songs_by_genre_returns_results_for_known_genre():
	result = browse_songs_by_genre("Rock", sample_size=12, per_artist_cap=2)

	assert result is not None
	assert result["genre_name"].lower() == "rock"
	assert result["total_tracks"] > 0
	assert result["total_artists"] > 1
	assert len(result["sample"]) > 0


def test_browse_songs_by_genre_sample_spans_multiple_artists():
	result = browse_songs_by_genre("Rock", sample_size=12, per_artist_cap=2)

	assert result is not None
	artists = {row["artist_name"] for row in result["sample"]}
	assert len(artists) > 1


def test_browse_songs_by_genre_respects_per_artist_cap():
	result = browse_songs_by_genre("Rock", sample_size=12, per_artist_cap=1)

	assert result is not None
	counts = Counter(row["artist_name"] for row in result["sample"])
	assert counts
	assert max(counts.values()) <= 1


def test_browse_songs_by_genre_returns_none_for_unknown_genre():
	assert browse_songs_by_genre("no-such-genre-xyz") is None


def test_search_song_by_title_fuzzy_exact_match_returns_results():
	result = search_song_by_title_fuzzy("Balls to the Wall", limit=5)

	assert result
	assert any(row["track_name"] == "Balls to the Wall" for row in result)
	assert all("artist_name" in row and "album_title" in row for row in result)


def test_search_song_by_title_fuzzy_typo_still_matches():
	result = search_song_by_title_fuzzy("Bals to the Wal", limit=5, threshold=50)

	assert result
	assert any("Balls to the Wall" == row["track_name"] for row in result)


def test_get_track_details_by_id_returns_complete_details():
	result = get_track_details_by_id(1)

	assert result is not None
	assert result["track_id"] == 1
	assert result["track_name"]
	assert "unit_price" in result
	assert "milliseconds" in result
	assert "media_type_name" in result


def test_get_track_details_by_id_returns_none_for_unknown_track():
	assert get_track_details_by_id(999999) is None
