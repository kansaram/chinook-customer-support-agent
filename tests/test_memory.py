import types

from chinook_agent.agents import memory_agent


class _DummyResponse:
	def __init__(self):
		self.tool_calls = []
		self.content = ""


class _DummyLLM:
	def invoke(self, _messages):
		return _DummyResponse()


def test_extract_explicit_music_preferences_from_user_only():
	messages = [
		{"role": "assistant", "content": "I love rock"},
		{"role": "user", "content": "I love rock. AC/DC is my favorite"},
	]

	found = memory_agent._extract_explicit_music_preferences(messages)
	assert "I love rock" in found
	assert "AC/DC is my favorite" in found
	assert len(found) == 2


def test_memory_llm_node_queues_detected_preferences_without_identifier(monkeypatch):
	monkeypatch.setattr(memory_agent, "llm", _DummyLLM())

	state = {
		"messages": [{"role": "user", "content": "I like Jazz"}],
		"customer_id": None,
		"customer_email": None,
		"customer_phone": None,
		"authenticated": False,
		"preferences": [],
		"pending_preferences": [],
		"preferences_loaded": False,
		"next_agent": "catalog_agent",
		"response": None,
	}

	updates = memory_agent.memory_llm_node(state)
	assert "pending_preferences" in updates
	assert "I like Jazz" in updates["pending_preferences"]


def test_memory_llm_node_saves_detected_preferences_with_identifier(monkeypatch):
	monkeypatch.setattr(memory_agent, "llm", _DummyLLM())

	calls = []

	def _fake_save(identifier: str, preferences: list[str]):
		calls.append((identifier, preferences))

	monkeypatch.setattr(memory_agent, "save_preferences_list", _fake_save)

	state = {
		"messages": [{"role": "user", "content": "AC/DC is my favorite"}],
		"customer_id": None,
		"customer_email": "fan@example.com",
		"customer_phone": None,
		"authenticated": False,
		"preferences": [],
		"pending_preferences": [],
		"preferences_loaded": True,
		"next_agent": "catalog_agent",
		"response": None,
	}

	updates = memory_agent.memory_llm_node(state)

	assert calls
	assert calls[0][0] == "email:fan@example.com"
	assert "AC/DC is my favorite" in calls[0][1]
	assert "preferences" in updates
	assert "AC/DC is my favorite" in updates["preferences"]


def test_memory_llm_node_does_not_append_existing_preference_case_insensitively(monkeypatch):
	monkeypatch.setattr(memory_agent, "llm", _DummyLLM())

	calls = []

	def _fake_save(identifier: str, preferences: list[str]):
		calls.append((identifier, preferences))

	monkeypatch.setattr(memory_agent, "save_preferences_list", _fake_save)

	state = {
		"messages": [{"role": "user", "content": "i love rock"}],
		"customer_id": None,
		"customer_email": "fan@example.com",
		"customer_phone": None,
		"authenticated": False,
		"preferences": ["I Love Rock"],
		"pending_preferences": [],
		"preferences_loaded": True,
		"next_agent": "catalog_agent",
		"response": None,
	}

	updates = memory_agent.memory_llm_node(state)

	assert calls == []
	assert "preferences" not in updates
