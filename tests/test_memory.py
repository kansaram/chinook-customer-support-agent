import types

from chinook_agent.agents import memory_agent
from chinook_agent.agents.state import add_preferences
from chinook_agent.database.memory_repository import save_preferences_list


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


def test_extract_explicit_music_preferences_includes_negative_statements():
	messages = [{"role": "user", "content": "I don't like Rock"}]

	found = memory_agent._extract_explicit_music_preferences(messages)
	assert found == ["I don't like Rock"]


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
	assert "You like AC/DC" in calls[0][1]
	assert "preferences" in updates
	assert "You like AC/DC" in updates["preferences"]


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


def test_memory_llm_node_records_contradicting_negative_preference(monkeypatch):
	monkeypatch.setattr(memory_agent, "llm", _DummyLLM())

	calls = []

	def _fake_save(identifier: str, preferences: list[str]):
		calls.append((identifier, preferences))

	monkeypatch.setattr(memory_agent, "save_preferences_list", _fake_save)

	state = {
		"messages": [{"role": "user", "content": "I don't like Rock"}],
		"customer_id": None,
		"customer_email": "fan@example.com",
		"customer_phone": None,
		"authenticated": False,
		"preferences": ["I love Rock"],
		"pending_preferences": [],
		"preferences_loaded": True,
		"next_agent": "catalog_agent",
		"response": None,
	}

	updates = memory_agent.memory_llm_node(state)

	assert calls
	assert calls[0][0] == "email:fan@example.com"
	assert "You dislike Rock" in calls[0][1]
	assert "preferences" in updates
	assert "You dislike Rock" in updates["preferences"]
	assert "I love Rock" not in calls[0][1]


def test_memory_llm_node_returns_saved_preferences_when_email_and_lookup_are_in_same_message(monkeypatch):
	email = "fan-lookup@example.com"
	save_preferences_list(f"email:{email}", ["You like Pop; You dislike Jazz"])

	state = {
		"messages": [
			{"role": "user", "content": f"Hi my email is {email}. could you please give me my preferences"},
		],
		"customer_id": None,
		"customer_email": None,
		"customer_phone": None,
		"authenticated": False,
		"preferences": [],
		"pending_preferences": [],
		"preferences_loaded": False,
		"next_agent": "memory_agent",
		"response": None,
	}

	updates = memory_agent.memory_llm_node(state)

	assert updates["response"]
	assert "Here are your saved preferences" in updates["response"]
	assert "You like Pop" in updates["response"]
	assert "You dislike Jazz" in updates["response"]


def test_resolve_identifier_prefers_email_over_customer_id():
	assert memory_agent.resolve_identifier(customer_id=7, email="fan@example.com") == "email:fan@example.com"


def test_add_preferences_replaces_same_subject_preferences():
	merged = add_preferences(["I love Rock"], ["I don't like Rock"])

	assert merged == ["I don't like Rock"]
