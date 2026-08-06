from chinook_agent.agents import supervisor


def test_is_mixed_intent_detects_invoice_and_catalog_terms():
	assert supervisor._is_mixed_intent("show invoices and suggest jazz tracks") is True
	assert supervisor._is_mixed_intent("show invoices") is False
	assert supervisor._is_mixed_intent("suggest jazz tracks") is False


def test_supervisor_routes_mixed_intent_to_invoice_agent_without_llm(monkeypatch):
	class _FailingRouter:
		def invoke(self, _messages):
			raise AssertionError("LLM should not be called for deterministic mixed-intent routing")

	monkeypatch.setattr(supervisor, "router_llm", _FailingRouter())

	state = {
		"messages": [{"role": "user", "content": "show my invoices and suggest jazz tracks"}],
		"customer_id": None,
		"customer_email": None,
		"customer_phone": None,
		"authenticated": False,
		"preferences": [],
		"pending_preferences": [],
		"preferences_loaded": False,
		"next_agent": None,
		"response": None,
	}

	result = supervisor.supervisor_node(state)
	assert result["next_agent"] == "invoice_agent"
