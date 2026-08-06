# agents/supervisor.py

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

from ..config.settings import settings
from ..agents.state import AgentState
from ..agents.prompts import SUPERVISOR_PROMPT
from ..config.logging import get_logger

logger = get_logger(__name__)

VALID_AGENTS = ["invoice_agent", "catalog_agent", "memory_agent"]

# No tools bound here — the supervisor only makes a routing decision, it never calls tools itself
router_llm = ChatOpenAI(model=settings.DEFAULT_MODEL, temperature=0)


def supervisor_node(state: AgentState) -> dict:
    """Decide which specialist agent should be primary for this turn."""
    messages = [SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"]
    response = router_llm.invoke(messages)
    raw = response.content.strip().lower()

    agent_name = next((a for a in VALID_AGENTS if a in raw), None)

    if agent_name is None:
        logger.warning("supervisor produced unparseable routing decision", extra={"raw_response": raw})
        agent_name = "catalog_agent"  # safe default

    logger.info("supervisor routed request", extra={"next_agent": agent_name})

    return {"next_agent": agent_name}