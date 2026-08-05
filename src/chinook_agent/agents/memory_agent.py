from langchain_openai import ChatOpenAI
from ..config.settings import settings
from ..agents.state import AgentState
from ..agents.prompts import MEMORY_AGENT_PROMPT
from ..tools.memory_tools import save_preference, get_preferences
from ..config.logging import get_logger

logger = get_logger(__name__)


MEMORY_TOOLS = [save_preference, get_preferences]

llm = ChatOpenAI(model=settings.DEFAULT_MODEL, temperature=0).bind_tools(MEMORY_TOOLS)


def memory_llm_node(state: AgentState) -> dict:
    """The memory agent's reasoning step — decides whether to save/retrieve a preference or respond directly."""
    messages = [{"role": "system", "content": MEMORY_AGENT_PROMPT}] + state["messages"]
    logger.info(
        "memory_agent invoked",
        extra={"customer_id": state.get("customer_id"), "customer_email": state.get("customer_email")},
    )

    response = llm.invoke(messages)
    return {"messages": [response]}