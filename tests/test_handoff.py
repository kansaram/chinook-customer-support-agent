from chinook_agent.graph import build_graph

graph = build_graph()
config = {"configurable": {"thread_id": "test-loop-1"}, "recursion_limit": 10}  # lower than default, fails fast

try:
    result = graph.invoke({"messages": [{"role": "user", "content": "Can you process a refund for my last purchase and also tell me my astrology sign preference"}]}, config=config)
    print(result.get("response"))
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")