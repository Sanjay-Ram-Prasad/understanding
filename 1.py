from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# =========================================================
# 1. STATE
# =========================================================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# =========================================================
# 2. TOOL
# =========================================================

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    
    weather = {
        "kochi": "32°C, sunny",
        "delhi": "35°C, hot",
        "bangalore": "24°C, cloudy"
    }

    return weather.get(
        city.lower(),
        f"Weather data unavailable for {city}"
    )


tools = [get_weather]


# =========================================================
# 3. LLM
# =========================================================

llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)

llm_with_tools = llm.bind_tools(tools)


# =========================================================
# 4. LLM NODE
# =========================================================

def llm_node(state: AgentState):

    print("\n--- LLM NODE ---")

    response = llm_with_tools.invoke(
        state["messages"]
    )

    print("LLM response:")
    print(response)

    return {
        "messages": [response]
    }


# =========================================================
# 5. TOOL NODE
# =========================================================

tool_node = ToolNode(tools)


# =========================================================
# 6. BUILD GRAPH
# =========================================================

graph = StateGraph(AgentState)


# Add nodes
graph.add_node("llm", llm_node)
graph.add_node("tools", tool_node)


# =========================================================
# 7. CONNECTIONS / TRANSITIONS
# =========================================================

# START → LLM
graph.add_edge(START, "llm")


# LLM → ?
# If LLM requested a tool → tools
# If LLM did NOT request a tool → END
graph.add_conditional_edges(
    "llm",
    tools_condition,
    {
        "tools": "tools",
        END: END
    }
)


# TOOL → LLM
graph.add_edge("tools", "llm")


# Compile
app = graph.compile()


# =========================================================
# 8. RUN
# =========================================================

result = app.invoke({
    "messages": [
        HumanMessage(
            content="What is the weather in Kochi?"
        )
    ]
})


# =========================================================
# 9. FINAL RESULT
# =========================================================

print("\n====================")
print("FINAL MESSAGES")
print("====================")

for message in result["messages"]:
    print(type(message).__name__)
    print(message.content)
    print()