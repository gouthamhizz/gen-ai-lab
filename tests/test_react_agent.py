from genai_platform.agents.react import ReActAgent
from genai_platform.agents.tools import default_tools
from genai_platform.llm.gateway import Chunk, Usage


class ScriptedGateway:
    """Fake gateway that plays back a fixed script of model outputs - no network, no real LLM."""

    def __init__(self, scripts):
        self._scripts = iter(scripts)

    def stream(self, messages, temperature=0.0, max_tokens=None, stop=None):
        for ch in next(self._scripts):
            yield Chunk(ch)
        yield Chunk("", Usage(10, 5))


def test_react_agent_calls_tool_then_answers():
    scripts = [
        "Thought: need to compute.\nAction: calculator\nAction Input: 21*2\n",
        "Thought: I now know the final answer\nFinal Answer: The result is 42.",
    ]
    agent = ReActAgent(ScriptedGateway(scripts), default_tools())
    events = list(agent.run_stream("what is 21*2"))
    kinds = [e["type"] for e in events]
    assert "action" in kinds and "observation" in kinds and "final" in kinds
    final = next(e for e in events if e["type"] == "final")
    assert "42" in final["answer"]


def test_react_agent_recovers_from_malformed_output():
    scripts = [
        "I forgot the format entirely.\n",
        "Thought: I now know the final answer\nFinal Answer: recovered.",
    ]
    agent = ReActAgent(ScriptedGateway(scripts), default_tools())
    final = next(e for e in agent.run_stream("q") if e["type"] == "final")
    assert final["answer"] == "recovered."


def test_react_agent_hits_step_limit():
    scripts = ["Thought: loop.\nAction: calculator\nAction Input: 1+1\n"] * 3
    agent = ReActAgent(ScriptedGateway(scripts), default_tools(), max_steps=2)
    final = next(e for e in agent.run_stream("q") if e["type"] == "final")
    assert final.get("truncated") is True
