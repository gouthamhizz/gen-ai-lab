"""STEP 8b - a ReAct (Reason + Act) agent written from scratch.

Loop:   Thought -> Action -> Action Input -> [system runs the tool] -> Observation -> ... -> Final Answer

Design decisions worth knowing:
  * we pass stop=["Observation:"] so the model stops BEFORE inventing tool results (the classic ReAct failure);
  * output is parsed with regexes; a malformed step is fed back as an Observation so the model can self-correct;
  * hard cap on steps (default 6) prevents infinite loops / runaway cost;
  * `run_stream` yields events (thought / action / observation / token / final) so the API can forward them over SSE;
  * tool exceptions become observations - the agent sees the error and can try something else.
"""
from __future__ import annotations

import re
import time
from typing import Iterator, Optional

from genai_platform.agents.tools import Tool
from genai_platform.llm.gateway import LLMGateway, Usage

REACT_SYSTEM = """You are a reasoning agent. Answer the question using the tools when they help.

Use EXACTLY this format:

Thought: what you are thinking and what you need next
Action: the tool name, one of [{tool_names}]
Action Input: the input for the tool, on a single line
Observation: (written by the system after the tool runs - NEVER write this line yourself)
... (Thought / Action / Action Input / Observation can repeat)
Thought: I now know the final answer
Final Answer: the answer to the original question (cite [n] passages if you used rag_search)

Available tools:
{tool_desc}

Rules: call ONE tool at a time; do math with the calculator, never in your head; if a tool errors, try a different input or tool;
if no tool is needed, go straight to Thought and Final Answer."""

_ACTION = re.compile(r"Action:\s*(?P<name>[^\n]+?)\s*\n\s*Action Input:\s*(?P<input>.+)", re.S)
_FINAL = "Final Answer:"


class ReActAgent:
    def __init__(self, gateway: LLMGateway, tools: list[Tool], max_steps: int = 6, obs_limit: int = 1500):
        self.gw, self.tools = gateway, {t.name: t for t in tools}
        self.max_steps, self.obs_limit = max_steps, obs_limit
        self.system = REACT_SYSTEM.format(tool_names=", ".join(self.tools),
                                          tool_desc="\n".join(f"- {t.name}: {t.description}" for t in tools))

    def _call_tool(self, name: str, arg: str) -> str:
        tool = self.tools.get(name.strip().strip("[]'\""))
        if tool is None:
            return f"Error: unknown tool '{name}'. Available: {', '.join(self.tools)}"
        try:
            out = str(tool(arg.strip().strip('"')))
        except Exception as e:  # noqa: BLE001
            out = f"Error while running {name}: {e}"
        return out[: self.obs_limit] + (" ...[truncated]" if len(out) > self.obs_limit else "")

    def run_stream(self, question: str, history: str = "") -> Iterator[dict]:
        scratch, usage, t0 = "", Usage(), time.perf_counter()
        head = (f"Conversation so far:\n{history}\n\n" if history else "") + f"Question: {question}\n"
        for step in range(1, self.max_steps + 1):
            messages = [{"role": "system", "content": self.system}, {"role": "user", "content": head + scratch}]
            buf, emitted, in_final = "", 0, False
            for ch in self.gw.stream(messages, temperature=0.0, stop=["Observation:"]):
                if ch.usage:
                    usage.prompt_tokens += ch.usage.prompt_tokens
                    usage.completion_tokens += ch.usage.completion_tokens
                buf += ch.text
                fi = buf.find(_FINAL)
                if not in_final and fi != -1 and "Action Input:" not in buf[:fi]:
                    in_final, emitted = True, fi + len(_FINAL)
                if in_final and len(buf) > emitted:
                    piece = buf[emitted:]
                    if emitted == fi + len(_FINAL):
                        piece = piece.lstrip()
                    if piece:
                        yield {"type": "token", "text": piece}
                    emitted = len(buf)
            fi, m = buf.find(_FINAL), _ACTION.search(buf)
            if fi != -1 and (m is None or fi < m.start()):
                yield {"type": "final", "answer": buf[fi + len(_FINAL):].strip(), "steps": step,
                       "usage": {"prompt": usage.prompt_tokens, "completion": usage.completion_tokens},
                       "latency_s": round(time.perf_counter() - t0, 3)}
                return
            if m:
                thought = buf[: m.start()].replace("Thought:", "").strip()
                name, arg = m.group("name").strip(), m.group("input").strip().splitlines()[0]
                yield {"type": "thought", "step": step, "text": thought}
                yield {"type": "action", "step": step, "tool": name, "input": arg}
                obs = self._call_tool(name, arg)
                yield {"type": "observation", "step": step, "text": obs}
                scratch += buf[: m.start()] + f"Action: {name}\nAction Input: {arg}\nObservation: {obs}\n"
            else:  # format violation: tell the model, do not crash
                yield {"type": "thought", "step": step, "text": buf.strip() or "(empty output)"}
                scratch += buf.rstrip() + "\nObservation: Invalid format. Reply with Thought/Action/Action Input, or Thought/Final Answer.\n"
        yield {"type": "final", "answer": "I could not finish within the step limit.", "steps": self.max_steps,
               "usage": {"prompt": usage.prompt_tokens, "completion": usage.completion_tokens},
               "latency_s": round(time.perf_counter() - t0, 3), "truncated": True}

    def run(self, question: str, history: str = "") -> dict:
        trace, final = [], {}
        for ev in self.run_stream(question, history):
            (final.update(ev) if ev["type"] == "final" else trace.append(ev)) if ev["type"] != "token" else None
        return {**final, "trace": trace}
