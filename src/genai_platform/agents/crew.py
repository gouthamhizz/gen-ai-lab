"""STEP 9 - multi-agent orchestration with CrewAI: Researcher -> Writer -> Reviewer.

Execution graph (strict, sequential - each task receives the previous outputs through `context=`):

    research_task ──► write_task ──► review_task ──► final report
    (Researcher,        (Writer,       (Reviewer, fact-checks the draft
     uses kb_search)     no tools)      against the research notes)

CrewAI is imported lazily: it is a heavy dependency and only this module needs it.
"""
from __future__ import annotations

from genai_platform.config import get_settings


def crew_llm():
    from crewai import LLM

    s = get_settings()
    if s.llm_provider == "ollama":
        return LLM(model=f"ollama/{s.ollama_model}", base_url=s.ollama_host, temperature=0.2)
    return LLM(model=f"gemini/{s.gemini_model}", api_key=s.gemini_api_key, temperature=0.2)


def build_crew(retriever, llm=None, verbose: bool = False):
    from crewai import Agent, Crew, Process, Task
    from crewai.tools import tool

    llm = llm or crew_llm()

    @tool("kb_search")
    def kb_search(query: str) -> str:
        """Search the ingested document knowledge base. Input: a search query. Returns numbered passages with sources."""
        hits = retriever.retrieve(query, 5)
        return "\n".join(f"[{i}] ({h.cite}) {h.text[:600]}" for i, h in enumerate(hits, 1)) or "No relevant passages."

    researcher = Agent(role="Senior Researcher", goal="Collect accurate, sourced facts about {topic} from the knowledge base",
                       backstory="You never state a fact without a source; you note gaps honestly.",
                       tools=[kb_search], llm=llm, allow_delegation=False, verbose=verbose)
    writer = Agent(role="Technical Writer", goal="Turn research notes about {topic} into a clear, well-structured report",
                   backstory="You write only from the notes you are given and keep the source citations.",
                   llm=llm, allow_delegation=False, verbose=verbose)
    reviewer = Agent(role="Fact-Checking Reviewer", goal="Verify the draft against the research notes and fix any unsupported claim",
                     backstory="You are strict: an unsupported sentence is removed or flagged.",
                     llm=llm, allow_delegation=False, verbose=verbose)

    research = Task(description="Research '{topic}' using kb_search (run several different queries). List key facts, each with its source.",
                    expected_output="Bullet list of facts, each ending with (source, page).", agent=researcher)
    write = Task(description="Write a 300-500 word report on '{topic}' using ONLY the research notes. Keep citations.",
                 expected_output="A structured report with headings and citations.", agent=writer, context=[research])
    review = Task(description="Fact-check the report against the research notes. Remove or flag unsupported claims, "
                              "fix clarity, and return the FINAL report followed by a short 'Review notes' section.",
                  expected_output="Final corrected report + review notes.", agent=reviewer, context=[research, write])
    return Crew(agents=[researcher, writer, reviewer], tasks=[research, write, review], process=Process.sequential, verbose=verbose)


def run_report(retriever, topic: str, llm=None) -> dict:
    result = build_crew(retriever, llm).kickoff(inputs={"topic": topic})
    return {"topic": topic, "report": result.raw, "task_outputs": [t.raw for t in result.tasks_output]}
