"""SSE helpers: one JSON event per line, `event:` name + `data:` payload, per the SSE spec."""
from __future__ import annotations

import json
from typing import Any, AsyncIterator


def sse(event: str, data: Any) -> dict:
    return {"event": event, "data": json.dumps(data, default=str)}


async def guarded(events: AsyncIterator[dict]) -> AsyncIterator[dict]:
    """Turn a mid-stream exception into a terminal `error` SSE event instead of a bare disconnect,
    so the client's EventSource always gets a clean, parseable end to the stream."""
    try:
        async for ev in events:
            yield ev
    except Exception as e:  # noqa: BLE001
        yield sse("error", {"message": str(e)})
    finally:
        yield sse("done", {})
