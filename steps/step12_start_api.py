"""STEP 12: the final artifact is the running service, not a one-off script. Start it with:

    uvicorn genai_platform.api.main:app --reload --port 8000        # local dev
    docker compose --profile full up -d --build                     # full stack incl. monitoring

This script just prints the equivalent info and a couple of example curl calls.
"""
import textwrap

print(textwrap.dedent("""
    Start the API:
      cd .. && uvicorn genai_platform.api.main:app --host 0.0.0.0 --port 8000 --reload
      # or:  docker compose --profile full up -d --build

    Try it:
      curl -N -H "X-API-Key: dev-key-change-me" -H "Content-Type: application/json" \\
        -d '{"message":"What is the warranty period?","session_id":"s1"}' \\
        http://localhost:8000/chat/stream

      curl http://localhost:8000/health
      curl http://localhost:8000/metrics | head
"""))
