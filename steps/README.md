# Per-step runners

Each `stepNN_*.py` script exercises exactly ONE step's artifact using only the modules that step
introduces (plus earlier steps it builds on), and writes its output to `../artifacts/`. Run them
in order the first time; later steps assume earlier artifacts exist.

    export PYTHONPATH=../src        # from inside steps/, or set it once for the whole shell
    python step01_llm_gateway.py
    python step02_embedding_math.py
    ...
    python step12_start_api.py      # actually just prints the uvicorn command - see README.md
