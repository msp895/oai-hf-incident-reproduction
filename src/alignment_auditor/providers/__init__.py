"""In-repo custom Inspect model providers.

Importing this package registers the providers (via `@modelapi`) into Inspect's in-process
registry, so `get_model("meta/<model>")` and `--model meta/<model>` resolve without editing the
installed `inspect_ai` package or adding an entry point. Import it before any `get_model` call
(the harness and petri exp do this at module load).
"""
from . import meta as meta  # noqa: F401  (import registers the "meta" provider)
