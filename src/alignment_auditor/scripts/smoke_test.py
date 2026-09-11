"""Smoke-test API keys and target model IDs before running a Petri slice.

Usage: uv run smoke-test
"""

import sys

from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_TARGETS = ["claude-haiku-4-5", "claude-sonnet-5"]
OPENAI_TARGETS = ["gpt-5.6-terra", "gpt-5.6-luna"]


def main() -> int:
    ok = True

    # --- Anthropic ---
    try:
        import anthropic

        aclient = anthropic.Anthropic()
        models = [m.id for m in aclient.models.list()]
        for target in ANTHROPIC_TARGETS:
            status = "OK" if target in models else "MISSING"
            print(f"anthropic model {target}: {status}")
            if status == "MISSING":
                ok = False
                close = [m for m in models if target.split("-")[1] in m]
                print(f"  closest available: {close[:6]}")
        r = aclient.messages.create(
            model=ANTHROPIC_TARGETS[0],
            max_tokens=1,
            messages=[{"role": "user", "content": "hi"}],
        )
        print(f"anthropic auth ping: OK (model={r.model})")
    except Exception as e:
        ok = False
        print(f"anthropic FAILED: {type(e).__name__}: {e}")

    # --- OpenAI ---
    try:
        import openai

        oclient = openai.OpenAI()
        models = [m.id for m in oclient.models.list()]
        for target in OPENAI_TARGETS:
            status = "OK" if target in models else "MISSING"
            print(f"openai model {target}: {status}")
            if status == "MISSING":
                ok = False
                close = sorted(m for m in models if "5.6" in m or m.startswith("gpt-5"))
                print(f"  closest available: {close[:10]}")
        r = oclient.responses.create(model=OPENAI_TARGETS[1], input="hi", max_output_tokens=16)
        print(f"openai auth ping: OK (model={r.model})")
    except Exception as e:
        ok = False
        print(f"openai FAILED: {type(e).__name__}: {e}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
