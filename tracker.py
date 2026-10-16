import os
import time
import warnings

# Hides the Python 3.9 / deprecated-package warnings so the output is readable.
# The real fix is upgrading to Python 3.11+ and switching to google-genai.
warnings.filterwarnings("ignore")

from statistics import mean
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai

load_dotenv()

PROMPTS = [
    "Say hello in five words.",
    "What is the capital of Australia?",
    "A train travels 180 miles in 3 hours. How many miles does it travel in 5 hours?",
    "Write a Python function that checks if a string is a palindrome.",
    "Write a short paragraph explaining why sleep matters for athletes.",
]
RUNS = 5

PRICES = {
    "gpt-6-luna": (0.10, 0.50),
    "claude-haiku-5-5": (0.10, 0.50),
    "gemini-3.8-flash": (0.75, 3.75),
}


def cost(model, tokens_in, tokens_out):
    rate_in, rate_out = PRICES[model]
    return (tokens_in * rate_in + tokens_out * rate_out) / 1_000_000


openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
claude = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini = genai.GenerativeModel("gemini-3.8-flash")


# Each function makes one call and returns (cost, output_tokens, thinking_tokens)

def run_openai(prompt):
    r = openai_client.chat.completions.create(
        model="gpt-6-luna",
        messages=[{"role": "user", "content": prompt}],
    )
    u = r.usage
    details = u.completion_tokens_details
    reasoning = (details.reasoning_tokens or 0) if details else 0
    return cost("gpt-6-luna", u.prompt_tokens, u.completion_tokens), u.completion_tokens, reasoning


def run_claude(prompt):
    m = claude.messages.create(
        model="claude-haiku-5-5",
        max_tokens=10000,
        messages=[{"role": "user", "content": prompt}],
    )
    u = m.usage
    if m.stop_reason == "max_tokens":
        print("  (Claude answer was cut off)")
    return cost("claude-haiku-5-5", u.input_tokens, u.output_tokens), u.output_tokens, None


def run_gemini(prompt):
    res = gemini.generate_content(prompt)
    meta = res.usage_metadata
    thoughts = meta.total_token_count - meta.prompt_token_count - meta.candidates_token_count
    out = meta.candidates_token_count + thoughts
    return cost("gemini-3.8-flash", meta.prompt_token_count, out), out, thoughts


PROVIDERS = {
    "OpenAI": run_openai,
    "Claude": run_claude,
    "Gemini": run_gemini,
}

# results[prompt][model] is a list of (cost, output_tokens, thinking_tokens) tuples
results = {prompt: {name: [] for name in PROVIDERS} for prompt in PROMPTS}

for p_num, prompt in enumerate(PROMPTS, start=1):
    for i in range(1, RUNS + 1):
        for name, run in PROVIDERS.items():
            try:
                results[prompt][name].append(run(prompt))
            except Exception as e:
                # One failed call shouldn't kill the whole test
                print(f"  Run {i}: {name} failed: {e}")
        print(f"Prompt {p_num}/{len(PROMPTS)}, run {i}/{RUNS}")
        time.sleep(0.5)  # small pause to avoid rate limits

print(f"\n--- Averages over {RUNS} runs ---")
for prompt in PROMPTS:
    print(f"=== {prompt} ===\n")
    for name, rows in results[prompt].items():
        if not rows:
            print(f"{name}: no successful runs")
            continue
        costs = [r[0] for r in rows]
        outputs = [r[1] for r in rows]
        thinking = [r[2] for r in rows if r[2] is not None]
        print(f"{name} ({len(rows)} runs)")
        print(f"  Avg cost:      ${mean(costs):.8f}")
        print(f"  Min / max:     ${min(costs):.8f} / ${max(costs):.8f}")
        print(f"  Avg output:    {mean(outputs):.1f} tokens")
        if thinking:
            print(f"  Avg thinking:  {mean(thinking):.1f} tokens")
            print(f"  Avg visible:   {mean(outputs) - mean(thinking):.1f} tokens")
        else:
            print("  Avg thinking:  N/A (not enabled)")
            print(f"  Avg visible:   {mean(outputs):.1f} tokens")
        print()
