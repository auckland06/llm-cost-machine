# llm-cost-machine

A Python tool that sends the same prompts to OpenAI, Anthropic (Claude), and Google (Gemini), measures the token usage of every call, converts it to dollars, and checks whether the answers are actually correct.

The goal: find out which model gives the most for the money. Not the cheapest price per token, but the cheapest **correct** answer.

## Why I built it

LLM pricing looks simple (a price per million tokens), but the real cost of a request depends on how many tokens a model decides to use, including hidden "thinking" tokens you never see but still pay for. I wanted to measure that directly instead of comparing price sheets.

## How it works

- Sends each prompt to all three models multiple times and averages the results, since token usage varies from run to run
- Tracks output tokens, hidden reasoning tokens, and visible answer tokens separately
- Converts token counts to cost using each model's input and output prices
- For prompts with a known answer, checks correctness and calculates **cost per correct answer**
- Handles failed API calls (like rate limits) without crashing the whole test

Prompts cover a range of task types: a constrained task, a factual question, easy math, a coding task, a short writing task, and a hard combinatorics problem.

## Key findings

### 1. Hidden reasoning tokens are often most of the bill
On "Say hello in five words," about **98%** of what Gemini billed was hidden thinking: roughly 336 thinking tokens to produce a 7-token answer (averaged over 20 runs).

### 2. Thinking doesn't track human difficulty
OpenAI used about **20** thinking tokens on an easy math word problem, about **90** on a coding task, and about **300** on the hard combinatorics problem. Gemini thought heavily on almost everything, including a simple paragraph. What's hard for a person isn't necessarily what makes a model think more.

### 3. Cost per call can be misleading. Cost per correct answer is what matters
Hard problem: *"How many integers from 1 to 100,000 have digits that add up to 23?"* (correct answer: 6,000), 10 runs per model in a single session:

| Model | Correct | Avg cost per call | Approx. cost per 1,000 correct answers |
|---|---|---|---|
| OpenAI | 10/10 | $0.00022 | ~$0.22 |
| Claude | 9/10 | $0.00046 | ~$0.51 |
| Gemini | 10/10 | $0.0090 | ~$8.99 |

Results were consistent across multiple sessions: costs stayed within a few percent, and the ranking didn't change.

Gemini's higher cost comes from two factors that multiply: it uses more tokens (mostly thinking) **and** it charges more per token.

### 4. More thinking didn't guarantee a correct answer
One Gemini run spent about 1,800 thinking tokens and still made an arithmetic slip (computed one term as 1,190 instead of 2,380). Claude, with thinking not enabled, made the same specific mistake twice across runs (using C(10,2) instead of C(5,2)). It caught and corrected it once, and missed it once.

### 5. The evaluator needed testing too
My first answer checker said OpenAI got **0/5** on the hard problem. That seemed too striking, so I read the raw responses. OpenAI was formatting its answers in LaTeX (`6{,}000`), which my regex didn't match. After normalizing number formatting, OpenAI was actually 5/5 and the cheapest correct model. Lesson: verify the evaluation, not just the model.

## Limitations

- **Small samples.** 5 to 20 runs per prompt. With 5 runs, a single wrong answer swings accuracy by 20 points.
- **Uneven comparison.** OpenAI and Gemini use reasoning by default. Claude's extended thinking was not enabled.
- **Prices are entered by hand** in the code and need to be kept current with each provider's pricing page.
- **The answer checker is simple.** It passes if the correct number appears anywhere in the response, not necessarily as the final answer.
- **Gemini's thinking tokens are calculated**, not reported directly (total minus prompt minus answer), because of the SDK version used.

## How to run it

1. Install dependencies:
   ```bash
   pip install openai anthropic google-generativeai python-dotenv
   ```
2. Create a `.env` file in the project folder with your own API keys:
   ```
   OPENAI_API_KEY=your_key
   ANTHROPIC_API_KEY=your_key
   GEMINI_API_KEY=your_key
   ```
   `.env` is listed in `.gitignore`, so keys are never uploaded.
3. Run it:
   ```bash
   python3 tracker.py
   ```
   Change `PROMPTS` and `RUNS` at the top of the file to test different prompts or run counts.

## Next steps

- Save every run (including responses) to a CSV for analysis
- Add charts comparing cost and accuracy per prompt
- Enable Claude's extended thinking for a fair reasoning-vs-reasoning comparison
- Add more hard prompts with verified answers, and run them more times
- Check generated code by running it against test cases in a sandbox
- Build a router that sends each prompt to the cheapest model likely to answer it correctly
