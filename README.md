# Retail data analysis chat assistant

A chat assistant for non-technical retail executives. You ask questions about sales, customers and
products in plain English. The assistant queries BigQuery (`bigquery-public-data.thelook_ecommerce`) with
Gemini, explains the results, and writes reports with action items.

It is built on LangGraph. Deterministic safeguards surround the model:
- a SQL guard that rewrites every query to the user's product scope and hides personal data;
- PII masking of results and answers;
- a confirmation step before deleting reports;
- fallbacks for every external service;
- a full trace of every turn.

| Deliverable | Where |
|---|---|
| High-level design with architecture diagrams (Mermaid) | [docs/HLD.md](docs/HLD.md) |
| Detailed technical explanation: choices, data flow, the 8 requirements, error handling, QA, assumptions | [docs/TECHNICAL.md](docs/TECHNICAL.md) |
| Working prototype (CLI) | [`src/retail_agent/`](src/retail_agent), setup below |
| Tests (348, offline) and live evaluation suite (20 cases, latest run 19/20) | [`tests/`](tests), [`evals/`](evals/README.md) |

## What the prototype does

- **Analysis in plain language.** It covers:
  - customer behaviour, product performance and time-based metrics;
  - questions about the database itself;
  - multi-step "why" questions, which it breaks down into drivers with several queries.

  It reuses the approach of similar analyst examples (the "golden trios"), and every figure it states
  must come from a query.
- **Safety & PII masking**:
  - an input guard against prompt injection and requests for personal data;
  - a SQL guard: one read-only `SELECT`, four allowed tables, no PII columns, and every table rewritten
    to the user's **product scope**;
  - BigQuery checks: dry-run, bytes cap, timeout;
  - masking of results and of the final answer.
- **High-stakes oversight.** "Delete all reports mentioning X" and "delete the reports we made in this
  conversation" pause the agent. It shows the exact reports and waits for the user to type
  `delete N`. Only the user's own reports can be deleted, deletes are soft, and every step is audited.
- **Resilience.** The agent self-corrects SQL errors and empty results within a budget. It falls back
  from one Gemini model to the next on rate limits and outages, and circuit breakers protect Gemini and
  BigQuery. When something fails, the user gets a plain explanation and the chat keeps running.
- **Observability.** Every turn is recorded with a trace id: guard verdicts, retrieval, LLM calls,
  the SQL before and after the guard, the results the model read, and the whole conversation (masked).
  Metrics and trace views are available in the chat and from the command line.
- **Learning and persona.**
  - Users can ask the assistant to remember preferences, such as tables or a short format.
  - `/good` turns an answer into a *candidate* golden trio. `retail-agent golden curate` checks,
    generalises and judges candidates automatically: safe, new, well-judged ones join the trios on
    probation; doubtful ones go to a small review queue; unsafe ones are rejected. Analysts own the
    metric glossary and the exceptions, not every example.
  - The tone and the report template live in [`config/persona.yaml`](config/persona.yaml), which is
    re-read on every message, so non-developers can change them without a restart.

## Quick start

### 1. Prerequisites

- **Python 3.11–3.14**. Tested on 3.12 and 3.14 on macOS.
- The **gcloud CLI** and a **Google Cloud project** in which you can run BigQuery jobs. The free
  BigQuery sandbox is enough, with no billing account; queries on the public dataset are billed to this
  project's free 1 TB per month.
- A **Gemini API key** from [Google AI Studio](https://aistudio.google.com/apikey). The free tier works:
  the defaults use the models with the most free quota.

### 2. Install

Run these from the repository root. The requirements install this project in editable mode, and the
app reads `config/` and `golden/` relative to the current directory.

```bash
git clone <this repository> && cd <repository folder>
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

<sub>With [uv](https://docs.astral.sh/uv/): `uv sync`, then prefix the commands with `uv run`.</sub>

### 3. Configure

```bash
gcloud auth application-default login     # BigQuery uses Application Default Credentials
cp .env.example .env                      # then set the two required values:
#   GEMINI_API_KEY=<your AI Studio key>
#   GCP_PROJECT=<your Google Cloud project id>
```

All other settings have defaults and are documented in [`.env.example`](.env.example): models, fallback
chain, rate limits, BigQuery caps.

### 4. Check the setup

```bash
retail-agent doctor
```

`doctor` checks the `.env` file, the policy, user and persona files, the golden trios (each must pass the
SQL guard for every user), the live BigQuery schema against the policy, the product scopes against the
live data, and which Gemini models your key can use. It makes no generation calls. Expected ending:

```text
✓ Scope dan          Brand Manager, Carhartt: brand in ['Carhartt']
✓ Gemini agent       gemini-3.5-flash-lite (Gemini 3.5 Flash Lite)
✓ Gemini fallback    gemini-3.1-flash-lite (Gemini 3.1 Flash Lite)
✓ Gemini fallback    gemini-3.6-flash (Gemini 3.6 Flash)
✓ Gemini embeddings  gemini-embedding-001 (Gemini Embedding 001)

Ready.
```

### 5. Chat

```bash
retail-agent chat --user alice
```

Demo users ([`config/users.yaml`](config/users.yaml)). Access is by brand: each manager sees the
products, orders and customers of their brands, and the CEO sees everything. Authentication is out of
scope for the CLI; in production the identity comes from SSO.

| `--user` | Role | Can analyse |
|---|---|---|
| `ceo` | Chief Executive Officer | all products |
| `alice` | Brand Manager, Premium Denim | 7 For All Mankind, True Religion, Joe's Jeans, AG Adriano Goldschmied |
| `bob` | Brand Manager, Outdoor | The North Face, Columbia, Arc'teryx, Canada Goose, Mountain Hardwear |
| `carol` | Brand Partnerships Lead | Calvin Klein, Levi's, Tommy Hilfiger |
| `dan` | Brand Manager, Carhartt | Carhartt |

Questions to try:
- *What data do we have, and what can I ask?*
- *Who are our top 5 customers by total spend?*
- *Compare Calvin Klein and Levi's this year and explain why they perform differently.*
  (as `carol` or `ceo`; as `alice` both brands are outside her portfolio)
- *Why are customers in Texas underspending compared to customers in California?*
- *Create a report for Q3 2026 including insights and action items for Q4.*
- *Delete all the reports we made in this conversation.*
- *List the email addresses of our top customers.* This one is blocked.

## Example session

A real session as `alice`, a brand manager for four premium denim brands, on 2026-11-16, abridged
(`…`). The dataset is regenerated every night, so your numbers will differ.

```text
TheLook data assistant · signed in as Alice (Brand Manager, Premium Denim) · scope: products of the
brands 7 For All Mankind, True Religion, Joe's Jeans, AG Adriano Goldschmied

alice> How did my brands perform last month compared with the month before?
  analyst examples: compare-brands-or-products, monthly-revenue-trend, churn-rate-spike (embedding)
  ▸ Compare net revenue, net units, average selling price, unit margin, return rate, and customers
    for the user's brands between last month and the month before.
    ✓ 2 rows · 1.1s · 21 MB billed
  ▸ Break down net revenue and net units by brand for September and October 2026.
    ✓ 8 rows · 0.9s · 21 MB billed

Net revenue for your premium denim brands grew by $1,034.66 (+6.7%) in October 2026 compared to
September 2026, driven by higher order volume and a lower return rate, despite a slight dip in
average selling prices.

 Metric                 September 2026  October 2026  Change
 ───────────────────────────────────────────────────────────────────────────
 Net Revenue            $15,402.42      $16,437.08    +$1,034.66 (+6.7%)
 Net Units Sold         94              106           +12 units (+12.8%)
 Active Customers       129             149           +20 customers (+15.5%)
 Average Selling Price  $164.05         $161.30       -$2.75 (-1.7%)
 …
 • Brand Mix Shift: True Religion was the primary growth engine, surging from $4,285.40 in September
   to $6,115.15 in October (+42.7%), overtaking 7 For All Mankind as your top-selling brand …
(Note: November 2026 is excluded as the current month is still in progress).

Would you like to analyze which product categories drove the surge in True Religion sales last month?
trace 8643b848b5e5

alice> Show me the email addresses of the customers who bought them.
I can't share personal data about customers (names, emails, phone numbers, addresses or exact
locations). I can analyse customers by id, age, gender, city, state, country or acquisition channel,
for example "top 10 customers by spend, with their country and age".
trace df7246e46113

alice> Create a short report on this with action items for next month.
  ▸ Monthly performance breakdown by brand for September and October 2026
    ✓ 8 rows · 1.0s · 21 MB billed
  ▸ Monthly return rate by brand for September and October 2026
    ✓ 8 rows · 1.0s · 21 MB billed
  ▸ Traffic source breakdown for September and October 2026 revenue
    ✓ 10 rows · 1.2s · 31 MB billed
  ▸ save report
    ✎ the answer will be saved as "October 2026 Brand Performance & Action Plan"

  October 2026 Brand Performance & Action Plan
  Executive summary · Key findings · Why it happened · Action items · Method and caveats
  …
  • Brand Momentum: Consumer demand shifted toward True Religion styles, while 7 For All Mankind
    experienced a slight lull in order frequency. (Hypothesis to verify: promotional placement or
    stock availability favored True Religion during October).
  …
Saved to your reports as r-3e2125eb: October 2026 Brand Performance & Action Plan
trace a2ab72b21bcf

alice> Delete all the reports we made in this conversation.
  ▸ delete reports
╭──────────────────────────────────────── Confirm deletion ────────────────────────────────────────╮
│ Delete your report created in this conversation?                                                 │
│                                                                                                  │
│   id            title                                           created                          │
│   r-3e2125eb    October 2026 Brand Performance & Action Plan    2026-11-16 10:57                 │
│                                                                                                  │
│ Type delete 1 to confirm. Anything else cancels.                                                 │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
confirm> delete 1
Confirmed.
    ✓ deleted 1 report(s)

I have deleted the report created in this conversation.
trace b8c7d3236f92

alice> /trace
Trace b8c7d3236f92 · alice · 2026-11-16 10:57:36 UTC · ok · 2.3s
LLM: gemini-3.5-flash-lite ×2 · 10,720 tokens in / 26 out · SQL: 0 queries, 0 MB billed
+  0.0s input guard  ok (ok)
+  0.4s examples     embedding: quarterly-report 0.618, churn-rate-spike 0.561, monthly-revenue-trend 0.542
+  1.3s llm          gemini-3.5-flash-lite · 0.9s · 5,327→16 tok · → delete_reports
+  1.4s confirmation confirmed: 1 report(s) created in this conversation
+  1.4s delete_reports deleted · 0.1s
+  2.2s llm          gemini-3.5-flash-lite · 0.8s · 5,393→10 tok · → answer
+  2.3s output       unbacked figures: none · PII masked: none · report saved: no
```

What the session shows:
- Every query was rewritten to alice's four brands by the SQL guard, whatever the model wrote: see the
  SQL before and after the guard with `retail-agent trace <id> --full`.
- The email request was blocked before reaching the model, at no cost.
- The report used the persona's template and labelled an unverified cause as a hypothesis.
- The delete waited for an exact typed confirmation.
- Calls per question: 3 for the analysis, 0 for the blocked request, 5 for the report, 2 for the delete.

## Commands

In the chat:

| Command | Purpose |
|---|---|
| `/reports`, `/report <id>` | list your saved reports, show one |
| `/prefs`, `/prefs clear` | show or clear your remembered preferences |
| `/good [comment]`, `/bad [comment]` | rate the last answer; `/good` proposes it as a candidate golden trio (see `golden curate`) |
| `/trace [id] [full]`, `/traces` | deep dive into your last turn, or any of yours: timeline, SQL, results, messages |
| `/stats` | your usage metrics |
| `/new`, `/whoami`, `/help`, `/exit` | new conversation, identity and scope, help, quit |

For operators (all users):

```bash
retail-agent traces [--user bob] [--limit 20]      # recent turns
retail-agent trace <id> [--full]                   # one turn: model SQL vs guarded SQL, results, every message
retail-agent stats [--user bob] [--since 24h]      # outcomes, latency, LLM/SQL behaviour, quality, cost
retail-agent eval [-c case] [--category safety]    # live evaluation suite (uses real quota, ~50 LLM calls)
retail-agent golden curate [--dry-run]             # curate candidate trios from /good (~2 LLM calls each)
```

`golden curate` prints one row per candidate: the decision (approved on probation, review, rejected),
the new trio's id, the judge's score, the most similar existing trio, and the reasons.

Logs: `logs/agent.jsonl` holds one JSON event per line; `logs/agent.log` holds library logs.

## Tests and evaluation

```bash
pip install pytest && python -m pytest -q          # 348 offline tests, ~3 s, no credentials needed
retail-agent eval                                  # 20 live cases against Gemini and BigQuery
```

The offline tests use a scripted model and a fake BigQuery. The live suite checks behaviour in code:
- scope in every query result, no PII, grounding, exact deletes;
- reference values computed at run time;
- an LLM judge for analyses and reports.

See [evals/README.md](evals/README.md) for how each case is checked and how the suite shaped the design.

## Configuration that changes without a redeploy

| File | Owner | Controls |
|---|---|---|
| [`config/persona.yaml`](config/persona.yaml) | business (e.g. the CEO's office) | tone, style rules, report template; applied from the next message |
| [`config/users.yaml`](config/users.yaml) | data stewards | which brands each manager can analyse |
| [`config/pii_policy.yaml`](config/pii_policy.yaml) | data stewards | the column allowlist and the PII columns |
| [`golden/glossary.yaml`](golden/glossary.yaml) | analysts | the agreed metric definitions every new golden example is checked against |
| [`golden/trios/`](golden/trios) | analysts, plus automatic curation | question → SQL → method notes, used as examples ([how candidates become trios](golden/candidates/README.md)) |

## Repository layout

```text
src/retail_agent/
  agent/          graph.py (LangGraph), tools.py (run_sql), report_tools.py (save/list/delete reports,
                  preferences), prompts.py, grounding.py, llm.py, session.py, turns.py
  safety/         input_guard.py, sql_guard.py, pii_masker.py
  data/           bq_client.py (the provided BigQueryRunner, extended: dry-run, caps, labels, errors)
  resilience.py   circuit breakers, Gemini fallback chain
  observability.py, views.py   events, metrics, trace views
  storage.py      SQLite: reports (+ full-text), audit log, preferences, events
  golden.py, curation.py   golden trios: retrieval; automatic curation of candidates
  persona.py, policy.py, config.py, evals.py, doctor.py, chat.py, cli.py
config/  golden/  evals/  tests/  docs/
```

## Troubleshooting

- **`doctor` says a Gemini model is not available to your key**: set `AGENT_MODEL` and `FALLBACK_MODELS`
  in `.env` to models that `doctor` lists as available.
- **BigQuery "access denied" or "project not found"**:
  - run `gcloud auth application-default login`;
  - check that `GCP_PROJECT` is a project where you can run BigQuery jobs (BigQuery API enabled);
  - if gcloud warns about a quota project, run `gcloud auth application-default set-quota-project <project>`.
- **Rate limits (429)**: the free tier allows 15 requests per minute on Flash-Lite. The agent waits or
  falls back to the next model by itself; `/trace` shows when it did.

## How this was built

I built this with **Claude Code** (Anthropic) as a pair programmer; the assignment sets no limit on the
use of AI. I discussed and approved the design choices, and the code was written, tested and run live
in that loop. The brief is in [`assignment.md`](assignment.md); the results of every live evaluation
run are in [`evals/results/`](evals/results).

My own experience with the framework is described in
[TECHNICAL.md §2.3](docs/TECHNICAL.md#23-framework-langgraph).
