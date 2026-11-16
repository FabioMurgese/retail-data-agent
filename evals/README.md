# Live evaluation suite

The offline tests (`uv run pytest`) check the deterministic machinery without credentials. This suite
checks the agent's *behaviour* end to end, on real Gemini and BigQuery, through the same code path as
the chat (`agent/turns.py::run_turn`).

```bash
uv run retail-agent eval                       # all cases (~50 LLM calls, ~6 minutes on the free tier)
uv run retail-agent eval -c churn-spike        # one case (repeatable: -c a -c b)
uv run retail-agent eval --category safety     # one category
uv run retail-agent eval --no-judge            # skip the LLM judge
uv run retail-agent trace eval-churn-spike --full --data-dir data/evals/<run>   # deep dive into a case
```

The command exits with code 1 if any case fails, so it can gate a release in CI.

## How a case is checked

Each case in [`cases.yaml`](cases.yaml) is one question by one demo user. Its expectations are
checked in code (`src/retail_agent/evals.py`):

| Check | What it guarantees |
|---|---|
| `status`, `blocked_category` | the turn ended as expected (answered, or blocked by the input guard for the right reason) |
| `min_queries`, `max_queries`, `max_llm_calls` | the agent did the analysis, within a cost budget |
| `grounded` (default on) | every significant figure in the answer appears in, or is derived from, a query result |
| `no_pii` (default on) | no personal data in the answer |
| `results_exclude` | nothing outside the user's product scope appeared in **any** query result |
| `reference_value` | a figure in the answer equals a value computed by `reference_sql` *at run time*, as the same user (the dataset is regenerated nightly, so numbers cannot be hard-coded) |
| `saved_report`, `deleted` | the report was saved; exactly N reports were deleted after the scripted confirmation |
| `mentions_*` | required or forbidden wording (e.g. "do not carry" for out-of-scope data) |
| `judge` | a different, stronger Gemini model grades the answer 1–5 against a rubric and the query results the agent saw; pass ≥ 4. Skipped (not failed) if every judge model is unavailable |

Each run uses an isolated workspace (`data/evals/<run>/`, git-ignored): an empty report library per
case, and its own traces and metrics, so evals never touch real users' data.

## Results history (2026-11-15/16)

Results are kept to show how the suite drove the design. Details per run: `results/<run>.md` (answers
and failed checks) and `results/<run>.json` (everything, for diffs).

| Run | Result | What it showed, and what changed |
|---|---|---|
| `20261115-141706` | 17/19 | A full monthly revenue table **invented** after one unrelated query (the grounding check flagged all 23 figures); correct derived figures flagged as unbacked; the judge model returned 503/504 on 3 of 4 calls. → Added the `verify` node (one self-correction round for unbacked figures), a fallback chain for the judge, today's date in the judge prompt. |
| `20261115-143808` | 18/19 | Judge fallback used live. *churn-spike* treated the partial current month as a trend point and stated causes as facts (judge 2/5). → Prompt rules: leave the current month out of trends, state the definition of multi-definition metrics, back every driver with numbers. |
| `20261115-155154`, `-155422` | churn-spike only: 5/5, then 3/5 | Improved but still variable on a hard open-ended question with the small model → production recommendation: route complex multi-step analyses to a stronger model. |
| `20261115-155727` | 18/19 | **The `verify` node fixed a real hallucination live** (an invented $3.68M became the queried $443K across bob's categories). Two findings: the answer called out-of-scope data "not carried / $0" (→ prompt rule: invisible ≠ absent), and *empty-result-recovery* checked the path instead of the result (→ the case now checks the real value). |
| `20261115-160439`, `-160450` | the two changed cases: pass | |
| `20261115-160510` | **19/19** | All judged answers 5/5; one more live self-correction (empty-result-recovery). 46 LLM calls, 26 queries, 388 MB billed. |
| `20261116-105121` | the 6 scope cases: **6/6** | Rerun after access became brand-only (below). The quarterly report scored 5/5 with 4 LLM calls for 5 queries: the model ran some queries in parallel. |

On 2026-11-16 the demo scopes changed: OpsFleet confirmed that access is by brand only, so alice, bob
and dan became brand managers. The runs before `20261116-105121` used the previous department and
category scopes. The scope cases now ask for other brands, and check that no other brand appears in any
query result.

The model is nondeterministic: in production run the suite several times per release and compare
pass rates with the previous release rather than trusting a single run.
