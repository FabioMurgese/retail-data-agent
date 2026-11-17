# Candidate trios

When a user rates an answer with `/good` in the chat, and the answer was backed by successful
queries, the assistant writes a *candidate* here: the question, the SQL it ran (`sql` and
`other_queries`), its answer (`report`), the trace id, and any figures the grounding check could not
verify (`warnings`).

Candidates are **never used** by the assistant. `retail-agent golden curate` processes them, and each
one leaves this folder for exactly one destination:

| Outcome | Where | When |
|---|---|---|
| approved, on probation | `golden/trios/` | safe for every user, new, consistent with the [glossary](../glossary.yaml), judged at least 4/5 |
| needs an analyst | [`golden/review/`](../review/README.md) | a doubt only a person can settle: a metric not in the glossary, a variant of an existing trio, a mid score, a model outage… |
| rejected | [`golden/rejected/`](../rejected/README.md) | unsafe SQL, personal data, instructions hidden in the text, a duplicate, a low score |

How the decision is made, and why human review is kept for exceptions only:
[docs/TECHNICAL.md §4.1](../../docs/TECHNICAL.md#41-hybrid-intelligence-golden-bucket).
Runtime candidates are git-ignored.
