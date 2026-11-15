# Candidate trios

When a user rates an answer with `/good` in the chat, and the answer was backed by successful
queries, the assistant writes a *candidate* trio here: the question, the SQL it ran and its answer.

Candidates are **never used** by the assistant. An analyst reviews each one:

1. check the SQL and the business definitions (net revenue, periods, filters);
2. generalise the question and add `tags` for retrieval;
3. rewrite `report` as method notes — how to read the result, caveats, follow-ups — without current
   numbers (the dataset is regenerated nightly);
4. move the file to `golden/trios/` and run `retail-agent doctor`, which checks that its SQL passes the
   SQL guard for every user.

A `# REVIEW WARNING` line in the header means the automatic grounding check could not verify some
figures of the original answer.

`2026-11-15-5fc68943610b.yaml` is a real candidate produced while testing (user `bob`, scoped to the
Jeans, Pants and Pants & Capris categories). Runtime candidates are git-ignored; in production this
folder becomes a review queue (see docs/HLD.md).
