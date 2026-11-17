# Rejected candidates

Candidates that curation rejected, each with the reasons in its header: SQL the guard refuses for some
user, SQL that singles out customers or orders, personal data or instructions in the text, a duplicate
of an existing trio, or a judge score of 2/5 or less. Unsafe candidates are rejected before any LLM
call is spent on them.

Rejected files keep the original answer, which can describe individual (pseudonymous) customers: in
production this folder has a short retention and restricted access.

[`2026-11-16-bd1b988fdd70.yaml`](2026-11-16-bd1b988fdd70.yaml) is a real example from testing: dan's
return-rate question, a duplicate of the `return-rate-by-category` trio (similarity 0.96). Other runtime
files here are git-ignored.
