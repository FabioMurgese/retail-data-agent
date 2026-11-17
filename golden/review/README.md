# Review queue

Candidates that curation could not settle on its own. Each file is the **generalised draft**, ready to
promote: the reasons are in its header, and the original candidate is kept at the bottom as comments.

To approve: fix what the reasons point at (for example, add a missing metric to
[`golden/glossary.yaml`](../glossary.yaml), or remove a follow-up that needs data we do not have), keep
`status: approved`, and move the file to `golden/trios/`. `retail-agent doctor` checks that its SQL
passes the SQL guard for every user. To reject: delete the file.

[`2026-11-16-f5ec1bb1ffe5.yaml`](2026-11-16-f5ec1bb1ffe5.yaml) is a real example from testing: dan's
delivery-time question. The glossary does not define delivery time (a deterministic check), and the
judge scored it 3/5 because the notes suggest an analysis by carrier, data the dataset does not have.
Other runtime files here are git-ignored.
