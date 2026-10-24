# Observed results — 24 October 2026

Pipeline: catalog-routed, judge-filtered evidence retrieval (see `docs/architecture.md`). The earlier A–F
ablation tables measured a different, retired pipeline and were removed.

## Indexing (training corpus, MacBook, Apple GPU for embeddings)

- 35/35 PDFs indexed, including `SAMA_EN_3575_VER1.pdf` (scanned) via PaddleOCR; 98 pages OCR'd in total.
- 36,029 paragraph units (60–220 words; gold chunks have a median of 63 words).
- Groq profiling: 33 LLM profiles, 2 rule-based fallbacks after free-tier rate limits; 48k input / 10k output tokens.
- Parse + OCR 735 s, total 1,526 s on a cold OCR cache. The OCR cache makes re-indexing the same PDFs much faster.

## Spot checks through the HTTP API (Groq free tier: gpt-oss-120b answers; gpt-oss-20b + qwen3.8-27b plan/judge)

| Question | Result vs gold | Evidence | Time | LLM tokens in/out |
|---|---|---|---|---|
| A34 BSF NPL coverage 2025 | 178.4% — correct; missed "down 3.3 pp YoY" | 1 | 51 s | 17.6k / 2.2k |
| A33 Refex rating agency | Acuité, ACUITE A- (Stable) reaffirmed — correct; gold could not read the rating | 4 | 17 s | 11.8k / 0.7k |
| A01 Gammon credit-risk red flags | Lender actions, NPA since 2017, ₹2,636.59 cr recall, ARCs, qualified opinion, going concern, suspension — correct; missed principal/interest default totals and SARFAESI possession | 27 (gold: 91) | 41 s | 24.4k / 3.9k |

These are three hand-checked questions, not the official metrics. The official notebook needs an OpenCode
key (`OPENCODE_API_KEY`) and has not been run. On Groq's free tier (about 8k tokens per minute per model)
the candidate and context budgets are deliberately small, which limits recall on broad questions.

## Official evaluator, OpenCode (24 October 2026)

Same three questions (A33, C30, A34), unchanged official notebook, answers by `deepseek-v4.1-flash`.

| Planner/judge model | Mean query time | Output tokens/query | Answer P | Answer R | Context P | Context R |
|---|---:|---:|---:|---:|---:|---:|
| deepseek-v4.1-flash (reasoning) | 268 s | 34,390 | 0.67 | 0.55 | 0.61 | 1.33 |
| minimax-m2.7 (no reasoning) | 21 s | 2,574 | 0.65 | 0.67 | 0.84 | 1.33 |

Three narrow questions only: not representative of the broad A/B/C questions. Context recall can exceed 1
because the official formula divides relevant returned chunks by the gold chunk count. On six broad
questions (A05, A20, B05, B20, C05, C20) the official evaluator's own deepseek calls failed
(`finish_reason=length` at its fixed 10,000-token limit, or truncated JSON) on 5 of 6, so no scores exist
for them yet.

## Six broad questions (A05, A20, B05, B20, C05, C20), 24 October 2026

Pipeline: minimax-m2.7 planner/judge, deepseek-v4.1-flash answers. All six answered in 30–114 s
(previously 4–7 minutes each).

**Official evaluator (deepseek-v4.1-flash judge): 0 of 6 scoreable.** Its own calls fail at the fixed
`max_tokens=10000` (`finish_reason=length` or truncated JSON) when splitting answers into points. A probe
on one of our answers decomposed 300/450/600 words but failed at 800; the gold answers for A05, A20 and
B20 (1,019–1,235 words) also fail to decompose, so those questions cannot be scored by that setup at all.

**Proxy: the same unchanged evaluator code with `OPENCODE_MODEL=minimax-m2.7` (configuration cell):**

| Question | Returned / gold chunks | Answer P | Answer R | Context P | Context R |
|---|---|---:|---:|---:|---:|
| A05 | 90 / 15 | 0.21 | 0.96 | 0.89 | 5.33 |
| A20 | 47 / 60 | 0.61 | 0.28 | 0.79 | 0.62 |
| B05 | 71 / 30 | 0.40 | 0.80 | 0.17 | 0.40 |
| B20 | 97 / 82 | 0.59 | 0.40 | 0.13 | 0.16 |
| C05 | 106 / 62 | 0.00 | 0.67 | 0.42 | 0.71 |
| C20 | 83 / 120 | 0.58 | 0.97 | 0.36 | 0.25 |
| **Mean** | | **0.40** | **0.68** | **0.46** | **1.24** (0.43 without A05) |

Proxy scores are not comparable to the official judge; the minimax judge is also inconsistent (C05: 0.00
precision yet 0.67 recall).

## Precision changes: grade-3-only answer evidence, line-deletion editor, ~350-word cap (24 October 2026)

**Official judge (deepseek-v4.1-flash), 3 narrow questions:**

| | Answer P | Answer R | Context P | Context R |
|---|---:|---:|---:|---:|
| Before | 0.65 | 0.67 | 0.84 | 1.33 |
| After | **0.92** | 0.54 | 0.63 | 1.21 |

Per question after (P/R/CP/CR): A34 1.00/0.33/0.50/1.00, A33 0.75/0.44/1.00/2.00, C30 1.00/0.83/0.38/0.62.
Context changes come from judge run-to-run variance (retrieval and judging were not changed).

**Six broad questions, proxy judge (minimax-m2.7):** the proxy's answer precision is invalid (it lists
every answer point as "extra" while matching them to gold points), so only recall and context are shown.

| | Answer R | Context P | Context R (excl. A05) |
|---|---:|---:|---:|
| Before | 0.68 | 0.46 | 0.43 |
| After | 0.50 | **0.58** | **0.58** |

Broad answers (444–575 words) still failed the official evaluator's own length limit.

## Recall recovery: aspect-linked grade-2 evidence, answer outline, figure qualifiers, "not stated" lines

**Official judge, 3 narrow questions (A33, A34, C30):**

| Version | Answer P | Answer R | Context P | Context R |
|---|---:|---:|---:|---:|
| v1 fast judge | 0.65 | 0.67 | 0.84 | 1.33 |
| v2 precision changes | 0.92 | 0.54 | 0.63 | 1.21 |
| v3 recall recovery | 0.73 | 0.65 | 0.87 | 1.25 |

**Proxy judge, 6 broad questions (answer precision omitted: proxy invalid):**

| Version | Answer R | Context P | Context R (excl. A05) |
|---|---:|---:|---:|
| v1 | 0.68 | 0.46 | 0.43 |
| v2 | 0.50 | 0.58 | 0.58 |
| v3 | 0.62 | 0.61 | 0.51 |

Notes: the official judge decomposes citations into points, so page numbers matter; 798 of ~840 gold chunks
match our physical page numbers (mismatches are repeated text). A33's gold says the rating table is an
unreadable image while our OCR reads it, which costs 3 extra and 3 missing points. Broad v3 answers grew
to 432–849 words and still fail the official evaluator's length limit.

## Failure buckets, 54 of 100 questions (deepseek answers, minimax judge; run stopped when OpenCode credit ran out)

`python -m evaluation.analyze` maps each gold chunk to our indexed paragraphs (no LLM calls) and traces it:

| | Questions | Gold chunks / question | Returned / question | Gold returned | Never retrieved | Judge rejected | Wrong document |
|---|---:|---:|---:|---:|---:|---:|---:|
| Narrow | 5 | 5 | 4 | 20% | 50% | 30% | 0% |
| Broad | 49 | 105 | 77 | 25% | **70%** | 4% | 2% |

Overall: 5% of gold chunks have no counterpart in our index (extraction gaps), routing recall is 0.99,
and 24% of returned chunks match a gold chunk lexically. Retrieval recall on broad questions is the
dominant loss: gold sets average 105 chunks (up to 352) spread over many pages, while candidate search
is capped and chunk-level. Full table: `reports/runs/full100/failure-analysis.md` (local).

The pipeline now runs on Groq (gpt-oss-120b answers; gpt-oss-20b + qwen3.8-27b plan/judge) with
free-tier budgets in `.env`. Official scoring needs OpenCode credit; Groq's ~8k-token request limit is too
small for the evaluator's context-judge call on broad questions.

## Proxy scores with a Groq judge (10 saved answers, 24 October 2026)

Unchanged evaluator code with the configuration cell pointed at Groq `openai/gpt-oss-120b`; retrieved
chunks judged in batches of 20 by the unchanged `context_precision_recall` (Groq rejects one 59-chunk
request with HTTP 413) and summed; an HTTP-level wait on Groq 429s. Answers are the saved v3 deepseek
answers. Proxy only: a different judge model from the official deepseek-v4.1-flash.

| Group | n | Answer P | Answer R | Context P | Context R |
|---|---:|---:|---:|---:|---:|
| All | 10 | 0.71 | 0.41 | 0.68 | 0.74 |
| Narrow (A09, A33, A34, A35, A36) | 5 | 0.73 | 0.32 | 0.76 | 1.07 |
| Broad (A07, A16, B05, B08, B19) | 5 | 0.68 | 0.49 | 0.61 | 0.41 |

Answer points: 93 gold facts missing, 18 gold "not stated" points missing, 36 extra facts, 6 extra
citation points. A36 asks about a 2025 report when only the 2024 report exists: the judge rejected every
passage as the wrong period, while the gold answer uses the 2024 report and notes the mismatch.

## Context metrics as the API contract scores them (page overlap, no LLM)

The organizers' API contract scores context by page overlap: a chunk is relevant when its file and pages
overlap a benchmark passage; recall is the share of benchmark (document, page) pairs covered. Exact
values on the 54 saved answers (deepseek answers, minimax judge), by which chunks are returned:

| Chunks returned | Chunks / question | Context P | Context R | F1 |
|---|---:|---:|---:|---:|
| All judged evidence (default) | 70 | 0.45 | 0.43 | 0.44 |
| One chunk per page | 39 | 0.42 | 0.43 | 0.43 |
| Grade-3 only | 28 | 0.51 | 0.29 | 0.37 |
| Cited in the answer only | 20 | 0.51 | 0.23 | 0.32 |

`evaluation.run` now reports these as `contract_context_precision` / `contract_context_recall` for every run.
