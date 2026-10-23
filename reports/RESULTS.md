# Observed results — 24 October 2026

All six experiments completed all 100 supplied questions with no retrieval failures. This was semantic MiniLM retrieval with extractive previews: no answer-model or official-judge calls. The four official metrics are **not measured** and are null in the exported JSON/CSV.

| Experiment | Page recall proxy | Delta vs A | Page precision proxy | Lexical Recall@6 proxy | Mean warm query ms | Mean context estimate |
|---|---:|---:|---:|---:|---:|---:|
| A: Baseline | 0.0596 | +0.0000 | 0.2543 | 0.0785 | 4.64 | 2054 |
| B: Metadata routing | 0.0584 | -0.0012 | 0.2563 | 0.0892 | 6.25 | 1911 |
| C: Compact + BM25 | 0.0777 | +0.0182 | 0.3123 | 0.1107 | 8.40 | 2186 |
| D: Multi-section | 0.1279 | +0.0684 | 0.1905 | 0.0800 | 145.78 | 7115 |
| E: Coverage selection | 0.0344 | -0.0251 | 0.2027 | 0.0529 | 40.75 | 1354 |
| F: All improvements | 0.1388 | +0.0792 | 0.2297 | 0.0655 | 171.75 | 6361 |

Page proxies compare physical (document, page) pairs with benchmark page numbers; printed-page offsets can lower these diagnostics. Lexical Recall@6 requires at least 60% of the gold chunk's distinct lexical terms in one of the first six returned passages from the same document. These are not the official LLM-judged context metrics. The first six passages in D/E/F are a coverage-oriented ordering, not a pure relevance ranking.

A/B/C/E allow six final passages; D/F allow 24 for broad questions. All use a 10,000 lexical-token context budget. Larger retrieval budgets explain part of the higher page recall in D/F. Model, index, query-embedding and text-feature caches were warmed before measurement; the recorded warmup was 17.63 seconds. These millisecond values exclude model loading, cold query embedding and remote answer generation, and are not production end-to-end latency.

C improved both page precision and recall proxies over A. F raised page recall but lowered page precision and first-six lexical coverage, increased context by roughly 3.1 times, and increased warm retrieval latency. E reduced context at the expense of recall. B had mixed results: slightly lower page recall, slightly higher page precision, and higher lexical coverage. Consequently the data does not establish that all four improvements together are the best configuration, nor that fewer embedding tokens improve answer quality.

## Indexing observations

- 35 input PDFs; 34 readable PDFs; 13,279 chunks. `SAMA_EN_3575_VER1.pdf` requires OCR and was explicitly skipped. Partially empty pages are recorded in document metadata.
- Full-chunk embedding inputs: 4,518,945 tokenizer-counted tokens.
- Compact embedding inputs: 1,843,757 tokenizer-counted tokens; **59.20% fewer logical input tokens**.
- Both representations were deliberately built for ablations. The first build computed 6,201,767 embedding tokens; the final metadata rebuild computed another 495,400 tokens for changed sketches and reused all 13,279 full-chunk vectors. These are actual shared-cache run counts, not independent cold full/compact runs.
- In the final rebuild, compact indexing reused 9,829 cached/duplicate inputs. Comparing only the final 495,400 computed tokens with a cold baseline would overstate savings; the 59.20% figure instead compares the complete logical input sizes.
- The first index build measured 297.73 seconds and the cached metadata rebuild measured 193.26 seconds up to index assembly; initial model download and final persistence were not included in those measurements. Persisted data is available locally under `data/collections/train/`.
- Sketch/planning LLM tokens: zero. Local embedding API charges: zero. Hardware cost is not estimated. Real answer/judge token usage, total deployment cost, and hosted Langfuse delivery remain unmeasured.

## Reproduce and next experiments

See the README for exact indexing and evaluation commands. The checked-in `comparison.json`, `comparison.csv`, and `runtime.json` preserve configuration, benchmark/evaluator hashes, budgets and installed versions. Full responses and per-question details remain local under `reports/runs/semantic-100-final/`.

Next: supply OpenCode/Langfuse credentials and run the unchanged official evaluator; compare C/F against a baseline with the same larger context budget; test routing confidence on held-out documents; add OCR and stronger table extraction; revise coverage weighting with a separate validation split. Do not infer the PS's 0.95 target from these proxy measurements.
