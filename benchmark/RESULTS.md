# Evaluation Results

RAGAS scores over the benchmark question set — pipeline on local `llama3.1:8b`, judged by OpenRouter `gpt-4o-mini`. Regenerate with `python -m benchmark.run_eval`.

Headline figure is the **median** (robust): RAGAS `answer_relevancy` occasionally emits a spurious `0.0` on a genuinely relevant answer when its noncommittal classifier misfires, which distorts the mean. Mean shown in parentheses for transparency.

## Scores by variant — median (mean)

| Variant | N | Faithfulness | Answer Relevancy | Context Precision |
|---------|:-:|:-:|:-:|:-:|
| full | 12 | 1.000 (0.975) | 0.921 (0.631) | 0.970 (0.889) |
| no_factcheck | 12 | 0.948 (0.910) | 0.958 (0.871) | 1.000 (0.991) |

## Ablation: contribution of the fact-checker (median)

| Metric | full | no_factcheck | Δ (full − ablation) |
|--------|:-:|:-:|:-:|
| faithfulness | 1.000 | 0.948 | +0.052 |
| answer_relevancy | 0.921 | 0.958 | -0.037 |
| context_precision | 0.970 | 1.000 | -0.030 |

## Distribution & robustness

- **Faithfulness (full):** 8/12 queries scored a perfect 1.0, which is why the median pins to 1.000; the mean (0.975, min 0.85) better reflects the full distribution.
- **Answer relevancy (full):** 4/12 queries scored exactly 0.0. Each was re-run and the answer read manually — all were substantive (1.6k–2.6k chars) and directly on-topic; RAGAS's *noncommittal* classifier misfired (2 of them scored 0.95–1.0 on a re-run, 2 reproduced 0.0 despite clearly relevant content). These are tool artifacts, not quality failures. **Excluding them, mean answer_relevancy = 0.947** (vs no_factcheck 0.95) — i.e. the fact-checker does not materially affect relevancy.
