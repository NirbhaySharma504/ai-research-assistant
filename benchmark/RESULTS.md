# Evaluation Results

RAGAS scores over the benchmark question set — pipeline on local `llama3.1:8b`, judged by OpenRouter `gpt-4o-mini`. Regenerate with `python -m benchmark.run_eval`.

Headline figure is the **median** (robust): RAGAS `answer_relevancy` occasionally emits a spurious `0.0` on a genuinely relevant answer when its noncommittal classifier misfires, which distorts the mean. Mean shown in parentheses for transparency.

## Scores by variant — median (mean)

| Variant | N | Faithfulness | Answer Relevancy | Context Precision |
|---------|:-:|:-:|:-:|:-:|
| full | 12 | 0.978 (0.936) | 0.957 (0.774) | 0.974 (0.964) |
| no_factcheck | 12 | 0.976 (0.908) | 0.834 (0.604) | 0.982 (0.937) |

## Ablation: contribution of the fact-checker

**Controlled:** both variants are synthesized & scored over the *identical* retrieved corpus (built once per question); the only changed variable is the fact-checker's verified claims. Δ shown for both median and mean.

| Metric | full (med/mean) | no_factcheck (med/mean) | Δ median | Δ mean |
|--------|:-:|:-:|:-:|:-:|
| faithfulness | 0.978 / 0.936 | 0.976 / 0.908 | +0.002 | +0.028 |
| answer_relevancy | 0.957 / 0.774 | 0.834 / 0.604 | +0.123 | +0.170 |
| context_precision | 0.974 / 0.964 | 0.982 / 0.937 | -0.008 | +0.027 |

## Distribution & robustness

- **Faithfulness (full):** 6/12 of the queries scored a perfect 1.0; the mean (0.936, min 0.733) reflects the full distribution.
- **Answer relevancy (full):** 2/12 queries scored exactly 0.0. Their answers are saved in `results.json` (`answer_preview`) and were inspected: each is a substantive, on-topic answer — RAGAS's *noncommittal* classifier misfiring, not a quality failure. **Excluding these artifacts, mean answer_relevancy = 0.929** (vs no_factcheck 0.906).
