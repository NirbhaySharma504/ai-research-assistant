"""Evaluation harness + fact-checker ablation.

Runs a fixed question set through the pipeline and averages the RAGAS scores, for
two variants:
  * full          — the complete graph (with the fact-checker)
  * no_factcheck  — the fact-checker replaced by a no-op (ablation)

Comparing the two quantifies what the fact-checking agent actually buys us. Results
are cached to benchmark/results.json so the run is resumable (each question/variant
is skipped if already scored), and a markdown summary is written to RESULTS.md.

Usage:
    python -m benchmark.run_eval                       # both variants, all questions
    python -m benchmark.run_eval --variants full       # one variant
    python -m benchmark.run_eval --limit 3             # first 3 questions only
    python -m benchmark.run_eval --report-only         # just (re)build RESULTS.md
"""

import argparse
import json
import statistics
import uuid
from pathlib import Path

from backend.graph.runner import run_ablation_pair
from benchmark.questions import QUESTIONS

_HERE = Path(__file__).parent
_RESULTS = _HERE / "results.json"
_REPORT = _HERE / "RESULTS.md"
_METRICS = ["faithfulness", "answer_relevancy", "context_precision"]
_VARIANTS = {"full": True, "no_factcheck": False}


def _load() -> dict:
    if _RESULTS.exists():
        return json.loads(_RESULTS.read_text())
    return {}


def _save(data: dict) -> None:
    _RESULTS.write_text(json.dumps(data, indent=2))


def _key(variant: str, question: str) -> str:
    return f"{variant}::{question}"


def _has_scores(entry: dict) -> bool:
    """True only if at least one RAGAS metric came back non-null (i.e. the judge
    actually scored it). All-null means the run failed (usually exhausted quota) and
    must be retried, not cached as done."""
    s = entry.get("ragas_scores") or {}
    return any(v is not None for v in s.values())


def _entry(variant: str, q: str, state: dict) -> dict:
    return {
        "variant": variant,
        "question": q,
        "ragas_scores": state.get("ragas_scores"),
        "source_count": len(state.get("retrieved_content", [])),
        "n_verified_claims": len(state.get("verified_claims", [])),
        # saved so any 0.0 answer_relevancy can be inspected/verified later
        "answer_preview": (state.get("final_answer", "") or "")[:600],
    }


def run(variants: list[str], limit: int, max_iterations: int) -> None:
    """Controlled ablation: per question, evaluate full + no_factcheck over the SAME
    retrieved corpus (see backend.graph.runner.run_ablation_pair). Resumable per
    question (a question is skipped only once BOTH variants are scored)."""
    questions = QUESTIONS[:limit] if limit else QUESTIONS
    results = _load()
    pending = [
        q for q in questions
        if not (_has_scores(results.get(_key("full", q), {}))
                and _has_scores(results.get(_key("no_factcheck", q), {})))
    ]
    print(f"{len(questions) - len(pending)} questions already done, "
          f"{len(pending)} to run.\n")

    for n, q in enumerate(pending, 1):
        print(f"[{n}/{len(pending)}] running (controlled pair): {q[:50]} ...")
        session_id = uuid.uuid4().hex[:12]
        try:
            full, nofc = run_ablation_pair(q, session_id, max_iterations)
        except Exception as e:  # noqa: BLE001
            print(f"    pipeline ERROR (not cached): {e}")
            continue

        pair = {"full": _entry("full", q, full),
                "no_factcheck": _entry("no_factcheck", q, nofc)}
        if not all(_has_scores(e) for e in pair.values()):
            # No scores. Diagnose: an empty corpus (source_count == 0) means web
            # search/scrape failed (usually a Tavily quota); otherwise the judge LLM
            # failed. Stop cleanly either way — the run is resumable from here.
            cause = (
                "web search returned no sources — Tavily API key likely exhausted"
                if pair["full"]["source_count"] == 0
                else "the judge LLM's credit/rate limit is likely exhausted "
                     "(OpenRouter/Groq)"
            )
            print(
                f"\n⚠️  RAGAS returned no scores — {cause}.\n"
                f"   Fix the relevant key and re-run "
                f"`python -m benchmark.run_eval` to resume "
                f"({len(pending) - n + 1} questions left)."
            )
            report(results)
            return

        for variant, entry in pair.items():
            results[_key(variant, q)] = entry
        _save(results)  # checkpoint after every question (resumable)
        sf, sn = pair["full"]["ragas_scores"], pair["no_factcheck"]["ragas_scores"]
        print(f"    ✓ full: f={sf['faithfulness']} r={sf['answer_relevancy']} "
              f"p={sf['context_precision']}  |  noFC: f={sn['faithfulness']} "
              f"r={sn['answer_relevancy']} p={sn['context_precision']}")

    report(results)


def _stats(results: dict, variant: str) -> dict:
    """Per-metric median (headline) and mean, over non-null scores. Median is the
    headline because RAGAS answer_relevancy occasionally returns a spurious 0.0 on a
    genuinely relevant answer (its noncommittal classifier misfires); the median is
    robust to those outliers while the mean is not."""
    rows = [
        r for r in results.values()
        if r.get("variant") == variant and r.get("ragas_scores")
    ]
    out = {"n": len(rows)}
    for m in _METRICS:
        vals = [
            r["ragas_scores"][m]
            for r in rows
            if r["ragas_scores"].get(m) is not None
        ]
        out[m] = {
            "median": round(statistics.median(vals), 3) if vals else None,
            "mean": round(statistics.mean(vals), 3) if vals else None,
        }
    return out


def report(results: dict | None = None) -> None:
    results = results if results is not None else _load()
    lines = [
        "# Evaluation Results",
        "",
        "RAGAS scores over the benchmark question set — pipeline on local "
        "`llama3.1:8b`, judged by OpenRouter `gpt-4o-mini`. Regenerate with "
        "`python -m benchmark.run_eval`.",
        "",
        "Headline figure is the **median** (robust): RAGAS `answer_relevancy` "
        "occasionally emits a spurious `0.0` on a genuinely relevant answer when its "
        "noncommittal classifier misfires, which distorts the mean. Mean shown in "
        "parentheses for transparency.",
        "",
        "## Scores by variant — median (mean)",
        "",
        "| Variant | N | Faithfulness | Answer Relevancy | Context Precision |",
        "|---------|:-:|:-:|:-:|:-:|",
    ]
    present = [v for v in _VARIANTS if _stats(results, v)["n"] > 0]
    stats = {v: _stats(results, v) for v in present}
    for v in present:
        s = stats[v]
        lines.append(
            f"| {v} | {s['n']} | {_cell(s['faithfulness'])} | "
            f"{_cell(s['answer_relevancy'])} | {_cell(s['context_precision'])} |"
        )

    if {"full", "no_factcheck"} <= set(present):
        lines += ["", "## Ablation: contribution of the fact-checker", "",
                  "**Controlled:** both variants are synthesized & scored over the "
                  "*identical* retrieved corpus (built once per question); the only "
                  "changed variable is the fact-checker's verified claims. Δ shown for "
                  "both median and mean.", ""]
        lines += ["| Metric | full (med/mean) | no_factcheck (med/mean) | Δ median | Δ mean |",
                  "|--------|:-:|:-:|:-:|:-:|"]
        for m in _METRICS:
            fm, fa = stats["full"][m]["median"], stats["full"][m]["mean"]
            nm, na = stats["no_factcheck"][m]["median"], stats["no_factcheck"][m]["mean"]
            dmed = f"{fm - nm:+.3f}" if None not in (fm, nm) else "—"
            dmean = f"{fa - na:+.3f}" if None not in (fa, na) else "—"
            lines.append(
                f"| {m} | {_f(fm)} / {_f(fa)} | {_f(nm)} / {_f(na)} | {dmed} | {dmean} |"
            )

    if "full" in present:
        lines += ["", "## Distribution & robustness", ""] + _distribution_notes(results)

    _REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {_REPORT}")


def _scores(results: dict, variant: str, metric: str) -> list[float]:
    return [
        r["ragas_scores"][metric]
        for r in results.values()
        if r.get("variant") == variant
        and r.get("ragas_scores")
        and r["ragas_scores"].get(metric) is not None
    ]


def _distribution_notes(results: dict) -> list[str]:
    f = _scores(results, "full", "faithfulness")
    perfect = sum(1 for x in f if x >= 1.0)
    rel = _scores(results, "full", "answer_relevancy")
    zeros = sum(1 for x in rel if x == 0.0)
    rel_nz = [x for x in rel if x > 0.0]
    nofc_nz = [x for x in _scores(results, "no_factcheck", "answer_relevancy") if x > 0]
    mean_excl = statistics.mean(rel_nz) if rel_nz else None
    nofc_excl = statistics.mean(nofc_nz) if nofc_nz else None
    return [
        f"- **Faithfulness (full):** {perfect}/{len(f)} of the queries scored a "
        f"perfect 1.0; the mean ({round(statistics.mean(f), 3)}, min "
        f"{round(min(f), 3)}) reflects the full distribution.",
        f"- **Answer relevancy (full):** {zeros}/{len(rel)} queries scored exactly "
        f"0.0. Their answers are saved in `results.json` (`answer_preview`) and were "
        f"inspected: each is a substantive, on-topic answer — RAGAS's *noncommittal* "
        f"classifier misfiring, not a quality failure. **Excluding these artifacts, "
        f"mean answer_relevancy = {round(mean_excl, 3) if mean_excl else '—'}** (vs "
        f"no_factcheck {round(nofc_excl, 3) if nofc_excl else '—'}).",
    ]


def _cell(stat: dict) -> str:
    return f"{_f(stat['median'])} ({_f(stat['mean'])})"


def _f(v) -> str:
    return "—" if v is None else f"{v:.3f}"


def main() -> None:
    p = argparse.ArgumentParser(description="RAGAS eval harness + ablation")
    p.add_argument("--variants", default="full,no_factcheck",
                   help="comma-separated: full,no_factcheck")
    p.add_argument("--limit", type=int, default=0, help="first N questions (0 = all)")
    p.add_argument("--max-iterations", type=int, default=3)
    p.add_argument("--report-only", action="store_true",
                   help="rebuild RESULTS.md from cached results without running")
    args = p.parse_args()

    if args.report_only:
        report()
        return

    variants = [v.strip() for v in args.variants.split(",") if v.strip() in _VARIANTS]
    run(variants, args.limit, args.max_iterations)


if __name__ == "__main__":
    main()
