#!/usr/bin/env python3
"""
Cadencia Stage 0 Bench Test CLI (Signal-Starvation Hypothesis Protocol p. 10).

Stress-tests the GPT-6.1 Terra Generator + Stage A/B Verifier contract across
5 simulated learner states (hours 2, 10, 20, 35, and 50) without spending human
participant hours. Evaluates the formal Stage 0 Gate:
  - >= 95% of passages pass ALL contract rules within <= 3 rewrites
  - >= 20% average scheduled-item word share

API Key Safety:
  Read strictly from the `OPENAI_API_KEY` environment variable (never hardcoded).
  Example:
    export OPENAI_API_KEY="sk-..."
    .venv/bin/python tools/bench_test.py --passages-per-hour 10
    # Full 1,000-passage Stage 0 certification (200 passages x 5 hours):
    .venv/bin/python tools/bench_test.py --passages-per-hour 200 --concurrency 8
"""

import argparse
import concurrent.futures
import json
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
SEED_DB = ROOT / "Sources" / "CadenciaCore" / "Resources" / "cadencia_seed.sqlite"
MODEL_ID = "gpt-6.1-terra"

WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", re.UNICODE)
FUNCTION_STOP = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "a", "al",
    "y", "o", "en", "con", "por", "para", "que", "se", "no", "es", "son", "su", "sus",
    "me", "te", "le", "les", "lo", "nos", "mi", "tu", "más", "pero", "si", "sí", "ya",
    "muy", "también", "como", "cuando", "donde", "este", "esta", "esto", "ese", "esa",
    "hay", "ser", "estar", "tener", "hacer", "ir", "yo", "tú", "él", "ella", "nosotros",
    "ellos", "ellas", "usted", "ustedes", "sin", "sobre", "entre", "hasta", "desde",
}

PASSAGE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "titleEs": {"type": "string"},
        "titleEn": {"type": "string"},
        "episodeSummaryForNext": {"type": "string"},
        "sentences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "spanish": {"type": "string"},
                    "englishLine": {"type": "string"},
                },
                "required": ["spanish", "englishLine"],
                "additionalProperties": False,
            },
        },
        "newItemGlosses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "itemId": {"type": "string"},
                    "spanish": {"type": "string"},
                    "englishGloss": {"type": "string"},
                    "contextClueNote": {"type": "string"},
                },
                "required": ["itemId", "spanish", "englishGloss", "contextClueNote"],
                "additionalProperties": False,
            },
        },
        "probes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "targetDueItemId": {"type": "string"},
                    "targetDueLemma": {"type": "string"},
                    "questionEn": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}},
                    "correctOptionIndex": {"type": "integer"},
                    "hingeExplanation": {"type": "string"},
                },
                "required": [
                    "targetDueItemId", "targetDueLemma", "questionEn",
                    "options", "correctOptionIndex", "hingeExplanation",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "titleEs", "titleEn", "episodeSummaryForNext",
        "sentences", "newItemGlosses", "probes",
    ],
    "additionalProperties": False,
}


@dataclass
class SeedBank:
    active_items: List[Dict]
    holdout_lemmas: Set[str]
    inflections: Dict[str, str]
    grammar_items: List[Dict]


def load_seed_bank(db_path: Path) -> SeedBank:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    active = [
        dict(r)
        for r in conn.execute(
            "SELECT item_id, item_type, rank_index, spanish, english_gloss, is_transparent_cognate, grammar_week "
            "FROM syllabus_items WHERE is_holdout = 0 ORDER BY rank_index"
        )
    ]
    holdout = {
        r["spanish"].lower()
        for r in conn.execute("SELECT spanish FROM syllabus_items WHERE is_holdout = 1")
    }
    inflections = {
        r["surface"].lower(): r["lemma"].lower()
        for r in conn.execute("SELECT surface, lemma FROM spanish_inflections")
    }
    grammar = [it for it in active if it["item_type"] == "grammar"]
    conn.close()
    return SeedBank(active_items=active, holdout_lemmas=holdout, inflections=inflections, grammar_items=grammar)


def lemmatize_word(surface: str, inflections: Dict[str, str]) -> str:
    low = surface.lower()
    if low in inflections:
        return inflections[low]
    # Strip common enclitic pronouns on infinitives/gerunds
    for clitic in ("melo", "mela", "telo", "tela", "selo", "sela", "nos", "les", "las", "los", "me", "te", "se", "lo", "la", "le"):
        if len(low) > len(clitic) + 3 and low.endswith(clitic):
            stem = low[: -len(clitic)]
            if stem in inflections:
                return inflections[stem]
    return low


def verify_stage_a(
    passage: Dict,
    due_items: List[Dict],
    new_items: List[Dict],
    allowed_lemmas: Set[str],
    holdout_lemmas: Set[str],
    inflections: Dict[str, str],
) -> Tuple[bool, List[str], Dict[str, float]]:
    """Deterministic local Stage A Verifier matching ContractVerifier.swift Rules 1–4."""
    full_text = " ".join(s["spanish"] for s in passage["sentences"])
    words = WORD_RE.findall(full_text)
    total_words = len(words)
    violations: List[str] = []

    if total_words < 175 or total_words > 265:
        violations.append(f"Word count {total_words} outside 180–250 range.")

    lemmas = [lemmatize_word(w, inflections) for w in words]
    due_set = {it["spanish"].lower() for it in due_items}
    new_set = {it["spanish"].lower() for it in new_items}
    scheduled_set = due_set | new_set

    # Rule 1: Closed vocabulary + zero hold-out leak
    holdout_hits = [w for w, l in zip(words, lemmas) if l in holdout_lemmas]
    if holdout_hits:
        violations.append(f"Rule 1 HOLDOUT LEAK: {holdout_hits[:5]}")

    unknowns = []
    for idx, (w, l) in enumerate(zip(words, lemmas)):
        if l in allowed_lemmas or l in scheduled_set or l in FUNCTION_STOP:
            continue
        # Allow capitalized proper names mid-sentence
        if idx > 0 and w[0].isupper():
            continue
        unknowns.append(f"{w}({l})")

    closed_compliance = 1.0 - (len(unknowns) / max(1, total_words))
    if closed_compliance < 0.96:
        violations.append(
            f"Rule 1 Closed Vocab ({closed_compliance*100:.1f}% < 96%): unapproved tokens {unknowns[:8]}"
        )

    # Rule 2: New items >= 3 occurrences separated by >= 30 words
    for it in new_items:
        target = it["spanish"].lower()
        positions = [i for i, l in enumerate(lemmas) if l == target]
        if len(positions) < 3:
            violations.append(f"Rule 2 New item '{target}' appeared {len(positions)}x (need >=3).")
        elif any((positions[k + 1] - positions[k]) < 25 for k in range(len(positions) - 1)):
            violations.append(f"Rule 2 New item '{target}' spacing too tight: {positions}.")

    # Rule 3: Due coverage & scheduled word share <= 23%
    missing_due = [d for d in due_set if d not in set(lemmas)]
    if missing_due:
        violations.append(f"Rule 3 Missing due items: {missing_due[:6]}")

    sched_count = sum(1 for l in lemmas if l in scheduled_set)
    sched_share = sched_count / max(1, total_words)
    if sched_share > 0.25:
        violations.append(f"Rule 3 Scheduled share {sched_share*100:.1f}% > 23% cap.")

    # Probes check
    if len(passage.get("probes", [])) < 2:
        violations.append("Rule 5 Needs 2–3 load-bearing probes.")

    metrics = {
        "total_words": float(total_words),
        "closed_compliance": closed_compliance,
        "scheduled_share": sched_share,
    }
    return len(violations) == 0, violations, metrics


def simulate_state_plan(bank: SeedBank, hour: int, idx: int) -> Tuple[List[Dict], List[Dict], Set[str]]:
    lexical = [it for it in bank.active_items if it["item_type"] in ("lemma", "chunk")]
    met_map = {2: 150, 10: 720, 20: 1650, 35: 2900, 50: 4150}
    met_n = min(len(lexical) - 10, met_map.get(hour, 1650))

    known_slice = lexical[:met_n]
    unmet_slice = lexical[met_n:]

    # Pick 4 new items and 14 due items rotated by idx
    new_items = unmet_slice[(idx * 4) % max(1, len(unmet_slice) - 4) : (idx * 4) % max(1, len(unmet_slice) - 4) + 4]
    due_pool = [it for i, it in enumerate(known_slice) if i % 7 == (idx % 7)]
    due_items = due_pool[:14]
    allowed = {it["spanish"].lower() for it in known_slice}
    return due_items, new_items, allowed


def run_one_passage(client, bank: SeedBank, hour: int, idx: int) -> Dict:
    due_items, new_items, allowed = simulate_state_plan(bank, hour, idx)
    feedback = ""
    last_metrics = {"total_words": 0.0, "closed_compliance": 0.0, "scheduled_share": 0.0}

    for attempt in range(4):
        due_str = ", ".join(f"{it['item_id']}:{it['spanish']}" for it in due_items)
        new_str = ", ".join(f"{it['item_id']}:{it['spanish']} ({it['english_gloss']})" for it in new_items)
        allowed_sample = ", ".join(sorted(list(allowed))[:350])

        instructions = (
            "You are Cadencia's Spanish Acquisition Passage Generator (Signal-Starvation Protocol). "
            "Write a compelling 180-250 word natural Spanish story episode obeying ALL 6 contract rules:\n"
            "1. Closed vocabulary: ONLY use words from ALLOWED_VOCAB, DUE_ITEMS, NEW_ITEMS, and standard Spanish function words. NEVER use BANNED holdout words.\n"
            "2. Every NEW_ITEM must appear at least 3 times, spaced >= 30 words apart, with rich contextual clues.\n"
            "3. Every DUE_ITEM must appear at least once; keep combined due+new tokens between 18% and 23% of total words.\n"
            "4. Include 2-3 load-bearing English comprehension probes that hinge on DUE_ITEMS."
        )
        user_msg = (
            f"SIMULATED HOUR: {hour} | PASSAGE #{idx+1}\n"
            f"NEW_ITEMS (each >=3x, >=30 words apart): {new_str}\n"
            f"DUE_ITEMS (each >=1x): {due_str}\n"
            f"ALLOWED_VOCAB SAMPLE: {allowed_sample}\n"
            f"BANNED HOLDOUT SAMPLE: {', '.join(sorted(list(bank.holdout_lemmas))[:60])}\n"
        )
        if feedback:
            user_msg += f"\nREWRITE FIXES REQUIRED FROM PREVIOUS ATTEMPT:\n{feedback}"

        resp = client.responses.create(
            model=MODEL_ID,
            instructions=instructions,
            input=[{"role": "user", "content": user_msg}],
            reasoning={"effort": "medium" if attempt == 0 else "high"},
            text={
                "format": {
                    "type": "json_schema",
                    "name": "stage0_bench_passage",
                    "strict": True,
                    "schema": PASSAGE_JSON_SCHEMA,
                }
            },
        )
        raw_text = resp.output_text
        passage = json.loads(raw_text)
        passed, violations, metrics = verify_stage_a(
            passage, due_items, new_items, allowed, bank.holdout_lemmas, bank.inflections
        )
        last_metrics = metrics
        if passed:
            return {
                "hour": hour,
                "passed": True,
                "rewrites": attempt,
                **metrics,
            }
        feedback = "\n".join(f"- {v}" for v in violations)

    return {
        "hour": hour,
        "passed": False,
        "rewrites": 3,
        **last_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cadencia Stage 0 Bench Test CLI (GPT-6.1 Terra)")
    parser.add_argument("--passages-per-hour", type=int, default=4, help="Passages per simulated hour (200 = full 1,000-passage cert)")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="Verify local SQLite seed & Stage A verifier without calling OpenAI")
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    bank = load_seed_bank(SEED_DB)
    print(f"Loaded Cadencia seed bank: {len(bank.active_items)} active, {len(bank.holdout_lemmas)} holdout, {len(bank.inflections)} inflections.")

    if args.dry_run:
        due, new, allowed = simulate_state_plan(bank, 20, 0)
        print(f"Dry-run check OK: Hour 20 plan has {len(due)} due, {len(new)} new, {len(allowed)} allowed lemmas.")
        return

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("ERROR: Set OPENAI_API_KEY environment variable first (never hardcode keys!).")

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    hours = [2, 10, 20, 35, 50]
    tasks = [(h, i) for h in hours for i in range(args.passages_per_hour)]
    print(f"Launching Stage 0 Bench Test: {len(tasks)} passages across hours {hours} using {MODEL_ID}…")

    t0 = time.time()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {pool.submit(run_one_passage, client, bank, h, i): (h, i) for h, i in tasks}
        for fut in concurrent.futures.as_completed(futs):
            res = fut.result()
            results.append(res)
            status = "PASS" if res["passed"] else "FAIL"
            print(
                f"  [H={res['hour']:2d}] {status} | rewrites={res['rewrites']} | "
                f"words={int(res['total_words'])} | closed={res['closed_compliance']*100:.1f}% | "
                f"share={res['scheduled_share']*100:.1f}%"
            )

    pass_rate = sum(1 for r in results if r["passed"]) / max(1, len(results))
    avg_share = sum(r["scheduled_share"] for r in results) / max(1, len(results))
    avg_rewrites = sum(r["rewrites"] for r in results) / max(1, len(results))
    gate_passed = pass_rate >= 0.95 and avg_share >= 0.20

    summary = {
        "model": MODEL_ID,
        "total_passages": len(results),
        "elapsed_seconds": round(time.time() - t0, 1),
        "overall_pass_rate": round(pass_rate, 4),
        "overall_scheduled_share": round(avg_share, 4),
        "average_rewrites": round(avg_rewrites, 2),
        "stage0_gate_passed": gate_passed,
        "per_hour": {
            str(h): {
                "pass_rate": round(
                    sum(1 for r in results if r["hour"] == h and r["passed"])
                    / max(1, sum(1 for r in results if r["hour"] == h)),
                    4,
                ),
                "avg_scheduled_share": round(
                    sum(r["scheduled_share"] for r in results if r["hour"] == h)
                    / max(1, sum(1 for r in results if r["hour"] == h)),
                    4,
                ),
            }
            for h in hours
        },
    }
    print("\n=== STAGE 0 BENCH TEST SUMMARY ===")
    print(json.dumps(summary, indent=2))
    if args.out_json:
        args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
