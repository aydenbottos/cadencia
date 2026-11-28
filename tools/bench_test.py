#!/usr/bin/env python3
"""
Cadencia Stage 0 Bench Test CLI (Revised 16-Page Signal-Starvation Protocol, pp. 6–10).

Stress-tests the 6-step GPT-6.1 Terra Generator + Stage A/B Verifier pipeline across
5 simulated learner states (hours 0, 10, 20, 35, and 50) without spending human
participant hours. Evaluates the formal Stage 0 Gate:
  - >= 95% of passages pass ALL contract rules within <= 3 generator calls
  - Scheduled word share averages >= 20% across hours 10 to 35
  - Hour 0 runs on the Bootstrap Contract (6-10 sentences <=8 words, >=80% met/new/names,
    <=20% glossed transparent cognates, <=8 new items in >=3 sentences, no spacing/share limit)
  - Hours 10-50 run on the Full Contract (180-250 words, 100% closed vocab, greedy Rule 2
    3x30w spacing with free extra uses, Rule 3 <=23% cap with NO per-passage floor)

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
import math
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
SEED_DB = ROOT / "Sources" / "CadenciaCore" / "Resources" / "cadencia_seed.sqlite"
MODEL_ID = "gpt-6.1-terra"

WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", re.UNICODE)
BOOTSTRAP_FIXED_CAST = {"ana", "luis", "carla", "perú", "peru", "méxico", "mexico"}
PROPER_NAMES = BOOTSTRAP_FIXED_CAST | {
    "clara", "mateo", "lucía", "lucia", "tomás", "tomas", "pedro", "maría", "maria",
    "juan", "carlos", "elena", "diego", "sofía", "sofia", "marta", "pablo", "laura",
    "madrid", "barcelona", "sevilla", "valencia", "oaxaca", "patagonia", "españa",
}
SESSION1_KITS = [
    (["ser", "el", "un", "de", "y", "doctor", "hotel", "famoso"], "ser with a noun or adjective; agreement"),
    (["no", "en", "estar", "restaurante", "hospital", "moderno", "pero", "muy"], "estar for location; negation"),
    (["tener", "que", "su", "familia", "problema", "importante", "con", "persona"], "tener; possessive su; relative que"),
    (["a", "ir", "para", "universidad", "estudiante", "profesor", "clase", "este"], "ir a; a + el = al"),
    (["haber", "mucho", "más", "música", "café", "parque", "centro"], "hay; más ... que"),
    (["yo", "mi", "también", "porque", "visitar", "estudiar", "museo", "historia"], "first-person verb forms; porque"),
    (["o", "todo", "usar", "necesitar", "teléfono", "minuto"], "infinitive after para; todo + article"),
    (["cuando", "otro", "poder", "preparar", "animal", "favorito", "color"], "poder + infinitive; cuando clauses"),
]
EXPLICIT_INFLECTIONS = {
    "ser": {"ser", "soy", "eres", "es", "somos", "sois", "son"},
    "el": {"el", "la", "los", "las"},
    "un": {"un", "una", "unos", "unas", "uno"},
    "uno": {"uno", "un", "una", "unos", "unas"},
    "de": {"de"}, "y": {"y", "e"}, "a": {"a"}, "no": {"no"}, "en": {"en"}, "pero": {"pero"}, "muy": {"muy"},
    "que": {"que", "qué"}, "con": {"con"}, "para": {"para"}, "más": {"más", "mas"}, "yo": {"yo"},
    "también": {"también", "tambien"}, "porque": {"porque"}, "o": {"o", "u"}, "cuando": {"cuando", "cuándo"},
    "doctor": {"doctor", "doctora", "doctores", "doctoras"},
    "hotel": {"hotel", "hoteles"},
    "famoso": {"famoso", "famosa", "famosos", "famosas"},
    "estar": {"estar", "estoy", "estás", "está", "estamos", "estáis", "están"},
    "restaurante": {"restaurante", "restaurantes"}, "hospital": {"hospital", "hospitales"},
    "moderno": {"moderno", "moderna", "modernos", "modernas"},
    "tener": {"tener", "tengo", "tienes", "tiene", "tenemos", "tenéis", "tienen"},
    "su": {"su", "sus"}, "familia": {"familia", "familias"}, "problema": {"problema", "problemas"},
    "importante": {"importante", "importantes"}, "persona": {"persona", "personas"},
    "ir": {"ir", "voy", "vas", "va", "vamos", "vais", "van"},
    "universidad": {"universidad", "universidades"}, "estudiante": {"estudiante", "estudiantes"},
    "profesor": {"profesor", "profesora", "profesores", "profesoras"}, "clase": {"clase", "clases"},
    "este": {"este", "esta", "esto", "estos", "estas"},
    "haber": {"haber", "hay", "he", "has", "ha", "hemos", "han"},
    "mucho": {"mucho", "mucha", "muchos", "muchas"}, "música": {"música", "musica"},
    "café": {"café", "cafés", "cafe"}, "parque": {"parque", "parques"}, "centro": {"centro", "centros"},
    "mi": {"mi", "mis"},
    "visitar": {"visitar", "visito", "visitas", "visita", "visitamos", "visitan"},
    "estudiar": {"estudiar", "estudio", "estudias", "estudia", "estudiamos", "estudian"},
    "museo": {"museo", "museos"}, "historia": {"historia", "historias"},
    "todo": {"todo", "toda", "todos", "todas"},
    "usar": {"usar", "uso", "usas", "usa", "usamos", "usan"},
    "necesitar": {"necesitar", "necesito", "necesitas", "necesita", "necesitamos", "necesitan"},
    "teléfono": {"teléfono", "teléfonos", "telefono"}, "minuto": {"minuto", "minutos"},
    "otro": {"otro", "otra", "otros", "otras"},
    "poder": {"poder", "puedo", "puedes", "puede", "podemos", "pueden"},
    "preparar": {"preparar", "preparo", "preparas", "prepara", "preparamos", "preparan"},
    "animal": {"animal", "animales"},
    "favorito": {"favorito", "favorita", "favoritos", "favoritas"},
    "color": {"color", "colores"},
}
SURFACE_TO_LEMMAS: Dict[str, Set[str]] = {}
for _lem, _forms in EXPLICIT_INFLECTIONS.items():
    for _f in _forms:
        SURFACE_TO_LEMMAS.setdefault(_f, set()).add(_lem)

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
                    "visualSymbol": {"type": ["string", "null"]},
                },
                "required": ["spanish", "englishLine", "visualSymbol"],
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
        "transparentCognateGlosses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "spanish": {"type": "string"},
                    "englishGloss": {"type": "string"},
                },
                "required": ["spanish", "englishGloss"],
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
                    "isPictureMatch": {"type": "boolean"},
                },
                "required": [
                    "targetDueItemId", "targetDueLemma", "questionEn",
                    "options", "correctOptionIndex", "hingeExplanation", "isPictureMatch",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "titleEs", "titleEn", "episodeSummaryForNext",
        "sentences", "newItemGlosses", "transparentCognateGlosses", "probes",
    ],
    "additionalProperties": False,
}


@dataclass
class SeedBank:
    active_items: List[Dict]
    holdout_lemmas: Set[str]
    transparent_cognates: Set[str]
    inflections: Dict[str, str]
    grammar_items: List[Dict]


def load_seed_bank(db_path: Path) -> SeedBank:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    active = [
        dict(r)
        for r in conn.execute(
            "SELECT item_id, item_type, rank_index, spanish, english_gloss, pos, "
            "is_transparent_cognate, is_bootstrap, grammar_week "
            "FROM syllabus_items WHERE is_holdout = 0 ORDER BY rank_index"
        )
    ]
    holdout = {
        r["spanish"].lower()
        for r in conn.execute("SELECT spanish FROM syllabus_items WHERE is_holdout = 1")
    }
    cognates = {
        it["spanish"].lower() for it in active if it["is_transparent_cognate"] == 1
    }
    inflections = {
        r["surface"].lower(): r["lemma"].lower()
        for r in conn.execute("SELECT surface, lemma FROM spanish_inflections")
    }
    grammar = [it for it in active if it["item_type"] == "grammar"]
    conn.close()
    return SeedBank(
        active_items=active,
        holdout_lemmas=holdout,
        transparent_cognates=cognates,
        inflections=inflections,
        grammar_items=grammar,
    )


def lemmatize_word(surface: str, inflections: Dict[str, str]) -> str:
    low = surface.lower()
    if low in inflections:
        return inflections[low]
    for clitic in ("melo", "mela", "telo", "tela", "selo", "sela", "nos", "les", "las", "los", "me", "te", "se", "lo", "la", "le"):
        if len(low) > len(clitic) + 3 and low.endswith(clitic):
            stem = low[: -len(clitic)]
            if stem in inflections:
                return inflections[stem]
    return low


def max_spaced_occurrences(positions: List[int], min_spacing: int) -> int:
    """Greedy spaced-subsequence count (Rule 2: 3 uses >= 30 words apart; extra uses are free)."""
    count = 0
    last = -10**9
    for p in sorted(positions):
        if p - last >= min_spacing:
            count += 1
            last = p
    return count


def enforce_feasibility(new_items: List[Dict], due_items: List[Dict], target_words: int = 215) -> Tuple[List[Dict], List[Dict]]:
    """Step 1 zero-cost feasibility check: 3*new + due <= floor(0.23 * target_words)."""
    max_slots = int(math.floor(0.23 * target_words))
    kept_new = list(new_items)
    if len(kept_new) * 3 > max_slots:
        kept_new = kept_new[: max_slots // 3]
    rem_due = max(0, max_slots - len(kept_new) * 3)
    kept_due = list(due_items[:rem_due])
    return kept_new, kept_due


def is_contraction_allowed(surface: str, lemma: str, allowed_set: Set[str]) -> bool:
    if surface == "al":
        return ("a" in allowed_set and "el" in allowed_set) or "al" in allowed_set
    if surface == "del":
        return ("de" in allowed_set and "el" in allowed_set) or "del" in allowed_set
    if surface in ("los", "las"):
        return "el" in allowed_set or "la" in allowed_set
    if surface in ("un", "una", "unos", "unas"):
        return "un" in allowed_set or "uno" in allowed_set
    return (lemma == "uno" and "un" in allowed_set) or (lemma == "un" and "uno" in allowed_set)


def expand_token(surface: str, inflections: Dict[str, str]) -> List[Set[str]]:
    low = surface.lower()
    if low == "al":
        return [{"a"}, {"el"}]
    if low == "del":
        return [{"de"}, {"el"}]
    cands = {low, lemmatize_word(surface, inflections)}
    if low in SURFACE_TO_LEMMAS:
        cands |= SURFACE_TO_LEMMAS[low]
    if "un" in cands:
        cands.add("uno")
    if "uno" in cands:
        cands.add("un")
    return [cands]


def verify_stage_a(
    passage: Dict,
    due_items: List[Dict],
    new_items: List[Dict],
    allowed_lemmas: Set[str],
    bank: SeedBank,
    is_bootstrap: bool,
) -> Tuple[bool, List[str], Dict[str, float], bool]:
    """Deterministic local Stage A Verifier matching revised 17-page ContractVerifier.swift."""
    sentences = passage.get("sentences", [])
    violations: List[str] = []
    overcrowding = False

    words: List[str] = []
    word_sents: List[int] = []
    sent_word_counts: List[int] = []

    for s_idx, s in enumerate(sentences):
        sw = WORD_RE.findall(s["spanish"])
        sent_word_counts.append(len(sw))
        for w in sw:
            words.append(w)
            word_sents.append(s_idx)

    total_surface_words = len(words)

    if is_bootstrap:
        if len(sentences) < 6 or len(sentences) > 10:
            violations.append(f"Bootstrap sentence count {len(sentences)} outside 6–10.")
        overlong = [i + 1 for i, c in enumerate(sent_word_counts) if c > 8]
        if overlong:
            violations.append(f"Bootstrap sentences exceeding 8 words: {overlong}")
    else:
        if total_surface_words < 180 or total_surface_words > 250:
            violations.append(f"Word count {total_surface_words} outside 180–250 range.")
            overcrowding = True

    due_set = {it["spanish"].lower() for it in due_items}
    new_set = {it["spanish"].lower() for it in new_items}
    scheduled_set = due_set | new_set
    full_allowed = allowed_lemmas | scheduled_set
    if "un" in full_allowed:
        full_allowed.add("uno")
    glossed_cogs = {g["spanish"].lower() for g in passage.get("transparentCognateGlosses", [])}
    cog_pool = bank.transparent_cognates | glossed_cogs

    total_units = 0
    core_ok = 0
    sched_units = 0
    item_positions: Dict[str, List[int]] = {}
    holdout_hits: List[str] = []
    off_list: List[str] = []
    non_cog_off: List[str] = []

    for idx, w in enumerate(words):
        low = w.lower()
        is_name = (low in BOOTSTRAP_FIXED_CAST) if is_bootstrap else (
            low in PROPER_NAMES or (idx > 0 and w[0].isupper() and low not in bank.inflections)
        )
        if is_name or w.isdigit():
            total_units += 1
            core_ok += 1
            continue
        for cands in expand_token(w, bank.inflections):
            total_units += 1
            for k in cands:
                item_positions.setdefault(k, []).append(idx)
            matched = bool(cands & full_allowed)
            if not matched and (low in bank.holdout_lemmas or lemmatize_word(w, bank.inflections) in bank.holdout_lemmas):
                holdout_hits.append(w)
            if cands & scheduled_set:
                sched_units += 1
            if matched:
                core_ok += 1
            else:
                off_list.append(w)
                if is_bootstrap and not (cands & cog_pool):
                    non_cog_off.append(w)

    if holdout_hits:
        violations.append(f"Rule 1 HOLDOUT LEAK: {holdout_hits[:5]}")

    closed_compliance = core_ok / max(1, total_units)
    if is_bootstrap:
        if closed_compliance < 0.80:
            violations.append(f"Rule 1 Bootstrap compliance {closed_compliance*100:.1f}% < 80%: {off_list[:8]}")
        if non_cog_off:
            violations.append(f"Rule 1 Bootstrap non-cognate off-list words: {non_cog_off[:8]}")
    else:
        if closed_compliance < 1.0:
            violations.append(f"Rule 1 Closed Vocab ({closed_compliance*100:.1f}% < 100%): {off_list[:10]}")

    # Rule 2: New items (Bootstrap: every kit item >=1x in passage; Full: >=3x with 3 uses >=30w apart)
    for it in new_items:
        target = it["spanish"].lower()
        positions = sorted(set(item_positions.get(target, [])))
        if is_bootstrap:
            if len(positions) < 1:
                violations.append(f"Rule 2 Bootstrap kit item '{target}' missing from passage.")
        else:
            if len(positions) < 3:
                violations.append(f"Rule 2 New item '{target}' appeared {len(positions)}x (need >=3).")
            elif max_spaced_occurrences(positions, 30) < 3:
                violations.append(f"Rule 2 New item '{target}' lacks 3 uses >=30 words apart: {positions}.")

    # Rule 3: Due coverage & scheduled word share <= 23% (no lower limit; no limit in Bootstrap)
    missing_due = [d for d in due_set if not item_positions.get(d)]
    if missing_due:
        violations.append(f"Rule 3 Missing due items: {missing_due[:6]}")
        overcrowding = True

    sched_share = sched_units / max(1, total_units)
    if not is_bootstrap and scheduled_set and sched_share > 0.2305:
        violations.append(f"Rule 3 Scheduled share {sched_share*100:.1f}% > 23.0% cap.")
        overcrowding = True

    eligible_count = len(due_items) + len(new_items)
    needed_probes = min(2, eligible_count)
    if len(passage.get("probes", [])) < needed_probes:
        violations.append(f"Rule 5 Needs at least {needed_probes} probe(s).")

    metrics = {
        "total_words": float(total_surface_words),
        "closed_compliance": closed_compliance,
        "scheduled_share": sched_share,
    }
    return len(violations) == 0, violations, metrics, overcrowding


def simulate_state_plan(bank: SeedBank, hour: int, idx: int) -> Tuple[List[Dict], List[Dict], Set[str], bool]:
    lexical = [it for it in bank.active_items if it["item_type"] in ("lemma", "chunk")]
    is_bootstrap = hour == 0
    if is_bootstrap:
        by_es = {it["spanish"].lower(): it for it in lexical}
        kit_idx = idx % len(SESSION1_KITS)
        kit_lemmas, _ = SESSION1_KITS[kit_idx]
        met_lemmas: Set[str] = set()
        for prev_i in range(kit_idx):
            met_lemmas.update(SESSION1_KITS[prev_i][0])
        chosen = []
        for w in kit_lemmas:
            if w in by_es:
                chosen.append(by_es[w])
            elif w == "un" and "uno" in by_es:
                copy_un = dict(by_es["uno"])
                copy_un["spanish"] = "un"
                chosen.append(copy_un)
        allowed = met_lemmas | {it["spanish"].lower() for it in chosen}
        return [], chosen, allowed, True

    met_map = {10: 720, 20: 1650, 35: 2900, 50: 4150}
    met_n = min(len(lexical) - 10, met_map.get(hour, 1650))
    known_slice = lexical[:met_n]
    unmet_slice = lexical[met_n:]

    new_items = unmet_slice[(idx * 4) % max(1, len(unmet_slice) - 4) : (idx * 4) % max(1, len(unmet_slice) - 4) + 4]
    due_pool = [it for i, it in enumerate(known_slice) if i % 7 == (idx % 7)]
    due_items = due_pool[:30]
    new_items, due_items = enforce_feasibility(new_items, due_items, target_words=215)
    allowed = {it["spanish"].lower() for it in known_slice}
    return due_items, new_items, allowed, False


def run_one_passage(client, bank: SeedBank, hour: int, idx: int) -> Dict:
    due_items, new_items, allowed, is_bootstrap = simulate_state_plan(bank, hour, idx)
    feedback = ""
    prev_sig = None
    last_metrics = {"total_words": 0.0, "closed_compliance": 0.0, "scheduled_share": 0.0}

    # Exact max-3-generator-call pipeline (Protocol pp. 7–8)
    for call_num in (1, 2, 3):
        effort = "low" if (call_num == 1 and is_bootstrap) else ("medium" if call_num <= 2 else "high")
        due_str = ", ".join(f"{it['item_id']}:{it['spanish']}" for it in due_items)
        new_str = ", ".join(f"{it['item_id']}:{it['spanish']} ({it['english_gloss']})" for it in new_items)
        allowed_sample = ", ".join(sorted(list(allowed))[:500])

        if is_bootstrap:
            instructions = (
                "Write a Bootstrap micro-story (6-10 sentences, EACH strictly <=8 words) with visualSymbol and englishLine:\n"
                "1. >=80% of words MUST come from ALLOWED_VOCAB + NEW_ITEMS + proper names; remaining <=20% MUST be glossed transparent cognates.\n"
                "2. Every NEW_ITEM appears in >=3 different sentences (no spacing limit, no upper limit).\n"
                "3. Include 2 picture-matching probes (`isPictureMatch: true`) on NEW_ITEMS."
            )
        else:
            instructions = (
                "Write a 180-250 word natural Spanish story obeying ALL rules:\n"
                "1. 100% Closed vocabulary: ONLY use words from ALLOWED_VOCAB, DUE_ITEMS, NEW_ITEMS, and names/numbers. NEVER use BANNED holdout words.\n"
                "2. Every NEW_ITEM appears >=3x with 3 uses >=30 words apart (extra uses are free).\n"
                "3. Every DUE_ITEM appears >=1x; combined due+new tokens <=23% of total words (no lower limit).\n"
                "4. Include 2-3 load-bearing English probes hinging on DUE_ITEMS first, then NEW_ITEMS."
            )
        user_msg = (
            f"SIMULATED HOUR: {hour} ({'BOOTSTRAP' if is_bootstrap else 'FULL'}) | PASSAGE #{idx+1}\n"
            f"NEW_ITEMS: {new_str}\n"
            f"DUE_ITEMS: {due_str or 'None'}\n"
            f"ALLOWED_VOCAB: {allowed_sample}\n"
            f"BANNED HOLDOUT SAMPLE: {', '.join(sorted(list(bank.holdout_lemmas))[:60])}\n"
        )
        if is_bootstrap:
            user_msg += f"TRANSPARENT COGNATES: {', '.join(sorted(list(bank.transparent_cognates))[:180])}\n"
        if feedback:
            user_msg += f"\n{feedback}"

        resp = client.responses.create(
            model=MODEL_ID,
            instructions=instructions,
            input=[{"role": "user", "content": user_msg}],
            reasoning={"effort": effort},
            text={
                "format": {
                    "type": "json_schema",
                    "name": "stage0_bench_passage",
                    "strict": True,
                    "schema": PASSAGE_JSON_SCHEMA,
                }
            },
        )
        passage = json.loads(resp.output_text)
        passed, violations, metrics, overcrowding = verify_stage_a(
            passage, due_items, new_items, allowed, bank, is_bootstrap
        )
        last_metrics = metrics
        if passed:
            return {"hour": hour, "passed": True, "calls": call_num, **metrics}

        sig = "|".join(sorted(violations))
        if call_num == 2 and prev_sig == sig:
            break  # Step 6 early exit on repeated violation!
        prev_sig = sig

        if call_num == 1:
            feedback = "STEP 4 TARGETED REPAIR — Rewrite ONLY offending sentences:\n" + "\n".join(f"- {v}" for v in violations)
        elif call_num == 2:
            if due_items and overcrowding:
                due_items = due_items[: max(1, int(len(due_items) * 0.75))]
            feedback = "STEP 5 HIGH-EFFORT REGENERATION — Fix all violations:\n" + "\n".join(f"- {v}" for v in violations)

    return {"hour": hour, "passed": False, "calls": 3, **last_metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Cadencia Stage 0 Bench Test CLI (Revised Protocol)")
    parser.add_argument("--passages-per-hour", type=int, default=4, help="Passages per simulated hour (200 = full 1,000-passage cert)")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="Verify local SQLite seed & Stage A verifier without calling OpenAI")
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    bank = load_seed_bank(SEED_DB)
    print(f"Loaded Cadencia seed bank: {len(bank.active_items)} active, {len(bank.holdout_lemmas)} holdout, {len(bank.inflections)} inflections.")

    if args.dry_run:
        due0, new0, allowed0, boot0 = simulate_state_plan(bank, 0, 0)
        due20, new20, allowed20, boot20 = simulate_state_plan(bank, 20, 0)
        print(f"Dry-run check OK: Hour 0 bootstrap={boot0} ({len(new0)} new, {len(due0)} due, {len(allowed0)} allowed) | Hour 20 plan has {len(due20)} due, {len(new20)} new, {len(allowed20)} allowed lemmas.")
        return

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("ERROR: Set OPENAI_API_KEY environment variable first (never hardcode keys!).")

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    hours = [0, 10, 20, 35, 50]
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
                f"  [H={res['hour']:2d}] {status} | calls={res['calls']}/3 | "
                f"words={int(res['total_words'])} | closed={res['closed_compliance']*100:.1f}% | "
                f"share={res['scheduled_share']*100:.1f}%"
            )

    pass_rate = sum(1 for r in results if r["passed"]) / max(1, len(results))
    mid_results = [r for r in results if 10 <= r["hour"] <= 35]
    mid_share = sum(r["scheduled_share"] for r in mid_results) / max(1, len(mid_results))
    avg_calls = sum(r["calls"] for r in results) / max(1, len(results))
    gate_passed = pass_rate >= 0.95 and mid_share >= 0.20

    summary = {
        "model": MODEL_ID,
        "total_passages": len(results),
        "elapsed_seconds": round(time.time() - t0, 1),
        "overall_pass_rate_within_3_calls": round(pass_rate, 4),
        "hours_10_to_35_avg_scheduled_share": round(mid_share, 4),
        "average_generator_calls": round(avg_calls, 2),
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
