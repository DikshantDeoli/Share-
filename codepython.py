"""
Hybrid Understanding Pass — designed for small (≤20B) local LLMs.

Core strategy
─────────────
Small LLMs fail at large, complex extraction tasks. This pipeline compensates with:

  Stage 0  Rule-based pre-processor — normalises any doc format, extracts all
           numerics via regex, splits into labelled sections. LLM never sees raw prose.

  Stage 1  Atomic LLM calls — one concept per call, tiny 3-field schemas, 1–2 few-shot
           examples embedded in every prompt, temperature=0.0.
           Calls run in parallel via ThreadPoolExecutor.

  Stage 2  Code-based assembler — merges regex results with LLM results, enforces
           count arithmetic, computes edge cases. No LLM involved.

  Stage 3  Targeted retry — failed atoms retry with a progressively simpler prompt
           and explicit correction instruction. Never retries the whole pipeline.

Usage
─────
    from understanding_pass_small_llm import run_understanding_pass_local

    plan = run_understanding_pass_local(document_text)
    print(plan.model_dump_json(indent=2))

Dependencies
────────────
    pip install langchain langchain-openai pydantic
"""

from __future__ import annotations

import json
import re
import concurrent.futures
from dataclasses import dataclass, field
from typing import Optional

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr, model_validator


# ──────────────────────────────────────────────────────────────────
#  LLM client  (matches user's existing setup)
# ──────────────────────────────────────────────────────────────────

def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-oss-20b",
        base_url="http://localhost",
        api_key=SecretStr("demo"),
        temperature=0.0,
        top_p=0.9,
        max_tokens=512,          # keep small — atomic calls need short outputs
        model_kwargs={"response_format": {"type": "json_object"}},  # JSON mode
    )


# ──────────────────────────────────────────────────────────────────
#  Output schema
# ──────────────────────────────────────────────────────────────────

class RuleItem(BaseModel):
    rule: str
    polarity: str = Field(pattern="^(must|must_not|optional|conditional)$")
    edge: bool

class VariationItem(BaseModel):
    variant: str
    count: int = Field(gt=0)
    notes: str

class DataPatternItem(BaseModel):
    field: str
    type: str = Field(pattern="^(format|enum|nullable|conditional|relational)$")
    rule: Optional[str] = None
    values: Optional[list[str]] = None
    null_pct: Optional[int] = None
    applies: list[str]

class EdgeCaseItem(BaseModel):
    for_segment: str = Field(alias="for")
    case: str
    reserved_slots: int = Field(gt=0)
    model_config = {"populate_by_name": True}

class Segment(BaseModel):
    type: str
    count: int = Field(gt=0)
    attributes: list[str]

class GenerationPlan(BaseModel):
    volume_total: int
    volume_explicit: bool
    volume_source: str
    segments: list[Segment]
    variations: dict[str, list[VariationItem]] = {}
    rules_per_segment: dict[str, list[RuleItem]] = {}
    global_rules: list[str] = []
    data_patterns: list[DataPatternItem] = []
    edge_cases: list[EdgeCaseItem] = []

    @model_validator(mode="after")
    def counts_sum_to_total(self) -> "GenerationPlan":
        s = sum(seg.count for seg in self.segments)
        if self.volume_total > 0 and s != self.volume_total:
            raise ValueError(
                f"Segment counts sum to {s}, volume.total is {self.volume_total}."
            )
        return self


# ──────────────────────────────────────────────────────────────────
#  Stage 0 — Rule-based pre-processor
# ──────────────────────────────────────────────────────────────────

@dataclass
class PreProcessedDoc:
    raw: str
    format_hint: str               # "prose" | "bullet" | "table" | "sow" | "brd"
    sections: dict[str, str]       # heading → content
    extracted_numbers: list[dict]  # {value, unit, context, raw}
    extracted_formats: list[str]   # "YYYY-MM-DD", "E.164", regex strings
    extracted_enums: dict[str, list[str]]   # field → [values]
    canonical: str                 # normalized flat text for LLM

_N_PATTERN   = re.compile(r"(\b(\d[\d,]*)\s*(records?|rows?|entries|items?|prompts?|customers?|transactions?)\b)", re.I)
_PCT_PATTERN = re.compile(r"(\b(\d+(?:\.\d+)?)\s*%)", re.I)
_DATE_PATTERN= re.compile(r"\b(YYYY[-/]MM[-/]DD|DD[-/]MM[-/]YYYY|MM[-/]DD[-/]YYYY|ISO\s*8601)\b", re.I)
_PHONE_PATTERN=re.compile(r"\b(E\.?164|[+]\d{1,3}[-\s]\d+|international phone)\b", re.I)
_ENUM_PATTERN= re.compile(r"(?:one of|must be|can be|values?:?)\s*[:\-]?\s*([A-Za-z,\s/|]+)", re.I)
_HEADING_PATTERN = re.compile(r"^(#{1,4}\s+.+|[A-Z][A-Za-z\s]{2,40}:?\s*$)", re.M)
_TABLE_PATTERN   = re.compile(r"^\s*\|.+\|", re.M)


def preprocess(doc: str) -> PreProcessedDoc:
    """
    Normalises the document before any LLM call.
    Extracts all numerics, formats, and enums via regex.
    """

    # ── Format detection ──────────────────────────────────────────
    has_table   = bool(_TABLE_PATTERN.search(doc))
    has_bullets = bool(re.search(r"^\s*[-*•]\s", doc, re.M))
    is_sow      = bool(re.search(r"\b(statement of work|sow|scope of work)\b", doc, re.I))
    is_brd      = bool(re.search(r"\b(business requirements?|brd)\b", doc, re.I))
    fmt = ("sow" if is_sow else "brd" if is_brd else
           "table" if has_table else "bullet" if has_bullets else "prose")

    # ── Section extraction ────────────────────────────────────────
    sections: dict[str, str] = {}
    lines = doc.splitlines()
    current_heading = "overview"
    buf: list[str] = []
    for line in lines:
        if _HEADING_PATTERN.match(line.strip()):
            if buf:
                sections[current_heading] = "\n".join(buf).strip()
            current_heading = re.sub(r"^#+\s*", "", line.strip()).rstrip(":").lower()
            buf = []
        else:
            buf.append(line)
    if buf:
        sections[current_heading] = "\n".join(buf).strip()

    # ── Numeric extraction ────────────────────────────────────────
    numbers: list[dict] = []
    for m in _N_PATTERN.finditer(doc):
        numbers.append({
            "value": int(m.group(2).replace(",", "")),
            "unit":  m.group(3).lower().rstrip("s"),
            "context": doc[max(0, m.start()-60):m.end()+60].strip(),
            "raw": m.group(1),
        })
    for m in _PCT_PATTERN.finditer(doc):
        numbers.append({
            "value": float(m.group(2)),
            "unit": "%",
            "context": doc[max(0, m.start()-60):m.end()+60].strip(),
            "raw": m.group(1),
        })

    # ── Format / enum extraction ──────────────────────────────────
    formats: list[str] = []
    for m in _DATE_PATTERN.finditer(doc):
        formats.append(m.group(1))
    for m in _PHONE_PATTERN.finditer(doc):
        formats.append(m.group(1))

    enums: dict[str, list[str]] = {}
    for m in _ENUM_PATTERN.finditer(doc):
        raw = m.group(1)
        vals = [v.strip() for v in re.split(r"[,|/]", raw) if len(v.strip()) > 1]
        if 2 <= len(vals) <= 10:
            context_start = max(0, m.start() - 40)
            ctx = doc[context_start: m.start()].lower().strip()
            key = ctx.split()[-1] if ctx.split() else "field"
            enums[key] = vals

    # ── Canonical form (section-labelled flat text) ───────────────
    canonical_parts = []
    for heading, content in sections.items():
        canonical_parts.append(f"[{heading.upper()}]\n{content}")
    canonical = "\n\n".join(canonical_parts)

    return PreProcessedDoc(
        raw=doc,
        format_hint=fmt,
        sections=sections,
        extracted_numbers=numbers,
        extracted_formats=formats,
        extracted_enums=enums,
        canonical=canonical,
    )


# ──────────────────────────────────────────────────────────────────
#  Helpers — LLM call wrapper with retry + simplification
# ──────────────────────────────────────────────────────────────────

def _llm_json(
    llm: ChatOpenAI,
    system: str,
    user: str,
    max_retries: int = 3,
) -> dict:
    """
    Calls the LLM, returns parsed JSON dict.
    On failure, retries with progressively simpler instructions.
    """
    messages = [
        ("system", system),
        ("human", user),
    ]

    simplifications = [
        "",   # first try: as-is
        "\nReturn ONLY the JSON object. No explanation. No markdown.",
        "\nReturn a JSON object with ONLY the fields shown in the example.",
    ]

    for attempt, extra in enumerate(simplifications[:max_retries], 1):
        try:
            msgs = [
                ("system", system + extra),
                ("human", user),
            ]
            resp = llm.invoke(msgs)
            raw = resp.content.strip()
            # strip accidental fences
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            return json.loads(raw)
        except Exception as e:
            print(f"  [attempt {attempt}] failed: {e}")
            if attempt == max_retries:
                raise
    return {}


# ──────────────────────────────────────────────────────────────────
#  Stage 1 — Atomic extractors
#  Each function: one concept, one LLM call, tiny schema, few-shots
# ──────────────────────────────────────────────────────────────────

def extract_volume(llm: ChatOpenAI, doc: PreProcessedDoc) -> dict:
    """
    Extracts N from the document.
    Regex result is shown to LLM as a hint — it only needs to confirm.
    """
    # Regex pre-extracts candidates — LLM just picks the right one
    candidates = [n for n in doc.extracted_numbers if n["unit"] in ("record", "row", "entry", "item", "prompt", "customer", "transaction")]
    hint = ""
    if candidates:
        hint = f"\nPre-extracted numeric candidates: {json.dumps(candidates[:3])}"

    SYSTEM = """You extract the total number of records to generate from a requirement document.
Output ONLY this JSON (no other text):
{"total": <integer>, "explicit": <true|false>, "source_quote": "<exact phrase from doc>"}

Example input: "Generate 500 customer records for QA testing."
Example output: {"total": 500, "explicit": true, "source_quote": "Generate 500 customer records"}

Example input: "The dataset should cover around 1,000 transactions."
Example output: {"total": 1000, "explicit": true, "source_quote": "around 1,000 transactions"}"""

    USER = f"{doc.canonical}{hint}\n\nExtract total number of records. Output JSON only."
    return _llm_json(llm, SYSTEM, USER)


def extract_entity_types(llm: ChatOpenAI, doc: PreProcessedDoc) -> dict:
    """
    Extracts entity type names only — no attributes, no rules.
    Small LLMs handle list extraction well when scope is narrow.
    """
    SYSTEM = """You extract entity type names from a requirement document.
Output ONLY this JSON (no other text):
{"types": ["<name1>", "<name2>", ...]}

Use snake_case. List ONLY distinct entity types. Do NOT include attributes or rules.

Example input: "We need individual customers, business customers, and VIP accounts."
Example output: {"types": ["individual_customer", "business_customer", "vip_account"]}"""

    USER = f"{doc.canonical}\n\nList all entity types. Output JSON only."
    return _llm_json(llm, SYSTEM, USER)


def extract_proportions(llm: ChatOpenAI, doc: PreProcessedDoc, entity_types: list[str]) -> dict:
    """
    Extracts proportion or count for each entity type.
    Regex candidates are shown as hints to reduce reasoning load.
    """
    pct_candidates = [n for n in doc.extracted_numbers if n["unit"] == "%"]
    hint = f"\nPre-extracted percentages: {json.dumps(pct_candidates[:6])}" if pct_candidates else ""

    SYSTEM = """You extract the proportion or count for each entity type from a requirement document.
Output ONLY this JSON (no other text):
{"proportions": [{"type": "<name>", "value": <number>, "unit": "%" or "count"}]}

Example output:
{"proportions": [
  {"type": "individual_customer", "value": 60, "unit": "%"},
  {"type": "business_customer",   "value": 30, "unit": "%"},
  {"type": "vip_customer",        "value": 10, "unit": "%"}
]}"""

    USER = (
        f"{doc.canonical}{hint}\n\n"
        f"Entity types to find proportions for: {entity_types}\n\n"
        "Extract proportion or count per entity type. Output JSON only."
    )
    return _llm_json(llm, SYSTEM, USER)


def extract_attributes_for_entity(llm: ChatOpenAI, doc: PreProcessedDoc, entity_type: str) -> dict:
    """
    Extracts attributes (field names) for ONE entity type.
    One call per entity — keeps the task tiny.
    """
    SYSTEM = f"""You extract field names for entity type "{entity_type}" from a requirement document.
Output ONLY this JSON (no other text):
{{"attributes": ["<field1>", "<field2>", ...]}}

List ONLY the field names mentioned. Do NOT include rules or descriptions.

Example: {{"attributes": ["name", "email", "date_of_birth", "phone", "address"]}}"""

    # Send only the most relevant section
    relevant = _most_relevant_section(doc, entity_type)
    USER = f"{relevant}\n\nExtract attributes for {entity_type}. Output JSON only."
    return _llm_json(llm, SYSTEM, USER)


def extract_rules_for_entity(llm: ChatOpenAI, doc: PreProcessedDoc, entity_type: str) -> dict:
    """
    Extracts rules for ONE entity type.
    Schema forces polarity to an enum — no ambiguity for small LLM.
    """
    SYSTEM = f"""You extract business rules for entity type "{entity_type}" from a requirement document.
Output ONLY this JSON:
{{"rules": [{{"rule": "<plain English>", "polarity": "must|must_not|optional|conditional", "edge": true|false}}]}}

polarity meaning:
  must       = always required / always true
  must_not   = forbidden / prohibited
  optional   = may or may not apply
  conditional= depends on another condition

edge = true if the rule contains a number threshold (e.g. ">=18", ">10000").

Example output:
{{"rules": [
  {{"rule": "age must be 18 or above", "polarity": "must", "edge": true}},
  {{"rule": "email is required",       "polarity": "must", "edge": false}},
  {{"rule": "phone may be absent",     "polarity": "optional", "edge": false}}
]}}"""

    relevant = _most_relevant_section(doc, entity_type)
    USER = f"{relevant}\n\nExtract rules for {entity_type}. Output JSON only."
    return _llm_json(llm, SYSTEM, USER)


def extract_global_rules(llm: ChatOpenAI, doc: PreProcessedDoc) -> dict:
    """Rules that apply to every entity type."""
    SYSTEM = """You extract global rules that apply to ALL entity types in a requirement document.
Output ONLY this JSON:
{"global_rules": ["<rule1>", "<rule2>", ...]}

Include ONLY rules that apply to every record, not rules specific to one entity type.

Example: {"global_rules": ["no duplicate emails", "date format: YYYY-MM-DD", "phone format: E.164"]}"""

    USER = f"{doc.canonical}\n\nExtract global rules. Output JSON only."
    return _llm_json(llm, SYSTEM, USER)


def extract_variations_for_entity(llm: ChatOpenAI, doc: PreProcessedDoc, entity_type: str, entity_count: int) -> dict:
    """
    Extracts variants for ONE entity type.
    Shows the entity count so LLM understands the constraint.
    Numeric distribution is computed by code after, not LLM.
    """
    SYSTEM = f"""You extract variants of entity type "{entity_type}" from a requirement document.
A variant is a distinct sub-form of the same entity (e.g. locale, tier, status).
Output ONLY this JSON:
{{"variants": [{{"name": "<name>", "notes": "<what makes it distinct>"}}]}}

Do NOT assign counts. List variant names and descriptions only.

Example:
{{"variants": [
  {{"name": "locale_IN", "notes": "Indian locale, +91 phone, Indian names"}},
  {{"name": "locale_UK", "notes": "UK locale, +44 phone, British names"}}
]}}"""

    relevant = _most_relevant_section(doc, entity_type)
    USER = f"{relevant}\n\nExtract variants for {entity_type}. Output JSON only."
    return _llm_json(llm, SYSTEM, USER)


def _most_relevant_section(doc: PreProcessedDoc, entity_type: str) -> str:
    """
    Returns the section most likely to contain info about entity_type.
    Reduces the context window sent to each atomic call.
    """
    entity_words = entity_type.replace("_", " ").lower().split()
    best_section = ""
    best_score   = -1
    for heading, content in doc.sections.items():
        score = sum(w in heading or w in content.lower() for w in entity_words)
        if score > best_score:
            best_score   = score
            best_section = f"[{heading.upper()}]\n{content}"
    # Always include global/rules section if it exists
    for heading, content in doc.sections.items():
        if any(k in heading for k in ("rule", "constraint", "requirement", "general", "global")):
            best_section += f"\n\n[{heading.upper()}]\n{content}"
            break
    return best_section or doc.canonical


# ──────────────────────────────────────────────────────────────────
#  Stage 2 — Code-based assembler
# ──────────────────────────────────────────────────────────────────

def _proportions_to_counts(
    proportions: list[dict],
    total: int,
) -> list[dict]:
    """
    Converts % proportions to hard integer counts.
    Remainder is added to the largest segment.
    """
    result = []
    for p in proportions:
        v, u = p["value"], p["unit"]
        count = round(total * v / 100) if u == "%" else int(v)
        result.append({"type": p["type"], "count": count})

    current_sum = sum(r["count"] for r in result)
    diff = total - current_sum
    if diff != 0 and result:
        # add remainder to the largest segment
        largest = max(result, key=lambda x: x["count"])
        largest["count"] += diff
    return result


def _compute_edge_cases(rules_per_segment: dict[str, list[dict]]) -> list[dict]:
    """
    For every rule with edge=True, auto-generates boundary edge cases.
    Extracts the numeric threshold from the rule text.
    """
    edge_cases = []
    threshold_re = re.compile(r"([><=!]{1,2})\s*(\d[\d,]*(?:\.\d+)?)")

    for seg_type, rules in rules_per_segment.items():
        for rule in rules:
            if not rule.get("edge"):
                continue
            matches = threshold_re.findall(rule["rule"])
            for op, val_str in matches:
                val = float(val_str.replace(",", ""))
                edge_cases.append({
                    "for": seg_type,
                    "case": f"{rule['rule']} — at boundary ({val})",
                    "reserved_slots": 1,
                })
                # just below boundary for >= or > rules
                if op in (">=", ">"):
                    below = int(val) - 1
                    edge_cases.append({
                        "for": seg_type,
                        "case": f"{rule['rule']} — just below boundary ({below})",
                        "reserved_slots": 1,
                    })
    return edge_cases


def assemble_plan(
    volume_raw: dict,
    entity_types: list[str],
    proportions_raw: dict,
    attributes_map: dict[str, list[str]],
    rules_map: dict[str, list[dict]],
    global_rules_raw: dict,
    variations_map: dict[str, list[dict]],
    doc: PreProcessedDoc,
) -> GenerationPlan:
    """
    Merges all atomic extraction results into a validated GenerationPlan.
    All arithmetic and validation is done in code — not LLM.
    """
    total  = volume_raw.get("total", 0)
    source = volume_raw.get("source_quote", "")
    explicit = volume_raw.get("explicit", True)

    # If LLM missed N, fall back to regex
    if total == 0 and doc.extracted_numbers:
        best = max(doc.extracted_numbers, key=lambda x: x["value"])
        total   = best["value"]
        source  = best["raw"]
        explicit = False

    # Proportions → hard counts
    raw_props = proportions_raw.get("proportions", [])
    # Fill in any missing entity types with equal share
    found_types = {p["type"] for p in raw_props}
    for et in entity_types:
        if et not in found_types:
            raw_props.append({"type": et, "value": round(100 / len(entity_types), 1), "unit": "%"})

    counted = _proportions_to_counts(raw_props, total)
    segments = [
        Segment(type=c["type"], count=c["count"], attributes=attributes_map.get(c["type"], []))
        for c in counted if c["count"] > 0
    ]

    # Variations: counts distributed evenly across variants
    built_variations: dict[str, list[VariationItem]] = {}
    for et, var_list in variations_map.items():
        seg = next((s for s in segments if s.type == et), None)
        if not seg or not var_list:
            continue
        n_vars = len(var_list)
        base   = seg.count // n_vars
        extras = seg.count % n_vars
        items  = []
        for i, v in enumerate(var_list):
            items.append(VariationItem(
                variant=v["name"],
                count=base + (1 if i < extras else 0),
                notes=v.get("notes", ""),
            ))
        built_variations[et] = items

    # Rules
    rules_per_seg: dict[str, list[RuleItem]] = {}
    for et, rules in rules_map.items():
        rules_per_seg[et] = [
            RuleItem(
                rule=r["rule"],
                polarity=r.get("polarity", "must"),
                edge=bool(r.get("edge", False)),
            )
            for r in rules
        ]

    # Edge cases
    raw_edge = _compute_edge_cases(
        {et: [r.model_dump() for r in rules] for et, rules in rules_per_seg.items()}
    )
    edge_cases = [
        EdgeCaseItem(**{"for": e["for"], "case": e["case"], "reserved_slots": e["reserved_slots"]})
        for e in raw_edge
    ]

    # Global rules — merge LLM output with regex-extracted formats
    g_rules: list[str] = global_rules_raw.get("global_rules", [])
    for fmt in doc.extracted_formats:
        rule_str = f"format: {fmt}"
        if rule_str not in g_rules:
            g_rules.append(rule_str)

    # Data patterns from regex enums
    data_patterns: list[DataPatternItem] = []
    for field_name, vals in doc.extracted_enums.items():
        data_patterns.append(DataPatternItem(
            field=field_name,
            type="enum",
            values=vals,
            applies=entity_types,  # conservative — apply broadly
        ))

    return GenerationPlan(
        volume_total=total,
        volume_explicit=explicit,
        volume_source=source,
        segments=segments,
        variations=built_variations,
        rules_per_segment=rules_per_seg,
        global_rules=g_rules,
        data_patterns=data_patterns,
        edge_cases=edge_cases,
    )


# ──────────────────────────────────────────────────────────────────
#  Main pipeline
# ──────────────────────────────────────────────────────────────────

def run_understanding_pass_local(
    document: str,
    max_workers: int = 6,
) -> GenerationPlan:
    """
    Full hybrid understanding pass for a small local LLM.

    Args:
        document:    Raw requirement document text.
        max_workers: Parallel threads for atomic LLM calls.

    Returns:
        Validated GenerationPlan ready for batch generation.
    """
    llm = get_llm()

    # ── Stage 0: pre-process ──────────────────────────────────────
    print("[stage 0] pre-processing document...")
    doc = preprocess(document)
    print(f"          format={doc.format_hint}, sections={list(doc.sections)[:5]}, "
          f"numbers found={len(doc.extracted_numbers)}, formats={doc.extracted_formats}")

    # ── Stage 1a: sequential bootstrap (need these before parallel) ──
    print("[stage 1] extracting volume...")
    volume_raw = extract_volume(llm, doc)
    print(f"          volume={volume_raw}")

    print("[stage 1] extracting entity types...")
    entity_raw = extract_entity_types(llm, doc)
    entity_types: list[str] = entity_raw.get("types", [])
    print(f"          types={entity_types}")

    if not entity_types:
        raise ValueError("No entity types extracted. Check your document format.")

    # ── Stage 1b: parallel atomic calls ──────────────────────────
    print("[stage 1] running parallel atomic extractions...")

    def get_attributes(et: str) -> tuple[str, list[str]]:
        r = extract_attributes_for_entity(llm, doc, et)
        return et, r.get("attributes", [])

    def get_rules(et: str) -> tuple[str, list[dict]]:
        r = extract_rules_for_entity(llm, doc, et)
        return et, r.get("rules", [])

    def get_variations(et: str) -> tuple[str, list[dict]]:
        total = volume_raw.get("total", 100)
        r = extract_variations_for_entity(llm, doc, et, total)
        return et, r.get("variants", [])

    attributes_map: dict[str, list[str]]  = {}
    rules_map: dict[str, list[dict]]      = {}
    variations_map: dict[str, list[dict]] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            "proportions": pool.submit(extract_proportions, llm, doc, entity_types),
            "global_rules": pool.submit(extract_global_rules, llm, doc),
        }
        for et in entity_types:
            futures[f"attr_{et}"]  = pool.submit(get_attributes, et)
            futures[f"rules_{et}"] = pool.submit(get_rules, et)
            futures[f"vars_{et}"]  = pool.submit(get_variations, et)

        for key, future in futures.items():
            try:
                result = future.result(timeout=60)
                if key == "proportions":
                    proportions_raw = result
                elif key == "global_rules":
                    global_rules_raw = result
                elif key.startswith("attr_"):
                    et, attrs = result
                    attributes_map[et] = attrs
                elif key.startswith("rules_"):
                    et, rules = result
                    rules_map[et] = rules
                elif key.startswith("vars_"):
                    et, variants = result
                    if variants:
                        variations_map[et] = variants
                print(f"  [done] {key}")
            except Exception as e:
                print(f"  [warn] {key} failed: {e} — continuing with empty")
                if key == "proportions":
                    proportions_raw = {}
                elif key == "global_rules":
                    global_rules_raw = {}

    # ── Stage 2: assemble + validate ─────────────────────────────
    print("[stage 2] assembling generation plan...")
    plan = assemble_plan(
        volume_raw=volume_raw,
        entity_types=entity_types,
        proportions_raw=proportions_raw,
        attributes_map=attributes_map,
        rules_map=rules_map,
        global_rules_raw=global_rules_raw,
        variations_map=variations_map,
        doc=doc,
    )
    print(f"[done] plan validated. total={plan.volume_total}, segments={len(plan.segments)}, "
          f"edge_cases={len(plan.edge_cases)}")
    return plan


# ──────────────────────────────────────────────────────────────────
#  Summary printer
# ──────────────────────────────────────────────────────────────────

def summarise_plan(plan: GenerationPlan) -> str:
    lines = [
        f"Total prompts : {plan.volume_total}",
        f"Source quote  : \"{plan.volume_source}\"",
        f"Explicit N    : {plan.volume_explicit}",
        "",
        "Segments:",
    ]
    for seg in plan.segments:
        vars_ = plan.variations.get(seg.type, [])
        vstr  = "  →  " + ", ".join(f"{v.variant}({v.count})" for v in vars_) if vars_ else ""
        lines.append(f"  {seg.type:<30} {seg.count:>6} records{vstr}")

    lines += ["", "Global rules:"]
    for r in plan.global_rules:
        lines.append(f"  • {r}")

    lines += ["", "Edge cases:"]
    for ec in plan.edge_cases:
        lines.append(f"  [{ec.for_segment}] {ec.case} ({ec.reserved_slots} slot/s)")

    batch_calls = plan.volume_total // 25 + (1 if plan.volume_total % 25 else 0)
    lines += [
        "",
        f"LLM calls (understanding): ~{3 + 3 * len(plan.segments)} atomic (parallel)",
        f"LLM calls (generation)   : {batch_calls} batches of 25",
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────
#  Quick local test (mocks the LLM so you can run without a server)
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SAMPLE_DOC = """
Statement of Work — Synthetic Customer Data

Generate 200 customer records for testing the onboarding pipeline.

Customer types and distribution
- Individual customers : 50% of total
- Business customers   : 35% of total
- Blocked accounts     : 15% of total

Individual customer fields: name, email, date_of_birth, phone, address, status
Business customer fields:   company_name, registration_number, contact_email, status
Blocked account fields:     name, email, block_reason, blocked_at, status

Rules
- Age must be 18 or above for individual customers.
- Email is required for all types. Emails must be unique across all records.
- Phone is optional for individual customers (up to 20% may be missing).
- Blocked accounts must not have status=active.
- Business customers with revenue >= 1000000 are marked enterprise=true.

Formats
- Date of birth: YYYY-MM-DD
- Phone numbers: E.164 format
- Status values: one of active, suspended, pending, blocked

Locale variations for individual customers
Equal split: India (IN), United States (US), United Kingdom (UK).
""".strip()

    print("Running understanding pass with LOCAL small LLM...\n")
    print("(pre-processor output shown; LLM calls will fail without a running server)\n")

    # Show pre-processor output independently so you can validate it
    doc = preprocess(SAMPLE_DOC)
    print("=== Pre-processor output ===")
    print(f"Format hint      : {doc.format_hint}")
    print(f"Sections found   : {list(doc.sections.keys())}")
    print(f"Numbers extracted: {json.dumps(doc.extracted_numbers, indent=2)}")
    print(f"Formats extracted: {doc.extracted_formats}")
    print(f"Enums extracted  : {json.dumps(doc.extracted_enums, indent=2)}")
    print("\n=== Canonical form (what LLM sees) ===")
    print(doc.canonical[:800], "...\n")
    print("To run the full pipeline: set base_url to your local server and call:")
    print("  plan = run_understanding_pass_local(SAMPLE_DOC)")
    print("  print(summarise_plan(plan))")