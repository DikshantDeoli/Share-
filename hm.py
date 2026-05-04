“””
Understanding Pass — Stage 1 of the bulk prompt generation pipeline.

Sends the full requirement document to the LLM once.
Receives back a compact generation plan JSON that drives all downstream batch calls.

Usage:
pip install anthropic pydantic

```
plan = run_understanding_pass(document_text)
```

“””

import json
import re
from typing import Optional
import anthropic
from pydantic import BaseModel, Field, model_validator

# ─────────────────────────────────────────────

# Output schema — what the LLM must return

# ─────────────────────────────────────────────

class Volume(BaseModel):
total: int                          = Field(gt=0, description=“Total number of records/prompts to generate”)
explicit: bool                      = Field(description=“True if N was directly stated; false if inferred”)
source_quote: str                   = Field(description=“Exact phrase from the document that specifies volume”)

class Segment(BaseModel):
type: str                           = Field(description=“Canonical entity type name, snake_case”)
count: int                          = Field(gt=0, description=“Hard integer count for this segment”)
attributes: list[str]               = Field(description=“Fields this entity type must have”)
source_quote: Optional[str]         = Field(default=None, description=“Phrase from doc that defines this type”)

class Variation(BaseModel):
variant: str                        = Field(description=“Variant label, snake_case”)
count: int                          = Field(gt=0, description=“Number of records for this variant”)
notes: str                          = Field(description=“What makes this variant distinct”)

class Rule(BaseModel):
rule: str                           = Field(description=“Plain-English statement of the rule”)
polarity: str                       = Field(pattern=”^(must|must_not|optional|conditional)$”)
edge: bool                          = Field(description=“True if this rule has a numeric threshold → forces boundary slots”)
source_quote: Optional[str]         = Field(default=None)

class Rules(BaseModel):
segment: dict[str, list[Rule]]      = Field(description=“Per-type rules keyed by segment type name”)
global_rules: list[str]             = Field(description=“Rules that apply to every record regardless of type”)

class DataPattern(BaseModel):
field: str
type: str                           = Field(pattern=”^(format|enum|nullable|conditional|relational)$”)
rule: Optional[str]                 = Field(default=None, description=“Format string or condition expression”)
values: Optional[list[str]]         = Field(default=None, description=“Allowed values for enum type”)
null_pct: Optional[int]             = Field(default=None, ge=0, le=100, description=”% of records where field may be absent”)
applies: list[str]                  = Field(description=“Segment types this pattern applies to”)

class EdgeCase(BaseModel):
for_segment: str                    = Field(alias=“for”, description=“Segment type name”)
case: str                           = Field(description=“Human-readable description of the edge case”)
reserved_slots: int                 = Field(gt=0, description=“Number of prompts reserved for this edge case”)

```
model_config = {"populate_by_name": True}
```

class GenerationPlan(BaseModel):
“””
Compact generation plan output by the understanding pass.
This single object drives every downstream batch generation call.
“””
volume: Volume
segments: list[Segment]
variations: dict[str, list[Variation]]  = Field(default_factory=dict, description=“Variants per segment type”)
rules: Rules
data_patterns: list[DataPattern]        = Field(default_factory=list)
edge_cases: list[EdgeCase]              = Field(default_factory=list)

```
@model_validator(mode="after")
def counts_must_sum_to_total(self) -> "GenerationPlan":
    """Segment counts + edge case reserved slots must equal volume.total."""
    segment_sum = sum(s.count for s in self.segments)
    if segment_sum != self.volume.total:
        raise ValueError(
            f"Segment counts sum to {segment_sum}, "
            f"but volume.total is {self.volume.total}. "
            f"Counts must match exactly."
        )
    return self

@model_validator(mode="after")
def variation_counts_must_match_segment(self) -> "GenerationPlan":
    """If variations exist for a segment, their counts must sum to segment.count."""
    seg_map = {s.type: s.count for s in self.segments}
    for seg_type, variants in self.variations.items():
        var_sum = sum(v.count for v in variants)
        seg_count = seg_map.get(seg_type, 0)
        if var_sum != seg_count:
            raise ValueError(
                f"Variations for '{seg_type}' sum to {var_sum} "
                f"but segment count is {seg_count}."
            )
    return self
```

# ─────────────────────────────────────────────

# System prompt — what the LLM is told to do

# ─────────────────────────────────────────────

SYSTEM_PROMPT = “””
You are a data requirement analyst.
Your job is to read a requirement document and output a compact generation plan
that will be used to generate synthetic data records via bulk LLM prompts.

────────────────────────────────────────────────
WHAT YOU MUST EXTRACT
────────────────────────────────────────────────

1. VOLUME
- Find the total number of records requested (N).
- Copy the exact phrase that states it as source_quote.
- If N is not explicit but can be inferred, set explicit=false and note it.
- If N cannot be determined at all, set total=0 and explicit=false.
1. SEGMENTS
- Identify every distinct entity type (e.g. individual_customer, business_customer).
- Convert any percentages to hard integer counts against N. Remainders go to the largest segment.
- List every field/attribute each type must have.
- Segment counts must sum to exactly N.
1. VARIATIONS
- Within each segment, list every meaningful variant (locale, tier, status, sub-type).
- Assign counts that sum to the parent segment count.
- Only create a variations entry if the document mentions or implies variants for that segment.
1. RULES
- Extract every business rule, constraint, and fact.
- Assign each rule to the correct segment(s), or to global_rules if it applies to all.
- Set polarity: must / must_not / optional / conditional.
- Set edge=true if the rule contains a numeric threshold (triggers boundary slot reservation).
1. DATA PATTERNS
- Extract field-level format rules (date formats, phone formats, regex).
- Extract enum value sets for constrained fields.
- Extract nullability rules with percentage if stated.
- Extract conditional/relational constraints between fields.
1. EDGE CASES
- For every rule where edge=true, create an edge_case entry.
- Each edge case with a threshold gets TWO entries: value AT the threshold, value JUST below.
- reserved_slots is the number of prompts that will be dedicated to this edge case.
- Edge case reserved_slots are already included in the parent segment counts above.

────────────────────────────────────────────────
OUTPUT RULES
────────────────────────────────────────────────

- Output ONLY valid JSON. No prose, no markdown, no explanation.
- Use snake_case for all keys and type/variant names.
- Do not invent rules or fields not found in the document.
- Do not omit any rule or constraint found in the document.
- If a section has nothing to extract, use an empty array [].

────────────────────────────────────────────────
OUTPUT SCHEMA
────────────────────────────────────────────────

{
“volume”: {
“total”: <int>,
“explicit”: <bool>,
“source_quote”: “<string>”
},
“segments”: [
{
“type”: “<string>”,
“count”: <int>,
“attributes”: [”<string>”],
“source_quote”: “<string or null>”
}
],
“variations”: {
“<segment_type>”: [
{
“variant”: “<string>”,
“count”: <int>,
“notes”: “<string>”
}
]
},
“rules”: {
“segment”: {
“<segment_type>”: [
{
“rule”: “<string>”,
“polarity”: “must | must_not | optional | conditional”,
“edge”: <bool>,
“source_quote”: “<string or null>”
}
]
},
“global_rules”: [”<string>”]
},
“data_patterns”: [
{
“field”: “<string>”,
“type”: “format | enum | nullable | conditional | relational”,
“rule”: “<string or null>”,
“values”: [”<string>”] ,
“null_pct”: <int or null>,
“applies”: [”<segment_type>”]
}
],
“edge_cases”: [
{
“for”: “<segment_type>”,
“case”: “<string>”,
“reserved_slots”: <int>
}
]
}
“””.strip()

USER_PROMPT_TEMPLATE = “””
<requirement_document>
{document}
</requirement_document>

Extract the generation plan from the document above.
Return only valid JSON matching the schema. Nothing else.
“””.strip()

# ─────────────────────────────────────────────

# Core understanding pass function

# ─────────────────────────────────────────────

def run_understanding_pass(
document: str,
model: str = “claude-opus-4-6”,
max_retries: int = 3,
) -> GenerationPlan:
“””
Sends the full requirement document to the LLM.
Returns a validated GenerationPlan.

```
Args:
    document:    Full text of the requirement document (max ~3 pages).
    model:       Claude model to use.
    max_retries: Number of times to retry on parse/validation failure.

Returns:
    GenerationPlan: Validated, structured generation plan.

Raises:
    ValueError: If the LLM output cannot be parsed or validated after retries.
"""
client = anthropic.Anthropic()

messages = [
    {
        "role": "user",
        "content": USER_PROMPT_TEMPLATE.format(document=document),
    }
]

last_error: Optional[Exception] = None

for attempt in range(1, max_retries + 1):

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    raw_text = response.content[0].text.strip()

    # ── 1. Strip accidental markdown fences ──────────────────────────────
    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
    raw_text = re.sub(r"\s*```$", "", raw_text)

    # ── 2. Parse JSON ─────────────────────────────────────────────────────
    try:
        raw_dict = json.loads(raw_text)
    except json.JSONDecodeError as e:
        last_error = e
        print(f"[attempt {attempt}] JSON parse failed: {e}")
        # Feed the error back so the model can self-correct
        messages.append({"role": "assistant", "content": raw_text})
        messages.append({
            "role": "user",
            "content": (
                f"Your response was not valid JSON. Parse error: {e}\n"
                f"Return only the corrected JSON. No prose."
            ),
        })
        continue

    # ── 3. Validate against Pydantic schema ───────────────────────────────
    try:
        plan = GenerationPlan.model_validate(raw_dict)
        print(f"[attempt {attempt}] Generation plan validated successfully.")
        return plan
    except Exception as e:
        last_error = e
        print(f"[attempt {attempt}] Schema validation failed: {e}")
        messages.append({"role": "assistant", "content": raw_text})
        messages.append({
            "role": "user",
            "content": (
                f"Your JSON did not pass schema validation. Error: {e}\n"
                f"Fix only what is wrong and return the corrected JSON."
            ),
        })
        continue

raise ValueError(
    f"Understanding pass failed after {max_retries} attempts. "
    f"Last error: {last_error}"
)
```

# ─────────────────────────────────────────────

# Helper — pretty-print the plan

# ─────────────────────────────────────────────

def summarise_plan(plan: GenerationPlan) -> str:
“”“Returns a human-readable summary of the generation plan.”””
lines = [
f”Total prompts to generate : {plan.volume.total}”,
f”Source quote              : "{plan.volume.source_quote}"”,
“”,
“Segments:”,
]
for seg in plan.segments:
variants = plan.variations.get(seg.type, [])
var_str = (
“  →  “ + “, “.join(f”{v.variant}({v.count})” for v in variants)
if variants else “”
)
lines.append(f”  {seg.type:<30} {seg.count:>5} records{var_str}”)

```
lines += ["", "Global rules:"]
for r in plan.rules.global_rules:
    lines.append(f"  • {r}")

lines += ["", "Edge cases reserved:"]
for ec in plan.edge_cases:
    lines.append(f"  [{ec.for_segment}]  {ec.case}  ({ec.reserved_slots} slot/s)")

total_edge = sum(ec.reserved_slots for ec in plan.edge_cases)
lines += [
    "",
    f"Edge case slots           : {total_edge}",
    f"Batch calls needed (~25)  : {plan.volume.total // 25 + (1 if plan.volume.total % 25 else 0)}",
]
return "\n".join(lines)
```

# ─────────────────────────────────────────────

# Example — run with a sample document

# ─────────────────────────────────────────────

EXAMPLE_DOCUMENT = “””
Data Generation Requirement — Customer Records

Overview
Generate 500 synthetic customer records for QA testing of the onboarding system.

Entity types and distribution

- Individual customers  : 60% of total (approx 300 records)
- Business customers    : 30% of total (approx 150 records)
- VIP customers         : 8%  of total (approx 40 records)
- Blocked customers     : 2%  of total (remaining records)

Individual customer fields
name, email (required), date_of_birth, phone (optional), address, status

Business customer fields
company_name, registration_number, vat_id (if VAT registered), contact_name, email, status

VIP customer fields
name, email, tier (gold or platinum), lifetime_spend, account_manager

Blocked customer fields
name, email, block_reason, blocked_at

Rules and constraints

- All customers must be aged 18 or above (individual only).
- Email is required for all types. No two records may share the same email.
- VIP customers must have lifetime_spend >= 10,000.
- VIP tier is gold if spend is between 10,000 and 50,000; platinum if above 50,000.
- Blocked customers must not have any active orders or transactions.
- Phone numbers follow E.164 format. Phone may be absent for up to 10% of individual records.
- Date of birth must use format YYYY-MM-DD.
- Status must be one of: active, suspended, pending.

Locale variations for individual customers
Records should be distributed equally across three locales:
India (IN), United Kingdom (UK), United States (US).
Names, phone country codes, and addresses must match the respective locale.

Additional notes
Include at least one record where an individual customer is exactly 18 years old.
Include boundary records for VIP threshold (spend = 10,000 and spend = 9,999).
“””.strip()

if **name** == “**main**”:
print(“Running understanding pass…\n”)
plan = run_understanding_pass(EXAMPLE_DOCUMENT)

```
print("\n── Generation Plan JSON ──────────────────────────\n")
print(json.dumps(plan.model_dump(by_alias=True), indent=2))

print("\n── Summary ───────────────────────────────────────\n")
print(summarise_plan(plan))
```