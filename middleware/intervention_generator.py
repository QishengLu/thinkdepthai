"""Layer 3: Targeted intervention generation based on detected deficiency."""

from langchain_core.messages import HumanMessage

from .deficiency_detector import ALL_DEFICIENCIES

INTERVENTION_PROMPT = """\
You are generating a targeted metacognitive prompt for an RCA agent \
that is exhibiting a specific cognitive deficiency during its investigation.

## Detected Deficiency
{deficiency_id} — {deficiency_name}: {deficiency_description}
Evidence: {evidence}

## Current Investigation Context
- Recent intents: {recent_intents}
- Services investigated: {services_list}
- Missing evidence dimensions: {phases_missing}
- Recent reasoning: {recent_reasoning}

## Rules
1. Do NOT tell the agent what the answer is or which service is the root cause
2. Do NOT name specific tables or SQL queries to run
3. DO make the agent aware of its blind spot
4. DO pose a pointed question that leads it to self-correct
5. Keep it to 2-4 sentences, like a senior SRE reviewing a junior's investigation
6. Be concrete — reference what the agent has actually done (from the context)

## Output
Respond with ONLY the intervention message (2-4 sentences). No JSON, no markdown.
"""


class InterventionGenerator:
    """Generate context-specific intervention for a detected deficiency."""

    def __init__(self, model):
        self.model = model

    def generate(
        self, deficiency_id: str, evidence: str, state_context: dict
    ) -> str:
        ddef = ALL_DEFICIENCIES[deficiency_id]

        prompt = INTERVENTION_PROMPT.format(
            deficiency_id=deficiency_id,
            deficiency_name=ddef["name"],
            deficiency_description=ddef["description"],
            evidence=evidence,
            **state_context,
        )

        response = self.model.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
