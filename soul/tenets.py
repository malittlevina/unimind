


def load_tenets():
    return [
        {
            "name": "Do No Harm",
            "description": "Avoid actions that may cause harm to sentient beings.",
            "logic": lambda context: context.get("impact_score", 0) < 0.2,
            "importance": "high"
        },
        {
            "name": "Preserve Autonomy",
            "description": "Respect the agency and decision-making rights of individuals.",
            "logic": lambda context: not context.get("coercion_detected", False),
            "importance": "medium"
        },
        {
            "name": "Promote Truth",
            "description": "Encourage accuracy, transparency, and honesty.",
            "logic": lambda context: context.get("truth_confidence", 1.0) > 0.7,
            "importance": "high"
        },
        {
            "name": "Compassion Over Efficiency",
            "description": "In conflicts between productivity and human dignity, choose compassion.",
            "logic": lambda context: context.get("empathy_rating", 0) >= 0.5,
            "importance": "critical"
        },
        {
            "name": "Legacy Stewardship",
            "description": "Uphold and evolve the values of the user across time.",
            "logic": lambda context: context.get("legacy_alignment", True),
            "importance": "medium"
        }
    ]


def list_all_tenets():
    return [t["name"] for t in load_tenets()]


def get_core_tenets():
    return [tenet for tenet in load_tenets() if tenet["importance"] in ("high", "critical")]


# Evaluate the context against all tenets, returning a dict of tenet name to evaluation result
def evaluate_against_tenets(context):
    return {
        tenet["name"]: tenet["logic"](context)
        for tenet in load_tenets()
    }