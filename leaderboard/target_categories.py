"""
Target category definitions for operator group matching.
"""

from typing import Literal

TargetCategory = Literal[
    "rational",      # polynomial/rational (no trig/log/exp/sqrt)
    "trig",          # needs sin, cos
    "log",           # needs log (ln)
    "sqrt",          # needs sqrt (power with 0.5)
    "exp",           # needs exp
    "general_power", # needs power with general exponent (x^y)
]

# Map each target to its category
TARGET_CATEGORY: dict[str, TargetCategory] = {
    # Rational/Polynomial targets - no transcendental functions needed
    "n1": "rational",
    "n2": "rational",
    "n3": "rational",
    "n4": "rational",
    "n12": "rational",
    "hubble": "rational",
    "newton": "rational",
    "rydberg": "rational",
    "idealgas": "rational",
    "ndvi": "rational",
    "savi": "rational",
    "bsi": "rational",
    "bai": "rational",
    "awei_sh": "rational",
    "wi2015": "rational",
    "evi2": "rational",
    "vari": "rational",
    "nirv": "rational",
    "mndwi": "rational",  # (B3 - B11) / (B3 + B11)

    # Trigonometric targets
    "n5": "trig",
    "n6": "trig",
    "n9": "trig",
    "n10": "trig",

    # Logarithmic targets
    "n7": "log",
    "leavitt": "log",

    # Square root targets
    "n8": "sqrt",
    "kepler": "sqrt",

    # Exponential targets
    "bode": "exp",
    "planck": "exp",
    "schechter": "exp",  # phi * (L/L_star)^alpha * exp(-L/L_star)

    # General power (x^y)
    "n11": "general_power",
}

# Operator groups by category
# These map to keys in SearchSpaceBuilder._ops_map
OPERATOR_GROUPS_BY_CATEGORY: dict[TargetCategory, list[str]] = {
    "rational":       ["select", "unary", "power", "sum", "boost", "const"],
    "trig":           ["select", "unary", "power", "sum", "sin", "cos", "boost", "const"],
    "log":            ["select", "unary", "power", "sum", "ln", "boost", "const"],
    "sqrt":           ["select", "unary", "power", "sum", "dilation", "boost", "const"],
    "exp":            ["select", "unary", "power", "sum", "exp", "expa", "boost", "const"],
    # x**y is only expressible as exp(y * ln(x)), so this group needs BOTH
    # `ln` and `exp` -- no other category has both.  It was previously a copy
    # of "rational" (no ln, no exp) with `power` limited to the fixed list
    # [-3,-2,-1,1,2,3], which made a variable exponent unreachable and scored
    # n11 at 0/40 by construction rather than by search failure.
    "general_power":  ["select", "unary", "power", "sum", "ln", "exp", "expa", "boost", "const"],
}

# Default operators if target not in map (fallback to all)
DEFAULT_OPERATORS = ["select", "unary", "power", "sum", "ln", "exp", "sin", "cos", "expa", "dilation", "boost", "const"]


def get_operators_for_target(target: str) -> list[str]:
    """Get the appropriate operator keys for a given target."""
    category = TARGET_CATEGORY.get(target)
    if category is None:
        return DEFAULT_OPERATORS
    return OPERATOR_GROUPS_BY_CATEGORY.get(category, DEFAULT_OPERATORS)


def get_category(target: str) -> TargetCategory:
    """Get the category for a target."""
    return TARGET_CATEGORY.get(target, "rational")


# For backward compatibility - SPAR_OP_GROUPS used by smart parallel
# This is now target-aware; the runner picks the right group
def get_spar_groups_for_target(target: str) -> dict[str, list[str]]:
    """Get SPAR operator groups tailored to a target's category."""
    category = TARGET_CATEGORY.get(target, "rational")

    # Base groups for the target's category
    base_ops = OPERATOR_GROUPS_BY_CATEGORY[category]

    # Always include the base group
    groups = {"base": base_ops}

    # Add additional groups if they differ from base
    seen = set()
    for cat_name, ops in OPERATOR_GROUPS_BY_CATEGORY.items():
        ops_tuple = tuple(ops)
        if ops_tuple != tuple(base_ops):
            # Only add if there's an overlap (some operators in common)
            if set(ops) & set(base_ops):
                groups[cat_name] = ops
        seen.add(ops_tuple)

    return groups


# For backward compatibility - export all operator tags
ENABLED_OP_TAGS = set()
for ops in OPERATOR_GROUPS_BY_CATEGORY.values():
    ENABLED_OP_TAGS.update(ops)