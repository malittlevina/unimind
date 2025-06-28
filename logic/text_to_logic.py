"""
text_to_logic.py

Converts natural language input into symbolic logic expressions or ASTs (abstract syntax trees).
Useful for tasks requiring logical inference, planning, or decision-making structures derived from user prompts or dialogue.

Integration: This module can be called by the symbolic_reasoner, prefrontal_cortex, or planning modules.
"""

from sympy import symbols, And, Or, Not, Implies
import re

def parse_text_to_logic(input_text):
    """
    Converts basic natural language logical statements into symbolic logic.
    Currently supports:
    - 'and', 'or', 'not', 'if ... then ...'

    Args:
        input_text (str): A natural language statement.
    Returns:
        logic_expr (sympy expression): Parsed symbolic logic expression.
    """
    # Define some symbolic variables (placeholder)
    A, B, C, D = symbols('A B C D')

    # Normalize text
    text = input_text.lower().strip()

    # Basic pattern replacement (expandable)
    text = re.sub(r'\bif (.*?) then (.*?)\b', r'Implies(\1, \2)', text)
    text = text.replace(' and ', ' And ')
    text = text.replace(' or ', ' Or ')
    text = text.replace(' not ', ' Not ')

    # Replace named variables with symbols (primitive approach)
    text = text.replace('a', 'A').replace('b', 'B').replace('c', 'C').replace('d', 'D')

    try:
        logic_expr = eval(text)
    except Exception as e:
        logic_expr = f"Error parsing logic: {e}"

    return logic_expr