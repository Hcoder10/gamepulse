from django import template

register = template.Library()


@register.filter
def multiply(value, arg):
    """Multiply a value by an argument. Usage: {{ value|multiply:100 }}"""
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return 0


@register.filter
def percentage(value):
    """Convert a 0-1 float to a percentage string."""
    try:
        return f"{float(value) * 100:.0f}%"
    except (ValueError, TypeError):
        return "0%"
