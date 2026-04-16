"""
Jinja2-based prompt loader.

Templates live in src/prompts/ alongside this module.
Usage:
    from src.prompts.loader import render_prompt
    system_prompt = render_prompt("system_prompt")
    prefix = render_prompt("context_prefix", memory_context=ctx_str)
"""

import os

from jinja2 import Environment, FileSystemLoader, StrictUndefined

_TEMPLATE_DIR = os.path.dirname(__file__)

_env = Environment(
    loader=FileSystemLoader(_TEMPLATE_DIR),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_prompt(name: str, **ctx: object) -> str:
    """
    Render a Jinja2 template from src/prompts/{name}.j2.

    Args:
        name: Template name without the .j2 extension.
        **ctx: Template variables.

    Returns:
        Rendered string.

    Raises:
        jinja2.TemplateNotFound: If the template file doesn't exist.
    """
    template = _env.get_template(f"{name}.j2")
    return template.render(**ctx)
