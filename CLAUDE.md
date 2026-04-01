# SUMMARY

This is a general purpose Agent repo that builds an agentic system around LLMs. Main system components are in the `src` folder.

# CODING RULES

1. Write clean, human-readable code.
2. Enforce typing or type hints; include clear function and file-level docstrings.
3. Always add unit and integration test coverage when adding new code.
4. Apply defensive coding.
5. Apply dependency inversion pattern.

# PLANNING RULES

1. Always start from a simple POC but leave room to scale up to 1000x usage.
2. List clear trade-offs on algorithm, tech, and design choices.
3. Persist a separate `plan.md` file in the local repo.
4. Always search online for the latest SOTA solutions or industry standard designs.

# TECH CHOICES

- **Language:** Python
- **Virtual env & package management:** uv
- **Linting:** ruff
