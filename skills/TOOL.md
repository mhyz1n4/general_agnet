# TOOL.md — External Tool Reference

> **v1 placeholder.** Populate this file with guidance on external tools as they are added.

This file is user-facing documentation for how to invoke external tools available in this
agent. It does **not** control which tools are loaded at runtime — that is determined by
the agent configuration.

---

## Format (for each tool)

```
### tool_name

**Purpose**: one-line description of what the tool does.

**When to use**: guidance on when the agent should prefer this tool over others.

**Inputs**:
- `param_name` (type) — description, constraints.

**Outputs**: description of what the tool returns.

**Example**:
    tool_name(param="value")
```

---

*No external tools are registered for v1.*
