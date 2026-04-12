# SKILL.md — Skill Catalogue

> **v1 placeholder.** Populate this file as skills are added to the `skills/` directory.

A **skill** is a reusable, self-contained capability that the agent can load on demand.
Each skill lives in its own subdirectory under `skills/` and contains a `SKILL.md` that
the agent reads before executing the skill.

---

## How skills work

1. The system prompt lists available skills when the `skills` template variable is set.
2. When a user request matches a skill, the agent reads that skill's `SKILL.md`.
3. The skill's instructions guide the agent through the task.
4. The agent follows the skill, then returns control to the normal conversation loop.

## Adding a skill

```
skills/
  my_skill/
    SKILL.md    ← required: name, description, step-by-step instructions
    *.py        ← optional: helper code the skill may reference
```

The `SKILL.md` for each skill should include:

```markdown
# Skill: <name>

**Description**: one-line summary (shown to the agent for skill selection).

**Trigger**: conditions under which this skill applies.

**Steps**:
1. ...
2. ...

**Output**: what the skill produces or returns to the user.
```

---

*No skills are registered for v1.*
