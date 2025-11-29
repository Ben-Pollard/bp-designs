# Exploration Directory

This directory contains detailed experiment logs from parameter exploration sessions.

## Purpose

- **Record raw experimental data** - Parameters tested, visual observations, failures
- **Archive detailed findings** - Too detailed for LEARNINGS.md but valuable for reference
- **Enable investigation** - When exploring similar parameter spaces, consult past experiments

## File Naming Convention

```
YYYY-MM-DD_topic-description.md
```

Examples:
- `2025-01-23_branch-angles.md`
- `2025-01-24_tapering-rates.md`
- `2025-01-25_collision-avoidance-params.md`

## Entry Format

Each experiment file should follow this structure:

```markdown
# [Topic] Exploration

**Date:** YYYY-MM-DD
**Pattern:** [Pattern family, e.g., Space Colonization]
**Goal:** [What you're trying to discover]

## Parameters Tested

| Parameter | Values Tested |
|-----------|---------------|
| branch_angle | 25°, 30°, 35°, 40°, 45°, 50° |
| attraction_distance | 8, 10, 12, 15 |
| seed | 0, 1, 2 |

## Results

### What Worked
- 35-40° branch angles produced most natural vein-like appearance
- attraction_distance of 10-12 gave good coverage without crowding

### What Didn't Work
- Angles <30°: Too cramped, branches overlapping
- Angles >45°: Too sparse, unnatural gaps
- attraction_distance >15: Branches too spread out

### Surprising Findings
- Small angle variations (±5°) significantly improved organic feel
- Consistent results across different seeds (good sign)

## Visual Notes
- [Attach images or describe key visual observations]
- Best result: seed=1, angle=37.5°, attraction=11

## Follow-Up Questions
- Test finer gradations around 35-40° range?
- Explore interaction between angle and attraction distance?

## Distilled to LEARNINGS.md
- [x] Updated LEARNINGS.md section 1.2 with branch angle findings
```

## Reading Guidelines

**These files are NOT read automatically by Claude.**

They should be consulted:
- When exploring similar parameter spaces
- When investigating why something works
- When comparing current results to past experiments

**After 3-5 related experiments:**
- Distill key findings into LEARNINGS.md
- Archive or summarize older logs if they're no longer relevant

## Maintenance

Every ~10 sessions or when directory gets large:
- Review files for patterns and insights
- Update LEARNINGS.md with consolidated findings
- Archive or remove outdated logs
- Keep recent (3-6 months) experiments accessible
