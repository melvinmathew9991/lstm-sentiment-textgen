## Phase

<!-- e.g. Phase 2 - Sentiment model & training -->

## What changed

<!-- Behaviour, not a file list. Reference FR-n and D1-D11 where a change closes one. -->

## Exit criteria

<!-- Paste the real output of this phase's Verify command from Phases.md. -->

```
```

## Project audit

<!-- python scripts/audit.py -- paste the summary line. Zero FAIL required. -->

```
```

## Checklist

- [ ] `pytest` green, output pasted above
- [ ] Every new public function has a type hint and a docstring
- [ ] No banned import introduced (`Rules.md` section 2)
- [ ] No new magic number outside a config file (C13)
- [ ] Metrics reported beside their baselines (C11/C16)
- [ ] `python scripts/audit.py` run, zero FAIL, summary pasted above
- [ ] Any WARN triaged: fixed, or recorded in `Memory.md` under Deferred
- [ ] `Memory.md` updated with the phase entry
- [ ] `CHANGELOG.md` updated
- [ ] No trailers in any commit message (`Rules.md` section 10)
