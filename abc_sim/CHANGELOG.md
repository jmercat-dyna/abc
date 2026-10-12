# Simulator task/protocol versions

`abc_sim/VERSION` versions simulator behavior and scoring independently of the
`abc-minimal` package release. Bump the major number for incompatible scoring
or episode termination; bump minor for compatible task features; bump patch
for fixes that preserve protocol semantics. Record this value as `sim_version`
in eval artifacts. Historical unversioned runs must not be mixed with a
versioned protocol.

## 2.0.2

- Permit shallow non-supporting contact with another rack peg without adding
  that contact's force to the required threaded-handle load.
- Permit shallow mug-to-mug tangency only when both mugs independently have
  released threaded-handle support and the required peg load. Unsupported
  stacks, table/gripper contacts and deep mug/rack penetration still fail.
- Training replay disagreements motivated these incidental-contact fixes;
  they preserve the intended criterion that every mug hangs independently.

## 2.0.1

- Allow shallow incidental trunk/base bracing alongside released, threaded
  handle support. Verified training replays exposed false negatives when mugs
  genuinely hung from a peg but lightly touched other rack collision pieces.
- Retain the configured minimum peg load, empty-handle geometry and rejection
  of table/mug-stack/gripper contact or deep rack penetration. Bracing alone
  never counts as hanging.
- Reuse reset MuJoCo scratch data and memoize per-sample peg geometry checks
  without changing scoring criteria.

## 2.0.0

- Require mug handles to enclose and bear load on a rack peg after release;
  nearby table mugs or stacked mugs no longer count as hanging.
- Put-relative placement tolerances become shared validated configuration for
  sampler/scorer, including task-specific evaluator options.
- Package and report an independent simulator protocol version.

## 1.0.0 — retrospective baseline

Commit `d8e7ad519d70b98d894f4ae7b49c76f384eac7e9` defines this baseline;
that historical commit did not yet contain a version file.

- Held ball-tray countdown completion replaces passive survival success.
- Relative prompts reject already-solved reset geometry and retry transient
  sampling failures.
- Invalid opaque-box insertions terminate failed episodes; exact counts retain
  final-state success, with `ever_correct` as a diagnostic.
