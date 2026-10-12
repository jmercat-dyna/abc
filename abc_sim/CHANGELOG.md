# Simulator task/protocol versions

`abc_sim/VERSION` versions simulator behavior and scoring independently of the
`abc-minimal` package release. Bump the major number for incompatible scoring
or episode termination; bump minor for compatible task features; bump patch
for fixes that preserve protocol semantics. Record this value as `sim_version`
in eval artifacts. Historical unversioned runs must not be mixed with a
versioned protocol.

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
