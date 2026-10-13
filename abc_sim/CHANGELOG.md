# Simulator task/protocol versions

`abc_sim/VERSION` versions simulator behavior and scoring independently of the
`abc-minimal` package release. Bump the major number for incompatible scoring
or episode termination; bump minor for compatible task features; bump patch
for fixes that preserve protocol semantics. Record this value as `sim_version`
in eval artifacts. Historical unversioned runs must not be mixed with a
versioned protocol.

## 3.0.2

- Normalize grab-clutter collision type values to Python integers. Newer
  MuJoCo enum bindings compare differently with NumPy scalar tuple membership;
  cylinders/capsules otherwise fail constructor geometry checks. Existing
  primitive regressions pass under MuJoCo 3.8.1 and 3.15.0.
- This compatibility candidate preserves scoring and the existing dependency
  pins. Testing a newer engine is separate from adopting it as the default.

## 3.0.1

- Disable free-body midpoint integration only for the nuts-and-bolts scene
  under the pinned MuJoCo 3.8.1 runtime, using `implicitfast` with the supported
  `invdiscrete` opt-out flag. The previous midpoint update makes nut rotational
  damping add energy: a 0.1 rad/s rotation doubles each 2 ms step and triggers
  BADQACC instead of dissipating energy.
- Preserve mass, inertia, damping, friction, contacts, timestep and scoring.
  The task-specific flag keeps rendering/batching compatible with the pinned
  MuJoCo Warp versions; switching to `implicit` would reject models in the
  upstream 3.8.0.3 backend. No discrete inverse-dynamics caller is used here.
- Retain instability detection and separate simulator-error accounting. This
  numerical fix changes task dynamics and requires separately versioned results.
  It does not claim to fix the independently observed chess solver errors.

## 3.0.0

- Ball-tray balancing now uses the actual scaled, rotated collision floor
  footprint rather than the smaller centering region. A ball center outside
  that footprint or at/below its top plane is latched failure (1 micrometre
  configurable plane tolerance). The sphere radius is read from the model.
- Require both handles and a lifted, table-free tray throughout the sampled
  countdown. Lift means at least 1 mm above the randomized reset height and
  no tray/table contact; the threshold is configurable. Countdown completion
  qualifies success, but native rollout continues until a configurable 30-second
  timeout or failure. Later ball loss revokes success; setting the tray down
  after qualification is allowed if the ball remains on it.
- Change the default minimum threaded-peg load from 50% to 25% of mug weight.
  The user reviewed a released training-demo mug supported about 42% by its
  threaded peg and 56% by the trunk and defined that shared-support pose as
  success. This deliberately changes the scoring criterion and is a major
  protocol version bump.
- Keep the threshold configurable; an explicit `support_weight_fraction=0.5`
  retains the stricter minimum. Empty-handle threading, release and rejection
  of table/gripper/unsupported stack/deep penetration remain unchanged.

- Conveyor pickup records genuine opposing-finger, lifted pickup at every physics
  substep. An object that leaves the belt before pickup or bin placement latches
  terminal failure; pushing an object off the belt cannot count as pickup.
- Grab-from-clutter latches failure immediately when a wrong object enters the
  crate. Released, floor-supported target placement starts a configurable
  three-second settling deadline. Success requires the target to remain inside
  at that deadline. Batched scoring uses each world's actual simulation clock.
- Detect MuJoCo BADQPOS/BADQVEL/BADQACC warnings and time regression at each CPU
  physics step, and stop before recording/scoring an automatic-reset frame.
  Record these episodes as `simulation_error`, with null success, and exclude
  them from model success/reward denominators. Preserve attempted and valid
  trial counts separately. This guard prevents silent continuation; it does
  not establish or repair the numerical trigger of the reported nuts scene.

## 2.0.3

- Search a bounded set of points on the actual peg segment within the
  collision-derived handle x envelope. An enclosed empty handle opening may
  contain the peg tip or an offset portion of the segment rather than the
  vessel-center plane; training replay and closeups confirmed both cases.
- Retain outside-vessel, solid-occupancy, four-ray enclosure, released-contact
  and minimum peg-load requirements. No extrapolation beyond the peg and no
  dense uniform segment sampling are introduced.

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
