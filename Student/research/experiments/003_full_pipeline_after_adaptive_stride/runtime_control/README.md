# Experiment 003 - Full Pipeline After Adaptive Stride

## Goal

Check whether the stronger MachineForward model from experiment 002 improves downstream Runtime Control.

## Result

The full pipeline runs, but Runtime Control still fails to choose useful real actions.

Observed runtime output:

* steps: `78`
* total reward: `-143.152`
* action mean: `[0.00154023, -0.00195964]`
* action std: `[0.00699817, 0.00688242]`
* search mean: `0.001529`

## Diagnosis

The failure is now a scale mismatch:

* `runtime_control_10_desired_delta` is much larger.
* `runtime_control_14_machine_candidate_delta` is much smaller.
* `runtime_control_15_chooseaction` therefore selects near-zero actions.

This makes sense after experiment 002: MachineForward was trained with adaptive stride, so Runtime Control now needs to compensate the time-scale mismatch before action search.

## Next Fix

Scale down the desired delta before comparing it to MachineForward candidate deltas:

```text
desired_delta_scaled = desired_delta / stride_mean
```

Initial conservative value:

```text
desired_delta_scale = 1 / 1.912294 ~= 0.52
```

If that is still too large, sweep:

```text
0.5, 0.25, 0.1, 0.05
```

## Decision

Experiment 003 found the next bottleneck. Proceed to experiment 004: Runtime Control desired-delta scaling.
