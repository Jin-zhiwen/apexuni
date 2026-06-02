# LightGlue Approach Lock Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the robot committed to a detected target after strong LightGlue evidence, so weak close-range texture does not immediately drop the system back into exploration.

**Architecture:** Add a lightweight "approach lock" in the Python INSiNav gate. Strict LightGlue geometry still acquires the target, but once acquired, a weaker near-range tracking gate may preserve `PENDING_FAR` or `PENDING_CLOSE` and keep the selected detection cloud alive long enough to finish the close approach. The C++ FSM should remain unchanged unless targeted tests show the Python-side fix is insufficient.

**Tech Stack:** Python, NumPy, ROS, pytest-style unit tests, existing `insinav_stop_gate.py` helpers.

---

### Task 1: Specify the lock behavior in pure helper tests

**Files:**
- Modify: `real_world_test_example/test_insinav_stop_gate.py`
- Test: `real_world_test_example/test_insinav_stop_gate.py`

- [ ] **Step 1: Write failing tests for approach lock state transitions**

```python
def test_approach_lock_latches_after_strong_identity_frame():
    ...

def test_approach_lock_tolerates_a_few_weak_close_frames():
    ...
```

- [ ] **Step 2: Write failing tests for close-range pending behavior under lock**

```python
def test_stop_gate_reports_pending_close_for_locked_weak_close_target():
    ...
```

- [ ] **Step 3: Run the focused Python tests**

Run: `python3 -m pytest -q ApexNav/real_world_test_example/test_insinav_stop_gate.py`
Expected: new tests fail because helper behavior does not exist yet.

### Task 2: Add pure approach-lock helpers

**Files:**
- Modify: `real_world_test_example/insinav_stop_gate.py`
- Test: `real_world_test_example/test_insinav_stop_gate.py`

- [ ] **Step 1: Add a helper that updates approach-lock state from strict and relaxed geometry checks**

```python
def update_lightglue_approach_lock(...):
    ...
```

- [ ] **Step 2: Extend stop-gate status logic to distinguish strict final confirmation from relaxed locked tracking**

```python
def stop_gate_status(..., approach_lock_active=False, tracking_min_match_points=None, ...):
    ...
```

- [ ] **Step 3: Run the focused Python tests again**

Run: `python3 -m pytest -q ApexNav/real_world_test_example/test_insinav_stop_gate.py`
Expected: helper tests pass.

### Task 3: Wire the lock into the real-world INSiNav node

**Files:**
- Modify: `real_world_test_example/real_world_test_insinav.py`
- Modify: `real_world_test_example/config/real_world_test_insinav.yaml`
- Test: `real_world_test_example/test_insinav_stop_gate.py`

- [ ] **Step 1: Store approach-lock state on the node**

```python
self.lightglue_target_locked = False
self.lightglue_target_lost_frames = 0
```

- [ ] **Step 2: Use strict thresholds for acquisition and relaxed thresholds for lock maintenance**

```python
acquire_ok = lightglue_candidate_passes(...)
tracking_ok = lightglue_candidate_passes(...)
self.lightglue_target_locked, self.lightglue_target_lost_frames = update_lightglue_approach_lock(...)
```

- [ ] **Step 3: Preserve the selected target candidate while the lock is active**

```python
if candidate_ok or (idx == selected_idx and self.lightglue_target_locked and tracking_ok):
    ...
```

- [ ] **Step 4: Add YAML knobs for relaxed tracking thresholds and lock patience**

```yaml
approach_min_match_points: 12
approach_min_inlier_points: 4
approach_min_inlier_ratio: 0.15
approach_lock_max_lost_frames: 3
approach_lock_max_distance: 1.6
```

- [ ] **Step 5: Run the focused Python tests**

Run: `python3 -m pytest -q ApexNav/real_world_test_example/test_insinav_stop_gate.py`
Expected: PASS

### Task 4: Verify whether FSM changes are still needed

**Files:**
- Review: `src/planner/exploration_manager/src/exploration_fsm_traj.cpp`
- Review: `src/planner/exploration_manager/src/exploration_fsm_traj_logic.cpp`
- Test: `src/planner/exploration_manager/test/exploration_fsm_traj_logic_test.cpp`

- [ ] **Step 1: Re-check FSM assumptions after the Python fix**

Expected: `PENDING_FAR` and `PENDING_CLOSE` should now persist longer, so existing FSM branches should already favor approach over exploration.

- [ ] **Step 2: Add or adjust C++ tests only if a remaining mismatch is proven**

Run: existing targeted C++ tests if build tooling is available.
Expected: no code change if Python-side behavior is sufficient.
