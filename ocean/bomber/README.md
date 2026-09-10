# Bomber (solo clearance)

An original Bomberman-inspired PufferLib 5.0 task. Destroy every crate and survive
until all bombs and flames resolve. This implements the solo mechanics and native
integration milestones. A Raylib viewer, tests, scripted baseline, benchmark, and
three-seed training experiment are included. Duel and self-play are not implemented.

## Rules and tick contract

The board is 11 × 11 with a solid border and even-row/even-column pillars.
One player starts at (1,1), with one active bomb allowed, blast range 2, fuse 8,
and a 400-tick finite horizon. Coordinates are zero-based. There are no upgrades,
kicking, hidden information, or opponents.

Each tick:

1. Kill a player on a flame active at tick start.
2. Apply a legal move, or wait for an invalid action. Moving into an old flame kills.
3. Surviving players may plant. The newly planted bomb does not age this tick.
4. Age pre-existing bombs, and queue expired bombs or bombs touching old flames.
5. Resolve a bounded queue, visiting each bomb once. Rays stop at walls, crates,
   and bombs; bomb hits trigger another detonation. All rays use the same terrain
   and bomb occupancy snapshots. Remove destroyed crates together.
6. Apply new flame damage. Old flames expire; new flames remain dangerous during
   this tick and the following tick. Check death, then clearance, then timeout.

A player can leave their bomb but cannot re-enter its tile. Masks enforce physical
legality only: wait is always valid, dangerous moves remain available. NaN, infinity,
fractional, out-of-range, blocked, and unavailable actions safely become wait.
`bomber_tick` is independent of rendering, buffers, RNG, and auto-reset.

Layouts use a per-environment 32-bit LCG (no shared RNG), seeded by the framework's
slot seed XOR `env.map_seed`. Reset advances that stream. The default density is
0.08. Spawn escape arms and a turn are clear; all nonwall cells are connected when
crates are removed, so destruction opens progression. Empty samples receive one
crate at (1,4). Density 0 therefore gives a fixed one-crate introductory map.

## Actions and observations

Actions: 0 wait, 1 up, 2 down, 3 left, 4 right, 5 plant.

Observations contain 849 float32 values, channel-major, with row-major cells:

| Offsets | Meaning |
|---|---|
| 0–120 | Solid walls |
| 121–241 | Crates |
| 242–362 | Player position |
| 363–483 | Opponent position (all zero) |
| 484–604 | Bomb presence |
| 605–725 | Fuse remaining / 8 |
| 726–846 | Flame ticks remaining / 2 |
| 847 | Available bomb capacity (0 or 1) |
| 848 | Remaining episode ticks / max_ticks |

At a decision boundary a new flame has one future dangerous tick, so its channel
value is 0.5. The detonation tick has already executed. All values lie in [0,1].

Clearance earns +1, death −1, timeout zero. Optional crate shaping divides
`shaping_budget` (default 0.2) by initial crate count. Destroying a crate twice is
impossible; simultaneous suicidal destruction can still earn shaping and is logged
as a death. `perf` is success, `score` is crates destroyed, and other logs report
return, length, crate fraction, death/timeout rates, and bombs planted. `n` is last.

On termination, the completed episode is logged before reset; observations and
masks describe the reset state while reward and terminal retain the transition.
Explicit `puf_reset` clears rewards and terminals. Action buffers belong to the
caller; human controls may write them. No simulation allocation occurs in step.

## Build and play

From the PufferLib root, with its normal dependencies installed:

```bash
bash build.sh bomber --cpu --debug
./bomber
bash build.sh bomber
./puffer train
./puffer eval
```

The environment is compiled into the executable. CPU mode is a viewer, not a CPU
trainer. Hold Left Shift with WASD/arrows to move, or Space to plant. Shift alone
waits. The viewer runs at eight ticks/second and displays countdowns and the last
outcome. Rendering does not change game state or RNG; it may set the action buffer.
The renderer compiled successfully but was not interactively inspected in this run.

## Verification and experiments

```bash
bash tests/run_bomber.sh
bash tests/train_bomber.sh ./puffer 10000000
```

The Linux test runner compiles against the real PufferLib interface and Raylib.
It runs ASan/UBSan mechanics and one million paired replay steps, repeats under
C++17, and compares wait, uniformly sampled legal actions, and a scripted planner
on 1,000 fixed initial map seeds (1000000–1000999). The planner uses BFS to reach a useful bomb position and a
nine-tick time/position search to verify escape. Its full-state search is a baseline,
not a policy feature or an action-mask restriction.

Training starts with three independent policy/map seeds (42, 43, 44). The script
saves checkpoints and logs under `build/bomber_experiment`, then runs headless
evaluation with map_seed=1000000 and policy sampling seed 123. Evaluation uses 256
slots and whole rollout batches, so its actual episode count can exceed 1,000.
Those continuing streams differ from the baseline's fixed one-episode-per-map set;
these are preliminary diagnostics, not a controlled final policy comparison.

For a curriculum, train with `--env.crate_density=0` first, then load that checkpoint
with `--base.load_model_path=...` and raise density to 0.08, then 0.2. Use distinct
map seeds for validation. These curriculum stages have not yet been validated.
See [RESULTS.md](RESULTS.md) for measured outcomes and current learning limitations.
