// Battleship with a radar. Same hidden fleet and same one-action-per-step loop
// as ocean/battleship, plus a second kind of action: a radar sweep of the 3x3
// block around a square, which reports how many ship cells are inside it but
// not which ones.
//
// Sweeps are free -- they do not count as a turn and score ignores them -- but
// an episode only gets sweep_charges of them. The limit is what makes this a
// game rather than a solved puzzle: the 3x3 count map is a separable box filter
// whose 1D factor (a 3-wide moving sum) has determinant -1 at n=10, so the
// operator is invertible and a full set of 100 sweeps would pin down the board
// exactly, for a free score of 1.0. Even a 9-sweep non-overlapping tiling
// determines 81 cells. So sweep_charges is a difficulty setting, not a
// hyperparameter, and must not be swept.

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "raylib.h"
typedef float obs_t;
#include "pufferenv.h"

#define MAX_GRID 10
#define MAX_CELLS (MAX_GRID * MAX_GRID)
#define MAX_SHIPS 8
// Planes: miss, hit on a floating ship, hit on a sunk ship, radar swept here,
// and that sweep's count (normalized). Then one slot per ship holding its
// length (normalized) while afloat, and last the charges left (normalized).
#define OBS_SIZE (5 * MAX_CELLS + MAX_SHIPS + 1)
// One head, not two. Indices below MAX_CELLS fire at that square, indices above
// sweep centred on it. Heads sample independently, so a separate square head
// and mode head could combine into a pair the mask forbids; one wide head lets
// the mask express the two legality rules exactly.
#define ACT_SIZES {2 * MAX_CELLS}
#define NUM_ATNS 1
#define PUF_STEPS_PER_SEC 4
#define CELL_EMPTY (-1)
#define PANEL_WIDTH 240

// Only use floats.
struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float solved;
    float shots;
    float sweeps;
    float hit_rate;
    float n; // Required as the last field.
};

struct Env {
    Log log;                     // Required field.
    Agent agents[1];
    int tag;
    int boundary_reached;
    int num_agents;
    int grid_size;
    int num_ships;
    int ship_sizes[MAX_SHIPS];
    int ship_hits[MAX_SHIPS];
    // Cells occupied by each ship, filled at placement. Sinking has to move
    // every cell of the ship between observation planes, and this is what lets
    // that cost the ship's length instead of a scan of the board.
    int ship_cell[MAX_SHIPS][MAX_GRID];
    int ship_cells;              // Total occupied cells across the fleet.
    int board_ship[MAX_CELLS];   // CELL_EMPTY, or the index of the ship there.
    unsigned char shot[MAX_CELLS];
    unsigned char swept[MAX_CELLS];      // Radar has been used on this centre.
    unsigned char radar[MAX_CELLS];      // Ship cells in the 3x3 around it.
    int hits;
    int shots;                   // Turns taken. Sweeps are not turns.
    int sweeps;
    int sweep_charges;           // Free sweeps allowed per episode.
    int max_steps;               // A bound on shots, not on actions.
    float hit_reward;
    float miss_reward;
    float sink_reward;
    float win_reward;
    float sweep_reward;
    float episode_return;
    int cell_size;
    int cursor_row;
    int cursor_col;
    unsigned int rng;
};
typedef Env BattleshipRadar;

// The board is stored on the fixed MAX_GRID lattice, so the action space is
// compile-time constant and an action index is also its observation index.
// Smaller boards leave the outer rows and columns empty and masked off.
void init_battleship_radar(BattleshipRadar* env) {
    int n = env->grid_size;
    int restarts = 0;
restart:
    for (int i = 0; i < MAX_CELLS; i++) {
        env->board_ship[i] = CELL_EMPTY;
    }
    for (int s = 0; s < env->num_ships; s++) {
        int len = env->ship_sizes[s];
        for (int attempt = 0; ; attempt++) {
            // A ship the partial layout has boxed in redraws the whole board,
            // which is why this is a goto and not a per-ship retry.
            if (attempt == 128) {
                restarts++;
                assert(restarts < 1000 && "fleet does not fit on the board");
                goto restart;
            }
            int horizontal = rand_r(&env->rng) % 2;
            int row = rand_r(&env->rng) % (horizontal ? n : n - len + 1);
            int col = rand_r(&env->rng) % (horizontal ? n - len + 1 : n);
            int head = row*MAX_GRID + col;
            int step = horizontal ? 1 : MAX_GRID;
            int fits = 1;
            for (int i = 0; i < len; i++) {
                fits &= env->board_ship[head + i*step] == CELL_EMPTY;
            }
            if (!fits) {
                continue;
            }
            for (int i = 0; i < len; i++) {
                env->board_ship[head + i*step] = s;
                env->ship_cell[s][i] = head + i*step;
            }
            break;
        }
    }
    // Every sweep answer is fixed once the fleet is placed: the count is over
    // the true board and ignores what has been fired at, so a repeated sweep of
    // the same centre tells the agent nothing and the mask retires it.
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            int count = 0;
            for (int dr = -1; dr <= 1; dr++) {
                for (int dc = -1; dc <= 1; dc++) {
                    int r = row + dr;
                    int c = col + dc;
                    if (r < 0 || r >= n || c < 0 || c >= n) {
                        continue;
                    }
                    count += env->board_ship[r*MAX_GRID + c] != CELL_EMPTY;
                }
            }
            env->radar[row*MAX_GRID + col] = count;
        }
    }
    memset(env->shot, 0, sizeof(env->shot));
    memset(env->swept, 0, sizeof(env->swept));
    memset(env->ship_hits, 0, sizeof(env->ship_hits));
    env->hits = 0;
    env->shots = 0;
    env->sweeps = 0;
    env->episode_return = 0.0f;
}

// Full rebuild of the observation and mask: a square may be fired at when it is
// on the board and unfired, and swept when it is on the board, unswept and a
// charge is left, which are the same tests the planes below make.
//
// Only reset uses this. A turn changes one square, so puf_step updates the
// buffers incrementally through obs_apply_sweep / obs_apply_shot below, which
// must agree with this function exactly -- ocean/battleship_radar/equivalence.c
// hashes both buffers every step to check that they do.
void compute_observations(BattleshipRadar* env) {
    obs_t* obs = env->agents[0].observations;
    unsigned char* mask = env->agents[0].action_mask;
    int charges = env->sweep_charges - env->sweeps;
    memset(obs, 0, OBS_SIZE * sizeof(obs_t));
    memset(mask, 0, 2 * MAX_CELLS);
    for (int row = 0; row < env->grid_size; row++) {
        for (int col = 0; col < env->grid_size; col++) {
            int cell = row*MAX_GRID + col;
            if (env->swept[cell]) {
                obs[3*MAX_CELLS + cell] = 1.0f;
                obs[4*MAX_CELLS + cell] = (float)env->radar[cell] / 9.0f;
            } else if (charges > 0) {
                mask[MAX_CELLS + cell] = 1;
            }
            if (!env->shot[cell]) {
                mask[cell] = 1;
                continue;
            }
            int ship = env->board_ship[cell];
            if (ship == CELL_EMPTY) {
                obs[cell] = 1.0f;
            } else if (env->ship_hits[ship] == env->ship_sizes[ship]) {
                obs[2*MAX_CELLS + cell] = 1.0f;
            } else {
                obs[MAX_CELLS + cell] = 1.0f;
            }
        }
    }
    for (int s = 0; s < env->num_ships; s++) {
        if (env->ship_hits[s] < env->ship_sizes[s]) {
            obs[5*MAX_CELLS + s] = (float)env->ship_sizes[s] / MAX_GRID;
        }
    }
    obs[5*MAX_CELLS + MAX_SHIPS] = (float)charges / (float)env->sweep_charges;
}

// A sweep reveals one square's count and retires that square from the mask.
static void obs_apply_sweep(BattleshipRadar* env, int cell) {
    obs_t* obs = env->agents[0].observations;
    unsigned char* mask = env->agents[0].action_mask;
    int charges = env->sweep_charges - env->sweeps;
    obs[3*MAX_CELLS + cell] = 1.0f;
    obs[4*MAX_CELLS + cell] = (float)env->radar[cell] / 9.0f;
    mask[MAX_CELLS + cell] = 0;
    if (charges == 0) {
        // Last charge spent, so every remaining sweep becomes illegal at once.
        memset(mask + MAX_CELLS, 0, MAX_CELLS);
    }
    obs[5*MAX_CELLS + MAX_SHIPS] = (float)charges / (float)env->sweep_charges;
}

// A shot retires its square and lands in one of the three outcome planes.
static void obs_apply_shot(BattleshipRadar* env, int cell, int ship, int sunk) {
    obs_t* obs = env->agents[0].observations;
    env->agents[0].action_mask[cell] = 0;
    if (ship == CELL_EMPTY) {
        obs[cell] = 1.0f;
        return;
    }
    if (!sunk) {
        obs[MAX_CELLS + cell] = 1.0f;
        return;
    }
    // Sinking moves the whole ship from the floating-hit plane to the sunk
    // plane, not just the square just fired at, and retires its length slot.
    for (int i = 0; i < env->ship_sizes[ship]; i++) {
        int c = env->ship_cell[ship][i];
        obs[MAX_CELLS + c] = 0.0f;
        obs[2*MAX_CELLS + c] = 1.0f;
    }
    obs[5*MAX_CELLS + ship] = 0.0f;
}

void puf_reset(BattleshipRadar* env) {
    env->agents[0].rewards[0] = 0.0f;
    env->agents[0].terminals[0] = 0.0f;
    init_battleship_radar(env);
    compute_observations(env);
}

// Hold Left Shift + WASD to move the cursor, space to fire, R to sweep. Both
// are refused where the mask would forbid them, so a human plays under the same
// legality rules as the policy.
int battleship_radar_human_controls(BattleshipRadar* env) {
    if (!IsWindowReady() || !IsKeyDown(KEY_LEFT_SHIFT)) {
        return 0;
    }
    int n = env->grid_size;
    if (IsKeyPressed(KEY_UP) || IsKeyPressed(KEY_W)) {
        env->cursor_row = (env->cursor_row - 1 + n) % n;
    }
    if (IsKeyPressed(KEY_DOWN) || IsKeyPressed(KEY_S)) {
        env->cursor_row = (env->cursor_row + 1) % n;
    }
    if (IsKeyPressed(KEY_LEFT) || IsKeyPressed(KEY_A)) {
        env->cursor_col = (env->cursor_col - 1 + n) % n;
    }
    if (IsKeyPressed(KEY_RIGHT) || IsKeyPressed(KEY_D)) {
        env->cursor_col = (env->cursor_col + 1) % n;
    }
    int cell = env->cursor_row*MAX_GRID + env->cursor_col;
    if (IsKeyPressed(KEY_SPACE) && !env->shot[cell]) {
        env->agents[0].actions[0] = cell;
        return 1;
    }
    if (IsKeyPressed(KEY_R) && !env->swept[cell]
            && env->sweeps < env->sweep_charges) {
        env->agents[0].actions[0] = MAX_CELLS + cell;
        return 1;
    }
    return -1;
}

void puf_step(BattleshipRadar* env) {
    if (battleship_radar_human_controls(env) < 0) {
        return;
    }
    int action = (int)env->agents[0].actions[0];
    assert(action >= 0 && action < 2*MAX_CELLS
        && env->agents[0].action_mask[action]
        && "illegal action: the mask forbids this");
    env->agents[0].terminals[0] = 0.0f;
    int cell = action % MAX_CELLS;
    env->cursor_row = cell / MAX_GRID;
    env->cursor_col = cell % MAX_GRID;

    float reward;
    if (action >= MAX_CELLS) {
        env->swept[cell] = 1;
        env->sweeps += 1;
        reward = env->sweep_reward;
        obs_apply_sweep(env, cell);
    } else {
        env->shot[cell] = 1;
        env->shots += 1;
        int ship = env->board_ship[cell];
        reward = env->miss_reward;
        int sunk = 0;
        if (ship != CELL_EMPTY) {
            reward = env->hit_reward;
            env->hits += 1;
            env->ship_hits[ship] += 1;
            if (env->ship_hits[ship] == env->ship_sizes[ship]) {
                reward += env->sink_reward;
                sunk = 1;
            }
        }
        obs_apply_shot(env, cell, ship, sunk);
    }

    int solved = env->hits == env->ship_cells;
    if (solved) {
        reward += env->win_reward;
        env->agents[0].terminals[0] = 1.0f;
    } else if (env->shots >= env->max_steps) {
        env->agents[0].terminals[0] = 1.0f;
    }
    env->agents[0].rewards[0] = reward;
    env->episode_return += reward;

    if (env->agents[0].terminals[0] > 0.0f) {
        // Shots, not actions: sweeps are free, so this is the same
        // shots-to-clear ocean/battleship reports and the two are directly
        // comparable -- radar is only worth having if it scores above ~0.393.
        // score is not episode_return, because the sweep optimizes env/score
        // (hardcoded in eval_loop, sweep.metric only steers the intermediate
        // curve points) and a reward-weighted objective would let a trial win
        // by paying itself more per hit.
        float efficiency = solved
            ? (float)env->ship_cells / (float)env->shots : 0.0f;
        env->log.perf += efficiency;
        env->log.score += efficiency;
        env->log.episode_return += env->episode_return;
        env->log.episode_length += (float)(env->shots + env->sweeps);
        env->log.solved += solved ? 1.0f : 0.0f;
        env->log.shots += (float)env->shots;
        env->log.sweeps += (float)env->sweeps;
        env->log.hit_rate += (float)env->hits / (float)env->shots;
        env->log.n += 1.0f;
        init_battleship_radar(env);
        compute_observations(env);
    }
}

// Raylib client
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color WATER = (Color){0, 52, 52, 255};
const Color SUNK = (Color){96, 0, 0, 255};
const Color RADAR = (Color){0, 96, 64, 255};

void puf_render(BattleshipRadar* env) {
    if (!IsWindowReady()) {
        env->cell_size = 640 / env->grid_size;
        InitWindow(env->grid_size*env->cell_size + PANEL_WIDTH,
            env->grid_size*env->cell_size, "PufferLib Battleship Radar");
        SetTargetFPS(60);
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    battleship_radar_human_controls(env);

    int sz = env->cell_size;
    float gap = sz * 0.08f;
    float inset = gap * 0.5f;
    float tile = sz - gap;
    float roundness = 0.16f;
    int segs = 8;

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);
    for (int row = 0; row < env->grid_size; row++) {
        for (int col = 0; col < env->grid_size; col++) {
            int cell = row*MAX_GRID + col;
            int ship = env->board_ship[cell];
            Color color = env->swept[cell] ? RADAR : WATER;
            if (env->shot[cell] && ship == CELL_EMPTY) {
                color = PUFF_CYAN;
            } else if (env->shot[cell]) {
                color = env->ship_hits[ship] == env->ship_sizes[ship]
                    ? SUNK : PUFF_RED;
            }
            Rectangle rec = {col*sz + inset, row*sz + inset, tile, tile};
            DrawRectangleRounded(rec, roundness, segs, color);
            if (env->swept[cell]) {
                DrawText(TextFormat("%d", env->radar[cell]),
                    col*sz + sz/2 - 5, row*sz + sz/2 - 8, 18, PUFF_WHITE);
            }
        }
    }
    Rectangle cursor = {env->cursor_col*sz + inset, env->cursor_row*sz + inset,
        tile, tile};
    DrawRectangleRoundedLinesEx(cursor, roundness, segs, 3.0f, PUFF_WHITE);

    int panel_x = env->grid_size*sz + 16;
    DrawText(TextFormat("Shots: %d", env->shots), panel_x, 16, 20, PUFF_WHITE);
    DrawText(TextFormat("Radar: %d / %d", env->sweeps, env->sweep_charges),
        panel_x, 44, 20, PUFF_WHITE);
    DrawText(TextFormat("Hits: %d / %d", env->hits, env->ship_cells),
        panel_x, 72, 20, PUFF_WHITE);
    for (int s = 0; s < env->num_ships; s++) {
        int sunk = env->ship_hits[s] == env->ship_sizes[s];
        DrawText(TextFormat("Ship %d (%d): %d", s, env->ship_sizes[s],
            env->ship_hits[s]), panel_x, 112 + 24*s, 20,
            sunk ? SUNK : PUFF_WHITE);
    }
    EndDrawing();
    puf_web_vsync();
}

void puf_close(BattleshipRadar* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "solved", log->solved);
    dict_set(out, "shots", log->shots);
    dict_set(out, "sweeps", log->sweeps);
    dict_set(out, "hit_rate", log->hit_rate);
    dict_set(out, "n", log->n);
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->grid_size = dict_get(kwargs, "grid_size");
    assert(env->grid_size >= 2 && env->grid_size <= MAX_GRID);
    // ship_sizes is a comma separated list in the ini, e.g. 5,4,3,3,2, which
    // src/ini.h parses into DictItem.values. A single ship has no comma, so it
    // arrives as a plain value with len 0.
    DictItem* fleet = dict_find(kwargs, "ship_sizes");
    assert(fleet != NULL && "missing [env] ship_sizes");
    env->num_ships = fleet->len > 0 ? fleet->len : 1;
    assert(env->num_ships <= MAX_SHIPS);
    env->ship_cells = 0;
    for (int s = 0; s < env->num_ships; s++) {
        int len = fleet->len > 0 ? fleet->values[s] : fleet->value;
        assert(len >= 1 && len <= env->grid_size);
        env->ship_sizes[s] = len;
        env->ship_cells += len;
    }
    assert(env->ship_cells <= env->grid_size * env->grid_size);
    env->max_steps = dict_get(kwargs, "max_steps");
    assert(env->max_steps >= env->ship_cells);
    env->sweep_charges = dict_get(kwargs, "sweep_charges");
    // Enough sweeps to tile the board inverts the box filter and hands over the
    // exact layout; see the note at the top of this file.
    assert(env->sweep_charges >= 1 && env->sweep_charges <= 8);
    env->hit_reward = dict_get(kwargs, "hit_reward");
    env->miss_reward = dict_get(kwargs, "miss_reward");
    env->sink_reward = dict_get(kwargs, "sink_reward");
    env->win_reward = dict_get(kwargs, "win_reward");
    env->sweep_reward = dict_get(kwargs, "sweep_reward");
    env->agents[0].policy = 0;
    init_battleship_radar(env);
}
