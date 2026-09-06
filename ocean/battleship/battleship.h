// Single-player Battleship. A fleet is randomly placed on a hidden board each
// episode; every step the agent fires at one square. Already-fired squares are
// removed by the action mask, so the board always clears and the objective is
// to do it in as few shots as possible.

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "raylib.h"
typedef float obs_t;
#include "pufferenv.h"

#define MAX_GRID 10
#define MAX_CELLS (MAX_GRID * MAX_GRID)
#define MAX_SHIPS 8
// Planes: miss, hit on a floating ship, hit on a sunk ship. Then one slot per
// ship holding its length (normalized) while afloat, 0 once sunk.
#define OBS_SIZE (3 * MAX_CELLS + MAX_SHIPS)
#define ACT_SIZES {MAX_CELLS}
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
    int ship_cells;              // Total occupied cells across the fleet.
    int board_ship[MAX_CELLS];   // CELL_EMPTY, or the index of the ship there.
    unsigned char shot[MAX_CELLS];
    int hits;
    int step_count;
    int max_steps;
    float hit_reward;
    float miss_reward;
    float sink_reward;
    float win_reward;
    float episode_return;
    int cell_size;
    int cursor_row;
    int cursor_col;
    unsigned int rng;
};
typedef Env Battleship;

// The board is stored on the fixed MAX_GRID lattice, so the action space is
// compile-time constant and an action index is also its observation index.
// Smaller boards leave the outer rows and columns empty and masked off.
void init_battleship(Battleship* env) {
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
            }
            break;
        }
    }
    memset(env->shot, 0, sizeof(env->shot));
    memset(env->ship_hits, 0, sizeof(env->ship_hits));
    env->hits = 0;
    env->step_count = 0;
    env->episode_return = 0.0f;
}

// Observations and the mask are one pass: a square is legal exactly when it is
// on the board and unfired, which is the same test the miss/hit planes make.
void compute_observations(Battleship* env) {
    obs_t* obs = env->agents[0].observations;
    unsigned char* mask = env->agents[0].action_mask;
    memset(obs, 0, OBS_SIZE * sizeof(obs_t));
    memset(mask, 0, MAX_CELLS);
    for (int row = 0; row < env->grid_size; row++) {
        for (int col = 0; col < env->grid_size; col++) {
            int cell = row*MAX_GRID + col;
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
            obs[3*MAX_CELLS + s] = (float)env->ship_sizes[s] / MAX_GRID;
        }
    }
}

void puf_reset(Battleship* env) {
    env->agents[0].rewards[0] = 0.0f;
    env->agents[0].terminals[0] = 0.0f;
    init_battleship(env);
    compute_observations(env);
}

// Hold Left Shift + WASD to move the cursor, space to fire. Firing is refused
// on an already-fired square so a human plays under the same legality rule the
// mask gives the policy.
int battleship_human_controls(Battleship* env) {
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
    return -1;
}

void puf_step(Battleship* env) {
    if (battleship_human_controls(env) < 0) {
        return;
    }
    int cell = (int)env->agents[0].actions[0];
    assert(cell >= 0 && cell < MAX_CELLS && env->agents[0].action_mask[cell]
        && "illegal action: the mask forbids this square");
    env->agents[0].terminals[0] = 0.0f;
    env->cursor_row = cell / MAX_GRID;
    env->cursor_col = cell % MAX_GRID;

    env->shot[cell] = 1;
    env->step_count += 1;
    int ship = env->board_ship[cell];
    float reward = env->miss_reward;
    if (ship != CELL_EMPTY) {
        reward = env->hit_reward;
        env->hits += 1;
        env->ship_hits[ship] += 1;
        if (env->ship_hits[ship] == env->ship_sizes[ship]) {
            reward += env->sink_reward;
        }
    }

    int solved = env->hits == env->ship_cells;
    if (solved) {
        reward += env->win_reward;
        env->agents[0].terminals[0] = 1.0f;
    } else if (env->step_count >= env->max_steps) {
        env->agents[0].terminals[0] = 1.0f;
    }
    env->agents[0].rewards[0] = reward;
    env->episode_return += reward;

    if (env->agents[0].terminals[0] > 0.0f) {
        // Perfect play fires exactly ship_cells shots, so perf is 1.0 at best.
        // score is the same quantity, not episode_return: the sweep optimizes
        // env/score (hardcoded in eval_loop, sweep.metric only steers the
        // intermediate curve points), and a reward-weighted objective would
        // let a trial win by inflating its own rewards.
        float efficiency = solved
            ? (float)env->ship_cells / (float)env->step_count : 0.0f;
        env->log.perf += efficiency;
        env->log.score += efficiency;
        env->log.episode_return += env->episode_return;
        env->log.episode_length += (float)env->step_count;
        env->log.solved += solved ? 1.0f : 0.0f;
        env->log.shots += (float)env->step_count;
        env->log.hit_rate += (float)env->hits / (float)env->step_count;
        env->log.n += 1.0f;
        init_battleship(env);
    }
    compute_observations(env);
}

// Raylib client
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color WATER = (Color){0, 52, 52, 255};
const Color SUNK = (Color){96, 0, 0, 255};

void puf_render(Battleship* env) {
    if (!IsWindowReady()) {
        env->cell_size = 640 / env->grid_size;
        InitWindow(env->grid_size*env->cell_size + PANEL_WIDTH,
            env->grid_size*env->cell_size, "PufferLib Battleship");
        SetTargetFPS(60);
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    battleship_human_controls(env);

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
            Color color = WATER;
            if (env->shot[cell] && ship == CELL_EMPTY) {
                color = PUFF_CYAN;
            } else if (env->shot[cell]) {
                color = env->ship_hits[ship] == env->ship_sizes[ship]
                    ? SUNK : PUFF_RED;
            }
            Rectangle rec = {col*sz + inset, row*sz + inset, tile, tile};
            DrawRectangleRounded(rec, roundness, segs, color);
        }
    }
    Rectangle cursor = {env->cursor_col*sz + inset, env->cursor_row*sz + inset,
        tile, tile};
    DrawRectangleRoundedLinesEx(cursor, roundness, segs, 3.0f, PUFF_WHITE);

    int panel_x = env->grid_size*sz + 16;
    DrawText(TextFormat("Shots: %d", env->step_count), panel_x, 16, 20,
        PUFF_WHITE);
    DrawText(TextFormat("Hits: %d / %d", env->hits, env->ship_cells),
        panel_x, 44, 20, PUFF_WHITE);
    for (int s = 0; s < env->num_ships; s++) {
        int sunk = env->ship_hits[s] == env->ship_sizes[s];
        DrawText(TextFormat("Ship %d (%d): %d", s, env->ship_sizes[s],
            env->ship_hits[s]), panel_x, 84 + 24*s, 20,
            sunk ? SUNK : PUFF_WHITE);
    }
    EndDrawing();
    puf_web_vsync();
}

void puf_close(Battleship* env) {
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
    env->hit_reward = dict_get(kwargs, "hit_reward");
    env->miss_reward = dict_get(kwargs, "miss_reward");
    env->sink_reward = dict_get(kwargs, "sink_reward");
    env->win_reward = dict_get(kwargs, "win_reward");
    env->agents[0].policy = 0;
    init_battleship(env);
}
