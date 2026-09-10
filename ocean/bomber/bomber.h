#pragma once
#include <math.h>
#include <stdint.h>
#include <string.h>
typedef float obs_t;
#include "pufferenv.h"

#define ACT_SIZES {6}
#define NUM_ATNS 1
#define BOMBER_SIDE 11
#define BOMBER_CELLS 121
#define OBS_SIZE (7 * BOMBER_CELLS + 2)
#define BOMBER_FUSE 8
#define BOMBER_RANGE 2

enum { BOMBER_WAIT, BOMBER_UP, BOMBER_DOWN, BOMBER_LEFT, BOMBER_RIGHT, BOMBER_PLANT };
enum { BOMBER_EMPTY, BOMBER_WALL, BOMBER_CRATE };
enum { BOMBER_RUNNING, BOMBER_SUCCESS, BOMBER_DEATH, BOMBER_TIMEOUT };

struct Log {
    float perf, score, episode_return, episode_length;
    float crate_fraction, death_rate, timeout_rate, bomb_count;
    float n;
};

struct Env {
    Log log;
    Agent agents[1];
    int tag, boundary_reached, num_agents;
    unsigned int rng;
    unsigned char terrain[BOMBER_CELLS];
    unsigned char bombs[BOMBER_CELLS];
    unsigned char flames[BOMBER_CELLS];
    int pos, tick, initial_crates, crates, planted, alive;
    int max_ticks, last_outcome;
    float crate_density, shaping_budget, episode_return;
};

static unsigned int bomber_random(Env* env) {
    env->rng = env->rng * 1664525u + 1013904223u;
    return env->rng;
}

static int bomber_neighbor(int pos, int action) {
    int r = pos / BOMBER_SIDE, c = pos % BOMBER_SIDE;
    if (action == BOMBER_UP) r--;
    if (action == BOMBER_DOWN) r++;
    if (action == BOMBER_LEFT) c--;
    if (action == BOMBER_RIGHT) c++;
    return r < 0 || r >= BOMBER_SIDE || c < 0 || c >= BOMBER_SIDE ? -1 : r * BOMBER_SIDE + c;
}

static int bomber_active_bombs(const Env* env) {
    int count = 0;
    for (int i = 0; i < BOMBER_CELLS; i++) count += env->bombs[i] != 0;
    return count;
}

static int bomber_legal(const Env* env, int action) {
    if (action == BOMBER_WAIT) return 1;
    if (action == BOMBER_PLANT) return !env->bombs[env->pos] && !bomber_active_bombs(env);
    if (action < BOMBER_UP || action > BOMBER_RIGHT) return 0;
    int dest = bomber_neighbor(env->pos, action);
    return dest >= 0 && env->terrain[dest] == BOMBER_EMPTY && !env->bombs[dest];
}

static void bomber_observe(Env* env) {
    float* obs = env->agents[0].observations;
    memset(obs, 0, OBS_SIZE * sizeof(*obs));
    for (int i = 0; i < BOMBER_CELLS; i++) {
        obs[i] = env->terrain[i] == BOMBER_WALL;
        obs[BOMBER_CELLS + i] = env->terrain[i] == BOMBER_CRATE;
        obs[4 * BOMBER_CELLS + i] = env->bombs[i] != 0;
        obs[5 * BOMBER_CELLS + i] = env->bombs[i] / (float)BOMBER_FUSE;
        obs[6 * BOMBER_CELLS + i] = env->flames[i] / 2.0f;
    }
    obs[2 * BOMBER_CELLS + env->pos] = env->alive;
    obs[7 * BOMBER_CELLS] = bomber_active_bombs(env) == 0;
    obs[7 * BOMBER_CELLS + 1] = (env->max_ticks - env->tick) / (float)env->max_ticks;
    if (env->agents[0].action_mask)
        for (int a = 0; a < 6; a++) env->agents[0].action_mask[a] = (unsigned char)bomber_legal(env, a);
}

void puf_reset(Env* env) {
    memset(env->bombs, 0, sizeof(env->bombs));
    memset(env->flames, 0, sizeof(env->flames));
    env->pos = BOMBER_SIDE + 1;
    env->tick = env->planted = env->crates = 0;
    env->alive = 1;
    env->episode_return = 0;
    for (int r = 0; r < BOMBER_SIDE; r++) {
        for (int c = 0; c < BOMBER_SIDE; c++) {
            int i = r * BOMBER_SIDE + c;
            int wall = r == 0 || c == 0 || r == 10 || c == 10 || (!(r % 2) && !(c % 2));
            // Two three-cell escape arms, including a turn, are always open.
            int safe = (r == 1 && c <= 3) || (c == 1 && r <= 3) || (r == 3 && c <= 3);
            env->terrain[i] = wall ? BOMBER_WALL : BOMBER_EMPTY;
            if (!wall && !safe && (bomber_random(env) >> 8) / 16777216.0f < env->crate_density) {
                env->terrain[i] = BOMBER_CRATE;
                env->crates++;
            }
        }
    }
    // Avoid episodes that succeed without learning to plant a bomb.
    if (!env->crates) {
        env->terrain[BOMBER_SIDE + 4] = BOMBER_CRATE;
        env->crates = 1;
    }
    env->initial_crates = env->crates;
    env->agents[0].rewards[0] = env->agents[0].terminals[0] = 0;
    bomber_observe(env);
}

// Pure simulation: no buffers, RNG, rendering, allocation, or auto-reset.
// Bomb and terrain snapshots make simultaneous explosions order independent.
static int bomber_tick(Env* env, int action, float* reward) {
    unsigned char old_bombs[BOMBER_CELLS], terrain[BOMBER_CELLS];
    unsigned char fresh[BOMBER_CELLS] = {0}, destroyed[BOMBER_CELLS] = {0};
    unsigned char queued[BOMBER_CELLS] = {0};
    int queue[BOMBER_CELLS], head = 0, tail = 0;
    memcpy(old_bombs, env->bombs, sizeof(old_bombs));
    memcpy(terrain, env->terrain, sizeof(terrain));
    *reward = 0;
    env->tick++;
    if (env->flames[env->pos]) env->alive = 0;
    if (!bomber_legal(env, action)) action = BOMBER_WAIT;
    if (env->alive && action >= BOMBER_UP && action <= BOMBER_RIGHT) {
        env->pos = bomber_neighbor(env->pos, action);
        if (env->flames[env->pos]) env->alive = 0;
    }
    if (env->alive && action == BOMBER_PLANT) {
        env->bombs[env->pos] = BOMBER_FUSE;
        env->planted++;
    }
    for (int i = 0; i < BOMBER_CELLS; i++) {
        if (old_bombs[i]) env->bombs[i]--;
        if ((old_bombs[i] && !env->bombs[i]) || (env->bombs[i] && env->flames[i])) {
            queue[tail++] = i;
            queued[i] = 1;
        }
    }
    while (head < tail) {
        int origin = queue[head++];
        fresh[origin] = 1;
        for (int dir = BOMBER_UP; dir <= BOMBER_RIGHT; dir++) {
            int pos = origin;
            for (int distance = 0; distance < BOMBER_RANGE; distance++) {
                pos = bomber_neighbor(pos, dir);
                if (pos < 0 || terrain[pos] == BOMBER_WALL) break;
                fresh[pos] = 1;
                if (terrain[pos] == BOMBER_CRATE) { destroyed[pos] = 1; break; }
                if (old_bombs[pos] || env->bombs[pos]) {
                    if (!queued[pos]) { queued[pos] = 1; queue[tail++] = pos; }
                    break;
                }
            }
        }
    }
    for (int i = 0; i < BOMBER_CELLS; i++) {
        if (queued[i]) env->bombs[i] = 0;
        if (destroyed[i]) {
            env->terrain[i] = BOMBER_EMPTY;
            env->crates--;
            *reward += env->shaping_budget / (float)env->initial_crates;
        }
        // Stored duration 1 means hazardous at the start of the next tick.
        if (env->flames[i]) env->flames[i]--;
        if (fresh[i]) env->flames[i] = 1;
    }
    if (fresh[env->pos]) env->alive = 0;
    int hazards = 0;
    for (int i = 0; i < BOMBER_CELLS; i++) hazards += env->bombs[i] || env->flames[i];
    int outcome = BOMBER_RUNNING;
    if (!env->alive) { *reward -= 1; outcome = BOMBER_DEATH; }
    else if (!env->crates && !hazards) { *reward += 1; outcome = BOMBER_SUCCESS; }
    else if (env->tick >= env->max_ticks) outcome = BOMBER_TIMEOUT;
    env->episode_return += *reward;
    return outcome;
}

void puf_step(Env* env) {
    float raw = env->agents[0].actions[0];
    int action = (raw >= 0 && raw < 6 && floorf(raw) == raw) ? (int)raw : BOMBER_WAIT;
    float reward;
    int outcome = bomber_tick(env, action, &reward);
    if (outcome) {
        float fraction = (env->initial_crates - env->crates) / (float)env->initial_crates;
        env->log.perf += outcome == BOMBER_SUCCESS;
        env->log.score += env->initial_crates - env->crates;
        env->log.episode_return += env->episode_return;
        env->log.episode_length += env->tick;
        env->log.crate_fraction += fraction;
        env->log.death_rate += outcome == BOMBER_DEATH;
        env->log.timeout_rate += outcome == BOMBER_TIMEOUT;
        env->log.bomb_count += env->planted;
        env->log.n++;
        env->last_outcome = outcome;
        puf_reset(env);
    } else bomber_observe(env);
    env->agents[0].rewards[0] = reward;
    env->agents[0].terminals[0] = outcome != BOMBER_RUNNING;
}

static double bomber_option(Dict* kwargs, const char* key, double fallback) {
    DictItem* item = dict_find(kwargs, key);
    return item ? item->value : fallback;
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->agents[0].policy = 0;
    double density = bomber_option(kwargs, "crate_density", 0.08);
    double shaping = bomber_option(kwargs, "shaping_budget", 0.2);
    double limit = bomber_option(kwargs, "max_ticks", 400);
    double seed = bomber_option(kwargs, "map_seed", 0);
    if (!(density >= 0 && density <= 1) || !(shaping >= 0 && shaping <= 1) ||
        !(limit >= 1 && limit <= 1000000) || !(seed >= 0 && seed <= 4294967295.0)) {
        fprintf(stderr, "bomber: invalid environment configuration\n");
        exit(1);
    }
    env->crate_density = (float)density;
    env->shaping_budget = (float)shaping;
    env->max_ticks = (int)limit;
    env->rng ^= (unsigned int)seed;
}

void puf_log(Log* log, Dict* out) {
#define BOMBER_LOG(field) dict_set(out, #field, log->field)
    BOMBER_LOG(perf); BOMBER_LOG(score); BOMBER_LOG(episode_return);
    BOMBER_LOG(episode_length); BOMBER_LOG(crate_fraction); BOMBER_LOG(death_rate);
    BOMBER_LOG(timeout_rate); BOMBER_LOG(bomb_count); BOMBER_LOG(n);
#undef BOMBER_LOG
}

void puf_render(Env* env) {
    if (!IsWindowReady()) { InitWindow(528, 624, "PufferLib Bomber"); SetTargetFPS(8); }
    // Like the native examples, rendering only writes the human action buffer.
    if (IsKeyDown(KEY_LEFT_SHIFT)) {
        int action = BOMBER_WAIT;
        if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W)) action = BOMBER_UP;
        else if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S)) action = BOMBER_DOWN;
        else if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A)) action = BOMBER_LEFT;
        else if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D)) action = BOMBER_RIGHT;
        else if (IsKeyDown(KEY_SPACE)) action = BOMBER_PLANT;
        env->agents[0].actions[0] = (float)action;
    }
    BeginDrawing();
    ClearBackground((Color){18, 24, 32, 255});
    for (int i = 0; i < BOMBER_CELLS; i++) {
        int x = (i % BOMBER_SIDE) * 48, y = (i / BOMBER_SIDE) * 48;
        Color color = (Color){35, 46, 56, 255};
        if (env->terrain[i] == BOMBER_WALL) color = (Color){85, 99, 113, 255};
        if (env->terrain[i] == BOMBER_CRATE) color = (Color){181, 120, 56, 255};
        if (env->flames[i]) color = ORANGE;
        DrawRectangle(x + 2, y + 2, 44, 44, color);
        if (env->bombs[i]) {
            DrawCircle(x + 24, y + 24, 18, BLACK);
            DrawText(TextFormat("%d", env->bombs[i]), x + 18, y + 12, 24, WHITE);
        }
        if (i == env->pos) DrawCircle(x + 24, y + 24, 10, SKYBLUE);
    }
    DrawText(TextFormat("Crates %d   Tick %d/%d", env->crates, env->tick, env->max_ticks), 12, 540, 20, WHITE);
    DrawText("Hold Shift: WASD/arrows move, Space plants", 12, 570, 18, WHITE);
    const char* outcomes[] = {"Ready", "Cleared!", "Died", "Time limit"};
    DrawText(TextFormat("Last episode: %s", outcomes[env->last_outcome]), 12, 598, 18, SKYBLUE);
    EndDrawing();
    puf_web_vsync();
}

void puf_close(Env* env) {
    (void)env;
    if (IsWindowReady()) CloseWindow();
}
