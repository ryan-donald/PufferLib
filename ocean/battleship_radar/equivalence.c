// Equivalence and throughput harness for ocean/battleship_radar.
//
// puf_step updates the observation and mask incrementally; compute_observations
// rebuilds them from scratch on reset. The two must agree exactly. This drives
// one env with a deterministic uniform-over-legal action stream and FNV-1a
// hashes both the game state and the observation/mask bytes every step.
//
// Hashing the observation is the point. A mask-driven driver never reads the
// observation, so a wrong plane leaves the game hash untouched and would pass
// silently.
//
// Build (from the repo root):
//   gcc -O2 -march=native -std=gnu11 -D_GNU_SOURCE \
//       -I. -Isrc -Iocean/battleship_radar -Iraylib-5.5_linux_amd64/include \
//       -DPLATFORM_DESKTOP \
//       -DENV_HEADER='"ocean/battleship_radar/battleship_radar.h"' \
//       ocean/battleship_radar/equivalence.c \
//       -Lraylib-5.5_linux_amd64/lib -l:libraylib.a -lm -lpthread -ldl \
//       -o /tmp/bsr_equiv
//
// Run:
//   /tmp/bsr_equiv 20000000          # hashes + throughput
//   /tmp/bsr_equiv 10000000 time     # throughput only, hashing off
//   /tmp/bsr_equiv 10000000 nostep   # driver overhead alone
//   /tmp/bsr_equiv 3000000  reset    # cost of one episode boundary
//
// Point ENV_HEADER at a pre-change copy of the header to compare. Verified
// 2026-09-06 at 20M steps / 199211 episodes, delta vs full rebuild:
//   game_hash 6f3fff8db416aa59   obs_hash 7597f9d54250dcd8   (identical)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include ENV_HEADER

static unsigned long long fnv(unsigned long long h, const void* p, size_t n) {
    const unsigned char* b = (const unsigned char*)p;
    for (size_t i = 0; i < n; i++) {
        h = (h ^ b[i]) * 1099511628211ULL;
    }
    return h;
}

// Driver RNG, kept separate from env->rng so the action stream does not depend
// on how many draws the env itself makes.
static unsigned int drng = 12345;
static unsigned int next_rand(void) {
    drng = drng * 1664525u + 1013904223u;
    return drng >> 8;
}

int main(int argc, char** argv) {
    long steps = (argc > 1) ? atol(argv[1]) : 20000000L;
    // Hashing FNV over ~2900 bytes a step costs more than the env does, so
    // timing runs turn it off. Correctness runs leave it on.
    int reset_only = (argc > 2 && strcmp(argv[2], "reset") == 0);
    int no_step = (argc > 2 && strcmp(argv[2], "nostep") == 0);
    int do_hash = !no_step && !(argc > 2 && strcmp(argv[2], "time") == 0);

    Dict kwargs = {0};
    kwargs.name = (char*)"battleship_radar";
    dict_set(&kwargs, "grid_size", 10);
    dict_set(&kwargs, "max_steps", 100);
    dict_set(&kwargs, "sweep_charges", 5);
    dict_set(&kwargs, "hit_reward", 1.0);
    dict_set(&kwargs, "miss_reward", -0.451693177);
    dict_set(&kwargs, "sink_reward", 1.87339783);
    dict_set(&kwargs, "win_reward", 3.94480491);
    dict_set(&kwargs, "sweep_reward", 0.0668581426);
    // Comma list: ini.h fills values[]/len for these, dict_set does not.
    DictItem* fleet = dict_item(&kwargs, "ship_sizes");
    dict_item_clear(fleet);
    static double sizes[5] = {5, 4, 3, 3, 2};
    fleet->values = sizes;
    fleet->len = 5;
    fleet->value = 5;

    Env* env = (Env*)calloc(1, sizeof(Env));
    obs_t* obs = (obs_t*)calloc(OBS_SIZE, sizeof(obs_t));
    float actions[1] = {0};
    float rewards[1] = {0};
    float terminals[1] = {0};
    unsigned char* mask = (unsigned char*)calloc(2 * MAX_CELLS, 1);
    env->agents[0].observations = obs;
    env->agents[0].actions = actions;
    env->agents[0].rewards = rewards;
    env->agents[0].terminals = terminals;
    env->agents[0].action_mask = mask;
    env->rng = 7;                      // pufferl.cu:1005 seeds this per env index

    puf_init(env, &kwargs);
    puf_reset(env);

    unsigned long long game_hash = 1469598103934665603ULL;
    unsigned long long obs_hash = 1469598103934665603ULL;
    long legal_total = 0;
    long episodes = 0;

    struct timespec t0, t1;
    if (reset_only) {
        // Cost of one episode boundary: fleet placement, the radar precompute,
        // and the full observation rebuild.
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (long s = 0; s < steps; s++) {
            init_battleship_radar(env);
            compute_observations(env);
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double rs = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
        printf("resets     %ld\n", steps);
        printf("ns_per_reset %.1f\n", 1e9 * rs / (double)steps);
        return 0;
    }
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (long s = 0; s < steps; s++) {
        int legal[2 * MAX_CELLS];
        int n = 0;
        for (int a = 0; a < 2 * MAX_CELLS; a++) {
            if (mask[a]) {
                legal[n++] = a;
            }
        }
        if (n == 0) {
            fprintf(stderr, "no legal action at step %ld\n", s);
            return 1;
        }
        legal_total += n;
        actions[0] = (float)legal[next_rand() % (unsigned int)n];

        if (!no_step) {
            puf_step(env);
        }

        if (terminals[0] > 0.0f) {
            episodes++;
        }
        if (!do_hash) {
            continue;
        }
        game_hash = fnv(game_hash, env->board_ship, sizeof(env->board_ship));
        game_hash = fnv(game_hash, env->shot, sizeof(env->shot));
        game_hash = fnv(game_hash, env->swept, sizeof(env->swept));
        game_hash = fnv(game_hash, env->radar, sizeof(env->radar));
        game_hash = fnv(game_hash, env->ship_hits, sizeof(env->ship_hits));
        game_hash = fnv(game_hash, &env->hits, sizeof(env->hits));
        game_hash = fnv(game_hash, &env->shots, sizeof(env->shots));
        game_hash = fnv(game_hash, &env->sweeps, sizeof(env->sweeps));
        game_hash = fnv(game_hash, rewards, sizeof(rewards));
        game_hash = fnv(game_hash, terminals, sizeof(terminals));
        obs_hash = fnv(obs_hash, obs, OBS_SIZE * sizeof(obs_t));
        obs_hash = fnv(obs_hash, mask, 2 * MAX_CELLS);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double secs = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
    printf("steps      %ld\n", steps);
    printf("episodes   %ld\n", episodes);
    printf("mean_legal %.3f\n", (double)legal_total / (double)steps);
    printf("game_hash  %016llx\n", game_hash);
    printf("obs_hash   %016llx\n", obs_hash);
    printf("seconds    %.3f\n", secs);
    printf("sps        %.0f\n", steps / secs);
    return 0;
}
