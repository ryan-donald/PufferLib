#include <time.h>
#include "battleship.h"

// Fires uniformly at random among the legal (unfired) squares.
int random_legal_action(Battleship* env) {
    int legal[MAX_CELLS];
    int n = 0;
    for (int i = 0; i < MAX_CELLS; i++) {
        if (env->agents[0].action_mask[i]) {
            legal[n++] = i;
        }
    }
    return legal[rand_r(&env->rng) % n];
}

int main(void) {
    Battleship env = {
        .grid_size = 10,
        .num_ships = 5,
        .ship_sizes = {5, 4, 3, 3, 2},
        .ship_cells = 17,
        .max_steps = 100,
        .hit_reward = 1.0f,
        .miss_reward = -0.1f,
        .sink_reward = 1.0f,
        .win_reward = 5.0f,
        .rng = (unsigned)time(NULL),
    };
    env.agents[0].observations = calloc(OBS_SIZE, sizeof(obs_t));
    env.agents[0].actions = calloc(1, sizeof(float));
    env.agents[0].rewards = calloc(1, sizeof(float));
    env.agents[0].terminals = calloc(1, sizeof(float));
    env.agents[0].action_mask = calloc(MAX_CELLS, sizeof(unsigned char));

    puf_reset(&env);
    // Opens the window: raylib's WindowShouldClose is true until one exists.
    puf_render(&env);
    while (!WindowShouldClose()) {
        env.agents[0].actions[0] = random_legal_action(&env);
        puf_step(&env);
        puf_render(&env);
    }
    puf_close(&env);
    return 0;
}
