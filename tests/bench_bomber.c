// Deterministic evaluation baselines; the planner has full state, like the policy.
#include <time.h>
#include "../ocean/bomber/bomber.h"

static int escape(Env state, int depth, unsigned char seen[10][BOMBER_CELLS], int* first) {
    if (!state.alive) return 0;
    if (depth == 9) return 1;
    if (seen[depth][state.pos]) return 0;
    seen[depth][state.pos] = 1;
    for (int action = 0; action <= 4; action++) {
        if (!bomber_legal(&state, action)) continue;
        Env next = state;
        float reward;
        int outcome = bomber_tick(&next, action, &reward);
        if (outcome == BOMBER_DEATH) continue;
        int ignored;
        if (escape(next, depth + 1, seen, &ignored)) { *first = action; return 1; }
    }
    return 0;
}

static int useful_bomb(Env state) {
    int useful = 0;
    for (int d = 1; d <= 4; d++) {
        int p = state.pos;
        for (int k = 0; k < BOMBER_RANGE; k++) {
            p = bomber_neighbor(p, d);
            if (p < 0 || state.terrain[p] == BOMBER_WALL) break;
            if (state.terrain[p] == BOMBER_CRATE) { useful = 1; break; }
        }
    }
    if (!useful) return 0;
    float reward;
    bomber_tick(&state, BOMBER_PLANT, &reward);
    unsigned char seen[10][BOMBER_CELLS] = {{0}};
    int first;
    return escape(state, 0, seen, &first);
}

static int scripted(Env* env) {
    int hazards = bomber_active_bombs(env);
    for (int i = 0; i < BOMBER_CELLS; i++) hazards += env->flames[i];
    if (hazards) {
        unsigned char seen[10][BOMBER_CELLS] = {{0}};
        int first = 0;
        escape(*env, 0, seen, &first);
        return first;
    }
    int queue[BOMBER_CELLS], first[BOMBER_CELLS], seen[BOMBER_CELLS] = {0};
    int head = 0, tail = 1;
    queue[0] = env->pos; first[env->pos] = BOMBER_PLANT; seen[env->pos] = 1;
    while (head < tail) {
        int p = queue[head++];
        Env probe = *env; probe.pos = p;
        if (useful_bomb(probe)) return first[p];
        for (int d = 1; d <= 4; d++) {
            int q = bomber_neighbor(p, d);
            if (q < 0 || seen[q] || env->terrain[q] != BOMBER_EMPTY) continue;
            seen[q] = 1; queue[tail++] = q;
            first[q] = p == env->pos ? d : first[p];
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    int episodes = argc > 1 ? atoi(argv[1]) : 1000;
    if (episodes < 1 || episodes > 100000) return 1;
    float obs[OBS_SIZE], action = 0, reward = 0, terminal = 0;
    unsigned char mask[6];
    for (int mode = 0; mode < 3; mode++) {
        Log sums = {0};
        clock_t start = clock();
        long steps = 0;
        for (int ep = 0; ep < episodes; ep++) {
            Env env = {0}; Dict kwargs = {0};
            env.rng = 1000000u + (unsigned int)ep;
            puf_init(&env, &kwargs);
            env.agents[0].observations = obs; env.agents[0].actions = &action;
            env.agents[0].rewards = &reward; env.agents[0].terminals = &terminal;
            env.agents[0].action_mask = mask;
            puf_reset(&env);
            unsigned int rng = 73u + (unsigned int)ep;
            do {
                int a = 0;
                if (mode == 1) {
                    int choices[6], count = 0;
                    for (int j = 0; j < 6; j++) if (mask[j]) choices[count++] = j;
                    rng = rng * 1664525u + 1013904223u;
                    a = choices[(rng >> 16) % count];
                } else if (mode == 2) a = scripted(&env);
                action = (float)a; puf_step(&env); steps++;
            } while (!terminal);
            sums.perf += env.log.perf; sums.death_rate += env.log.death_rate;
            sums.timeout_rate += env.log.timeout_rate; sums.crate_fraction += env.log.crate_fraction;
        }
        double sec = (double)(clock() - start) / CLOCKS_PER_SEC;
        double p = sums.perf / episodes, n = episodes, z2 = 3.8416;
        double center = (p + z2 / (2*n)) / (1 + z2/n);
        double radius = 1.96 * sqrt(p*(1-p)/n + z2/(4*n*n)) / (1+z2/n);
        printf("%s n=%d success=%.4f 95%%Wilson=[%.4f,%.4f] death=%.4f timeout=%.4f crates=%.4f steps=%ld cpu_seconds=%.3f steps/s=%.0f\n",
            mode == 0 ? "wait" : mode == 1 ? "masked_random" : "scripted",
            episodes, p, center-radius, center+radius, sums.death_rate/n, sums.timeout_rate/n,
            sums.crate_fraction/n, steps, sec, steps/sec);
    }
    Env env = {0}; Dict kwargs = {0};
    env.rng = 42; puf_init(&env, &kwargs);
    env.agents[0].observations = obs; env.agents[0].actions = &action;
    env.agents[0].rewards = &reward; env.agents[0].terminals = &terminal;
    env.agents[0].action_mask = mask; puf_reset(&env);
    unsigned int rng = 73;
    clock_t start = clock();
    const int steps = 10000000;
    double checksum = 0;
    for (int t = 0; t < steps; t++) {
        rng = rng * 1664525u + 1013904223u;
        action = (float)((rng >> 16) % 6);
        puf_step(&env);
        checksum += reward + obs[848];
    }
    double sec = (double)(clock() - start) / CLOCKS_PER_SEC;
    printf("environment_only steps=%d cpu_seconds=%.3f steps/s=%.0f checksum=%.3f\n",
        steps, sec, steps/sec, checksum);
    return 0;
}
