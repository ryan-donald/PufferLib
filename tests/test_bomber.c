#include <assert.h>
#include <time.h>
#include "../ocean/bomber/bomber.h"

typedef struct Fixture {
    Env env;
    float obs[OBS_SIZE], action, reward, terminal;
    unsigned char mask[6];
} Fixture;

static void setup(Fixture* f, unsigned int seed) {
    memset(f, 0, sizeof(*f));
    f->env.rng = seed;
    Dict kwargs = {0};
    puf_init(&f->env, &kwargs);
    f->env.agents[0].observations = f->obs;
    f->env.agents[0].actions = &f->action;
    f->env.agents[0].rewards = &f->reward;
    f->env.agents[0].terminals = &f->terminal;
    f->env.agents[0].action_mask = f->mask;
    puf_reset(&f->env);
}

static void empty(Fixture* f) {
    setup(f, 7);
    for (int i = 0; i < BOMBER_CELLS; i++)
        if (f->env.terrain[i] == BOMBER_CRATE) f->env.terrain[i] = BOMBER_EMPTY;
    // Keep a distant objective so isolated mechanics tests do not finish.
    f->env.terrain[108] = BOMBER_CRATE;
    f->env.crates = f->env.initial_crates = 1;
}

static void step(Fixture* f, float action) { f->action = action; puf_step(&f->env); }

static void check_buffers(Fixture* f) {
    for (int i = 0; i < OBS_SIZE; i++) assert(isfinite(f->obs[i]) && f->obs[i] >= 0 && f->obs[i] <= 1);
    assert(f->mask[0]);
    for (int a = 0; a < 6; a++) assert(f->mask[a] == bomber_legal(&f->env, a));
    assert(isfinite(f->reward));
}

static void test_fuse_and_flames(void) {
    Fixture f; empty(&f);
    step(&f, BOMBER_PLANT);
    assert(f.env.bombs[12] == 8 && !f.mask[BOMBER_PLANT]);
    step(&f, BOMBER_RIGHT);
    assert(f.env.pos == 13 && !f.mask[BOMBER_LEFT]);
    step(&f, BOMBER_LEFT); // Invalid re-entry is wait.
    assert(f.env.pos == 13);
    step(&f, BOMBER_RIGHT);
    step(&f, BOMBER_DOWN); // Pillar-free corner (2,3) is safe.
    for (int i = 0; i < 3; i++) step(&f, BOMBER_WAIT);
    assert(f.env.bombs[12] == 1 && !f.env.flames[12]);
    step(&f, BOMBER_WAIT);
    assert(!f.env.bombs[12] && f.env.flames[12] == 1 && !f.terminal);
    step(&f, BOMBER_WAIT);
    assert(!f.env.flames[12]);
}

static void test_blasts(void) {
    Fixture f; empty(&f);
    f.env.pos = 100;
    f.env.bombs[12] = 1;
    f.env.terrain[13] = BOMBER_CRATE;
    f.env.crates = f.env.initial_crates = 2;
    step(&f, 0);
    assert(f.env.terrain[13] == BOMBER_EMPTY && !f.env.flames[14]);
    assert(!f.env.flames[1] && !f.env.flames[11]);
    assert(fabsf(f.reward - 0.1f) < 1e-6f);
    empty(&f); f.env.pos = 100;
    f.env.bombs[12] = 1; f.env.bombs[14] = 8; f.env.bombs[36] = 8;
    step(&f, 0);
    assert(!f.env.bombs[12] && !f.env.bombs[14] && !f.env.bombs[36]);
    assert(f.env.flames[36] && f.env.flames[38]);
    empty(&f); f.env.pos = 100;
    f.env.bombs[12] = f.env.bombs[14] = 1;
    f.env.terrain[13] = BOMBER_CRATE;
    f.env.crates = f.env.initial_crates = 2;
    step(&f, 0);
    assert(f.env.crates == 1 && fabsf(f.reward - 0.1f) < 1e-6f);
    // The left ray cannot pass through the snapshot crate to trigger (1,1).
    empty(&f); f.env.pos = 100;
    f.env.bombs[12] = 8; f.env.bombs[14] = 1;
    f.env.terrain[13] = BOMBER_CRATE;
    f.env.crates = f.env.initial_crates = 2;
    step(&f, 0);
    assert(f.env.bombs[12] == 7);
    // A perpendicular earlier ray must not open a crate for a later ray.
    empty(&f); f.env.pos = 100;
    f.env.bombs[14] = f.env.bombs[35] = 1;
    f.env.terrain[36] = BOMBER_CRATE;
    f.env.crates = f.env.initial_crates = 2;
    step(&f, 0);
    assert(!f.env.flames[37] && f.env.crates == 1);
    // Existing flames trigger a bomb even with a long fuse.
    empty(&f); f.env.pos = 100;
    f.env.bombs[14] = 8; f.env.flames[14] = 1;
    step(&f, 0);
    assert(!f.env.bombs[14] && f.env.flames[36]);
    // Dense cyclic chain visits each bomb at most once.
    empty(&f); f.env.pos = 100;
    for (int i = 0; i < BOMBER_CELLS; i++)
        if (!f.env.terrain[i]) f.env.bombs[i] = 8;
    f.env.bombs[12] = 1;
    float reward;
    bomber_tick(&f.env, 0, &reward);
    assert(!bomber_active_bombs(&f.env));
}

static void test_terminals(void) {
    Fixture f; empty(&f);
    f.env.flames[12] = 1;
    step(&f, BOMBER_RIGHT); // Cannot flee an already burning cell.
    assert(f.terminal == 1 && f.reward == -1 && f.env.tick == 0);
    assert(f.env.log.death_rate == 1 && f.env.last_outcome == BOMBER_DEATH);
    step(&f, 0); assert(!f.terminal && f.reward == 0);
    empty(&f); f.env.flames[13] = 1;
    step(&f, BOMBER_RIGHT); assert(f.terminal && f.reward == -1);
    empty(&f); f.env.max_ticks = 1;
    step(&f, 0);
    assert(f.terminal && f.reward == 0 && f.env.log.timeout_rate == 1 && f.env.tick == 0);
    empty(&f); f.env.crates = 0; f.env.terrain[108] = 0; f.env.flames[36] = 1;
    step(&f, 0);
    assert(f.terminal && f.reward == 1 && f.env.log.perf == 1);
    empty(&f); f.env.crates = 0; f.env.terrain[108] = 0; f.env.bombs[36] = 2;
    step(&f, 0); assert(!f.terminal);
    step(&f, 0); assert(!f.terminal);
    step(&f, 0); assert(f.terminal && f.reward == 1);
    Dict out = {0}; puf_log(&f.env.log, &out);
    assert(dict_get(&out, "perf") == 1 && dict_get(&out, "n") == 1);
    free(out.items);
}

static void test_maps_and_replay(void) {
    Fixture a, b;
    for (unsigned int seed = 0; seed < 1000; seed++) {
        setup(&a, seed);
        assert(!a.env.terrain[12] && !a.env.terrain[13] && !a.env.terrain[14]);
        assert(!a.env.terrain[23] && !a.env.terrain[34] && !a.env.terrain[35] && !a.env.terrain[36]);
        // Every nonwall tile is connected when destructible terrain is removed.
        int seen[BOMBER_CELLS] = {0}, queue[BOMBER_CELLS], head = 0, tail = 1;
        queue[0] = 12; seen[12] = 1;
        while (head < tail) {
            int p = queue[head++];
            for (int d = 1; d <= 4; d++) {
                int q = bomber_neighbor(p, d);
                if (q >= 0 && !seen[q] && a.env.terrain[q] != BOMBER_WALL) {
                    seen[q] = 1; queue[tail++] = q;
                }
            }
        }
        for (int i = 0; i < BOMBER_CELLS; i++) assert(a.env.terrain[i] == BOMBER_WALL || seen[i]);
    }
    setup(&a, 42); setup(&b, 42);
    unsigned int rng = 1;
    for (int t = 0; t < 1000000; t++) {
        rng = rng * 1664525u + 1013904223u;
        float action = (float)((rng >> 16) % 9) - 1;
        memset(a.obs, 0xff, sizeof(a.obs)); memset(a.mask, 0xff, sizeof(a.mask));
        step(&a, action); step(&b, action);
        check_buffers(&a);
        assert(memcmp(a.obs, b.obs, sizeof(a.obs)) == 0);
        assert(memcmp(a.env.bombs, b.env.bombs, sizeof(a.env.bombs)) == 0);
        assert(a.reward == b.reward && a.terminal == b.terminal && a.env.rng == b.env.rng);
    }
    empty(&a);
    float invalid[] = {NAN, INFINITY, -INFINITY, 1e30f, -3, 2.5f};
    for (unsigned int i = 0; i < sizeof(invalid) / sizeof(*invalid); i++) {
        step(&a, invalid[i]); assert(a.env.pos == 12);
    }
    a.reward = a.terminal = 99; puf_reset(&a.env);
    assert(a.reward == 0 && a.terminal == 0); check_buffers(&a);
}

int main(void) {
    test_fuse_and_flames(); test_blasts(); test_terminals(); test_maps_and_replay();
    puts("Bomber: all mechanics, buffers, 1000 maps, and 1M replay steps passed");
    return 0;
}
