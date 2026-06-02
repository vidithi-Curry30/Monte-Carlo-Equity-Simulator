#pragma once
#include <cstdint>
#include <cstring>

// xoshiro256++ — period 2^256-1, passes PractRand and BigCrush.
// ~3x faster than std::mt19937 with equivalent statistical quality.
// Reference: Blackman & Vigna (2021), https://prng.di.unimi.it/
class Xoshiro256pp {
public:
    explicit Xoshiro256pp(uint64_t seed) {
        // SplitMix64 to expand the seed into 4 independent state words.
        // Using a single seed directly would leave most state bits zero.
        for (int i = 0; i < 4; ++i) {
            seed += 0x9e3779b97f4a7c15ULL;
            uint64_t z = seed;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            state[i] = z ^ (z >> 31);
        }
    }

    // Returns a uniform uint64 in [0, 2^64).
    inline uint64_t next() {
        const uint64_t result = rotl(state[0] + state[3], 23) + state[0];
        const uint64_t t = state[1] << 17;
        state[2] ^= state[0];
        state[3] ^= state[1];
        state[1] ^= state[2];
        state[0] ^= state[3];
        state[2] ^= t;
        state[3] = rotl(state[3], 45);
        return result;
    }

    // Returns a uniform double in [0, 1).
    inline double nextDouble() {
        // Use top 53 bits for full double mantissa precision.
        return (next() >> 11) * 0x1.0p-53;
    }

    // Box-Muller transform: produces a standard normal N(0,1) sample.
    // Generates two normals per call; caches the second for the next call.
    inline double nextNormal() {
        if (has_spare) {
            has_spare = false;
            return spare;
        }
        double u, v, s;
        do {
            u = nextDouble() * 2.0 - 1.0;
            v = nextDouble() * 2.0 - 1.0;
            s = u * u + v * v;
        } while (s >= 1.0 || s == 0.0);
        const double mul = __builtin_sqrt(-2.0 * __builtin_log(s) / s);
        spare = v * mul;
        has_spare = true;
        return u * mul;
    }

private:
    uint64_t state[4];
    double spare = 0.0;
    bool has_spare = false;

    static inline uint64_t rotl(uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }
};
