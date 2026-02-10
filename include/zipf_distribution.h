#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <queue>
#include <cstdint>

// SplitMix64: high speed splittable pseudo-random number generator
//  requiring 9 64-bit ALU instructions to generate 64-bits of output
static inline uint64_t splitmix64(uint64_t &x)
{
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// Fast RNG (Xoshiro_256ss): XOR-shift-rotate based
struct Xoshiro_256ss
{
    uint64_t s[4];

    // 4-way split
    explicit Xoshiro_256ss(uint64_t seed = 1)
    {
        s[0] = splitmix64(seed);
        s[1] = splitmix64(seed);
        s[2] = splitmix64(seed);
        s[3] = splitmix64(seed);
    }
    // Rotate left, xoshift, and xorshift
    static inline uint64_t rotl(const uint64_t x, int k)
    {
        return (x << k) | (x >> (64 - k));
    }

    // Generate the next random number
    uint64_t next()
    {
        const uint64_t result = rotl(s[1] * 5ULL, 7) * 9ULL;

        const uint64_t t = s[1] << 17;

        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];

        s[2] ^= t;

        s[3] = rotl(s[3], 45);

        return result;
    }

    // uniform double [0, 1)
    double nextDouble()
    {
        // take top 53 bits -> double in [0, 1)
        return (next() >> 11) * (1.0 / 9007199254740992.0);
    }

    uint32_t nextBounded(uint32_t bound)
    {
        // Use only lower 32 bits for bounded sampling
        uint32_t x = static_cast<uint32_t>(next());
        uint64_t m = static_cast<uint64_t>(x) * bound;
        uint32_t l = static_cast<uint32_t>(m);
        
        if (l < bound) {
            uint32_t t = (0u - bound) % bound;
            while (l < t) {
                x = static_cast<uint32_t>(next());
                m = static_cast<uint64_t>(x) * bound;
                l = static_cast<uint32_t>(m);
            }
        }
        return m >> 32;
    }
};

class ZipfAlias
{
public:
    ZipfAlias(size_t N, double s, uint64_t seed = 12345) : N_(N), s_(s), rng_(seed)
    {
        assert((N > 1) && "Number of distinct items must be more than one.");
        assert((s > 0) && "Skewness must be positive.");

        prob_.resize(N_);
        alias_.resize(N_);
        buildAliasTable();
    }

    // returns k in [1, N]
    uint32_t sample()
    {
        // After prprocessing we build N cols of height 1
        // Column i encodes a probability split: top part belongs to i and bottom part belongs to alias_[i]

        // Each column has equal probability 1/N
        uint32_t col = rng_.nextBounded(static_cast<uint32_t>(N_));
        // Choose a height inside the column [0, 1)
        double u = rng_.nextDouble();

        // height = 1
        // │
        // │  alias[col]   ← if u ≥ prob[col]
        // │
        // ├────────────── prob[col]
        // │
        // │  col          ← if u < prob[col]
        // │
        // └────────────── 0

        //std::cout << "col: " << col << ", u: " << u << std::endl;
        if (col >= N_ || u < 0.0 || u >= 1.0)
        {
            std::cerr << "Invalid column or height." << std::endl;
            return 0;
        }

        uint32_t idx = (u < prob_[col]) ? col : alias_[col];
        return idx + 1; // Convert to 1-based index
    }

private:
    void buildAliasTable()
    {
        // Generalized Harmonic Number, or normalization constant to make prob sum to 1.
        double H_N_s = 0.0;

        for (size_t i = 1; i <= N_; i++)
        {
            H_N_s += 1.0 / std::pow(static_cast<double>(i), static_cast<double>(s_));
        }

        // scaled the probabilities
        std::vector<long double> p(N_, 0.0);
        for (size_t i = 1; i <= N_; i++)
        {
            long double pmf = 1.0 / std::pow(static_cast<long double>(i), static_cast<long double>(s_)) / H_N_s;
            p[i - 1] = pmf * (long double)N_;
        }

        // create alias table
        std::queue<size_t> small, large;

        for (size_t i = 0; i < N_; i++)
        {
            (p[i] < 1.0 ? small : large).push(i);
        }

        while (!small.empty() && !large.empty())
        {
            size_t s = small.front();
            size_t l = large.front();

            small.pop();
            large.pop();

            prob_[s] = p[s];
            alias_[s] = l;

            p[l] = p[l] + p[s] - 1.0;
            (p[l] < 1.0 ? small : large).push(l);
        }

        // remaining items
        while (!small.empty())
        {
            size_t s = small.front();
            small.pop();

            prob_[s] = 1.0f;
            alias_[s] = s;
        }

        while (!large.empty())
        {
            size_t l = large.front();
            large.pop();

            prob_[l] = 1.0f;
            alias_[l] = l;
        }
    }

    size_t N_; // Number of Distinct Items
    double s_; // Skewness

    Xoshiro_256ss rng_;
    std::vector<double> prob_;
    std::vector<size_t> alias_;
};