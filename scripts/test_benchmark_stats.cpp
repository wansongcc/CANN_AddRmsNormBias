#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "../benchmarks/benchmark_stats.h"

int main()
{
    const auto stats = ComputeLatencyStats({4.0, 1.0, 3.0, 2.0, 100.0});
    assert(std::fabs(stats.minimum - 1.0) < 1e-12);
    assert(std::fabs(stats.p50 - 3.0) < 1e-12);
    assert(std::fabs(stats.p90 - 100.0) < 1e-12);
    assert(std::fabs(ToPerLaunchUs(0.001, 1000) - 0.001) < 1e-12);
    bool threw = false;
    try {
        (void)ToPerLaunchUs(1.0, 0);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);
}
