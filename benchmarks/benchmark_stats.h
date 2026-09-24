#ifndef ADD_RMS_NORM_BIAS_BENCHMARK_STATS_H
#define ADD_RMS_NORM_BIAS_BENCHMARK_STATS_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

struct LatencyStats {
    double minimum;
    double p50;
    double p90;
};

inline double NearestRank(const std::vector<double> &sorted, double quantile)
{
    if (sorted.empty()) {
        throw std::invalid_argument("latency samples must not be empty");
    }
    const double rawRank = std::ceil(quantile * sorted.size());
    const size_t rank = static_cast<size_t>(rawRank < 1.0 ? 1.0 : rawRank);
    const size_t index = std::min(rank - 1, sorted.size() - 1);
    return sorted[index];
}

inline LatencyStats ComputeLatencyStats(std::vector<double> samples)
{
    if (samples.empty()) {
        throw std::invalid_argument("latency samples must not be empty");
    }
    for (double sample : samples) {
        if (!std::isfinite(sample) || sample < 0.0) {
            throw std::invalid_argument("latency samples must be finite and nonnegative");
        }
    }
    std::sort(samples.begin(), samples.end());
    return {samples.front(), NearestRank(samples, 0.50),
            NearestRank(samples, 0.90)};
}

inline double ToPerLaunchUs(double elapsedMilliseconds, uint64_t launches)
{
    if (launches == 0) {
        throw std::invalid_argument("launch count must be positive");
    }
    if (!std::isfinite(elapsedMilliseconds) || elapsedMilliseconds < 0.0) {
        throw std::invalid_argument("elapsed time must be finite and nonnegative");
    }
    return elapsedMilliseconds * 1000.0 / static_cast<double>(launches);
}

#endif
