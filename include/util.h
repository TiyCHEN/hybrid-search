#pragma once

#include "core.h"

float EuclideanDistance(const std::vector<float>& a,
                        const std::vector<float>& b) {
    float sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

int64_t time_cost(const std::chrono::system_clock::time_point &st, const std::chrono::system_clock::time_point &en) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(en - st).count();
}
