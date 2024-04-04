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