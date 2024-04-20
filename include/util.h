#pragma once

#include "core.h"

float EuclideanDistanceSquare(const std::vector<float>& a,
                        const std::vector<float>& b) {
    float sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

std::size_t HashVector(const std::vector<float>& v) {
    const char* data = reinterpret_cast<const char*>(v.data());
    std::size_t size = v.size() * sizeof(v[0]);
    std::hash<std::string_view> hash;
    return hash(std::string_view(data, size));
}

int64_t time_cost(const std::chrono::system_clock::time_point &st, const std::chrono::system_clock::time_point &en) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(en - st).count();
}
