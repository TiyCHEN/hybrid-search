#pragma once

#include "core.h"
#include "data_format.h"
#include "io.h"

void ReadKNN(std::vector<std::vector<uint32_t>>& knns, const std::string& path);

float EuclideanDistanceSquare(const std::vector<float>& a,
                        const std::vector<float>& b) {
    float sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

// https://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector
std::size_t HashVector(const std::vector<float>& v) {
    const char* data = reinterpret_cast<const char*>(v.data());
    std::size_t size = v.size() * sizeof(v[0]);
    std::hash<std::string_view> hash;
    return hash(std::string_view(data, size));
}

void Recall(const std::string& path,
            const std::string& truth_path, QuerySet& query_set) {
    const int query_num = 10000;
    std::vector<std::vector<uint32_t>> knns(query_num);
    std::vector<std::vector<uint32_t>> truth(query_num);
    ReadKNN(knns, path);
    ReadKNN(truth, truth_path);
    const int K = 100;
    const uint32_t N = knns.size();
    int hit = 0;
    for (unsigned i = 0; i < N; ++i) {
        auto &knn = knns[i];
        auto &truth_knn = truth[i];
        std::sort(knn.begin(), knn.end());
        std::sort(truth_knn.begin(), truth_knn.end());
        // std::cout << knn[0] << " " << truth_knn[0] << std::endl;
        std::vector<uint32_t> intersection;
        std::set_intersection(knn.begin(), knn.end(), truth_knn.begin(), truth_knn.end(),
                              std::back_inserter(intersection));

        hit += static_cast<int>(intersection.size());
    }
    float recall = static_cast<float>(hit) / (N * K);
    std::cout << "Overall Recall: " << recall << std::endl;

    for (int i = 0; i < 4; i++) {
      auto cur_query_index = query_set._type_index[i];
      int hit = 0;
      for (auto index : cur_query_index) {
        auto &knn = knns[index];
        auto &truth_knn = truth[index];
        std::sort(knn.begin(), knn.end());
        std::sort(truth_knn.begin(), truth_knn.end());
        std::vector<uint32_t> intersection;
        std::set_intersection(knn.begin(), knn.end(), truth_knn.begin(), truth_knn.end(),
                              std::back_inserter(intersection));
        hit += static_cast<int>(intersection.size());
      }
      float recall = static_cast<float>(hit) / (cur_query_index.size() * K);
      std::cout << "Recall for Query Type " << i << ": " << recall << std::endl;
    }
}

int64_t time_cost(const std::chrono::system_clock::time_point &st, const std::chrono::system_clock::time_point &en) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(en - st).count();
}
