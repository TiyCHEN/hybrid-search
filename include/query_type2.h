// single-label filter-knns

#pragma once

#include "core.h"
#include "data_format.h"
#include "util.h"
#include "../third_party/hnswlib/hnswalg.h"
#include "../third_party/hnswlib/hnswlib.h"

// const int HNSW_BUILD_THRASHOLD = 1000;  // TODO: hyperparameter, adjust later
const float SAMPLE_PROPORTION= 0.001;

void solve_query_type2(
    const std::vector<Node>& nodes,
    const std::vector<Query>& queries,
    std::unordered_map<int32_t, std::vector<int32_t>>& data_label_index,
    std::vector<int32_t>& query_indexs,
    std::vector<std::vector<uint32_t>>& knn_results) {
    auto n = nodes.size();
    auto sn = uint32_t(n * SAMPLE_PROPORTION);
    // solve query
    for (auto& query_index : query_indexs) {
        const auto& query = queries[query_index];
        const int32_t query_type = query._type;
        const int32_t label = query._label;
        const float l = query._l;
        const float r = query._r;
        const auto& query_vec = query._vec;
        auto& knn = knn_results[query_index];

        for (auto j = 0; j < sn; j++) {
            if (nodes[j]._timestamp >= l && nodes[j]._timestamp <= r) {
                knn.push_back(j);
                if (knn.size() >= K) {
                    break;
                }
            }
        }

        if (knn.size() < K) {
            auto s = 1;
            while (knn.size() < K) {
                knn.push_back(n -s);
                s++;
            }
        }
    }
};