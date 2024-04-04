#pragma once

#include "core.h"
#include "data_format.h"
#include "util.h"
#include "../third_party/hnswlib/hnswalg.h"
#include "../third_party/hnswlib/hnswlib.h"

void solve_query_type0(
    const std::vector<Node>& nodes,
    const std::vector<Query>& queries,
    std::vector<int32_t>& query_indexs,
    std::vector<std::vector<uint32_t>>& knn_results) {
    // build index
    int M = 16;
    int ef_construction = 200;
    int ef_search = 1024;
    base_hnsw::L2Space space(VEC_DIMENSION);
    std::unique_ptr<base_hnsw::HierarchicalNSW<float>> single_hnsw = std::make_unique<base_hnsw::HierarchicalNSW<float>>(
                &space, nodes.size(), M, ef_construction);;
    for (uint32_t i = 0; i < nodes.size(); i++) {
      single_hnsw->addPoint(nodes[i]._vec.data(), i);
      single_hnsw->setEf(ef_search);
    }

    // solve query
    for (auto& query_index : query_indexs) {
        const auto& query = queries[query_index];
        const int32_t query_type = query._type;
        const int32_t label = query._label;
        const float l = query._l;
        const float r = query._r;
        const auto& query_vec = query._vec;
        auto& knn = knn_results[query_index];

        std::priority_queue<std::pair<float, base_hnsw::labeltype>> result;
        result = single_hnsw->searchKnn(query_vec.data(), 100);

        while (knn.size() < K) {
            if (result.empty()) {
              break;
            }
            knn.push_back(result.top().second);
            result.pop();
        }
    }
};