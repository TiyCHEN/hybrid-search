#pragma once

#include "data_format.h"
#include "hnswlib/rangehnswalg.h"

void solve_query_type02(
        const std::vector<Node>& nodes,
        const std::vector<Query>& queries,
        std::vector<std::vector<int32_t>>& query_indexes,
        std::vector<std::vector<uint32_t>>& knn_results) {
    // build index
    const int M = 16;
    const int ef_construction = 200;
    const int ef_search = 128;

    base_hnsw::L2Space space(VEC_DIMENSION);
    std::unique_ptr<base_hnsw::RangeHierarchicalNSW<float>> single_hnsw = std::make_unique<base_hnsw::RangeHierarchicalNSW<float>>(
            &space, nodes.size(), M, ef_construction);

#pragma omp parallel for schedule(dynamic, NUM_THREAD)
    for (uint32_t i = 0; i < nodes.size(); i++) {
        single_hnsw->addPoint(nodes[i]._vec.data(), nodes[i]._id, nodes[i]._timestamp);
    }
    single_hnsw->setEf(ef_search);


    // solve query type0
    auto &query_type0_indexes = query_indexes[0];
#pragma omp parallel for schedule(dynamic, NUM_THREAD)
    for (uint32_t i = 0; i < query_type0_indexes.size(); i++)  {
        const auto& query = queries[query_type0_indexes[i]];
        const int32_t query_type = query._type;
        const int32_t label = query._label;
        const float l = query._l;
        const float r = query._r;
        const auto& query_vec = query._vec;
        auto& knn = knn_results[query_type0_indexes[i]];

        std::priority_queue<std::pair<float, base_hnsw::labeltype>> result;
        result = single_hnsw->searchKnn(query_vec.data(), 100, 0, 1);

        while (knn.size() < K) {
            if (result.empty()) {
                break;
            }
            knn.push_back(result.top().second);
            result.pop();
        }
    }

    // solve query type2
    auto &query_type2_indexes = query_indexes[2];
#pragma omp parallel for schedule(dynamic, NUM_THREAD)
    for (uint32_t i = 0; i < query_type2_indexes.size(); i++)  {
        const auto& query = queries[query_type2_indexes[i]];
        const int32_t query_type = query._type;
        const int32_t label = query._label;
        const float l = query._l;
        const float r = query._r;
        const auto& query_vec = query._vec;
        auto& knn = knn_results[query_type2_indexes[i]];

        std::priority_queue<std::pair<float, base_hnsw::labeltype>> result;
        result = single_hnsw->searchKnn(query_vec.data(), 100, l, r);

        while (knn.size() < K) {
            if (result.empty()) {
                break;
            }
            knn.push_back(result.top().second);
            result.pop();
        }
    }
}
