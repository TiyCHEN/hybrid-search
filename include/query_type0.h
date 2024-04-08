#pragma once

#include "core.h"
#include "data_format.h"
#include "util.h"
#include "hnswlib/hnswalg.h"
#include "hnswlib/hnswlib.h"

void solve_query_type0(
    const std::vector<Node>& nodes,
    const std::vector<Query>& queries,
    std::vector<int32_t>& query_indexs,
    std::vector<std::vector<uint32_t>>& knn_results) {
    // build index
    const int M = 16;
    const int ef_construction = 200;
    const int ef_search = 128;
    #if defined(USE_AVX)
        // try use simd16 to get wider data process line.
        base_hnsw::L2Space space(VEC_DIMENSION + ALIGN_SIMD_AVX);
    #else
        base_hnsw::L2Space space(VEC_DIMENSION);
    #endif
    std::unique_ptr<base_hnsw::HierarchicalNSW<float>> single_hnsw = std::make_unique<base_hnsw::HierarchicalNSW<float>>(
                &space, nodes.size(), M, ef_construction);

#pragma omp parallel for schedule(dynamic, NUM_THREAD)
    for (uint32_t i = 0; i < nodes.size(); i++) {
        single_hnsw->addPoint(nodes[i]._vec.data(), nodes[i]._id);
    }
    single_hnsw->setEf(ef_search);

    // solve query
#pragma omp parallel for schedule(dynamic, NUM_THREAD)
    for (uint32_t i = 0; i < query_indexs.size(); i++)  {
        const auto& query = queries[query_indexs[i]];
        const int32_t query_type = query._type;
        const int32_t label = query._label;
        const float l = query._l;
        const float r = query._r;
        const auto& query_vec = query._vec;
        auto& knn = knn_results[query_indexs[i]];

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
}
