#pragma once

#include "data_format.h"
#include "hnswlib/rangehnswalg.h"

void SolveQueryType02(
        DataSet& data_set,
        QuerySet& query_set,
        std::vector<std::vector<uint32_t>>& knn_results) {
    // build index
    const int M = 16;
    const int ef_construction = 200;
    const int ef_search = 512 + 256;

    base_hnsw::L2Space space(VEC_DIMENSION);
    std::unique_ptr<base_hnsw::RangeHierarchicalNSW<float>> single_hnsw = std::make_unique<base_hnsw::RangeHierarchicalNSW<float>>(
            &space, data_set.size(), M, ef_construction);

#pragma omp parallel for schedule(dynamic, NUM_THREAD)
    for (uint32_t i = 0; i < data_set.size(); i++) {
        single_hnsw->addPoint(data_set._vecs[i].data(), i, data_set._timestamps[i]);
    }
    single_hnsw->setEf(ef_search);


    // solve query type0
    auto &q0_indexes = query_set._type_index[0];
#pragma omp parallel for schedule(dynamic, NUM_THREAD)
    for (uint32_t i = 0; i < q0_indexes.size(); i++)  {
        const auto& query = query_set._queries[q0_indexes[i]];
        const auto& query_vec = query._vec;
        auto& knn = knn_results[q0_indexes[i]];

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
    auto &q2_indexes = query_set._type_index[2];
#pragma omp parallel for schedule(dynamic, NUM_THREAD)
    for (uint32_t i = 0; i < q2_indexes.size(); i++)  {
        const auto& query = query_set._queries[q2_indexes[i]];
        const float l = query._l;
        const float r = query._r;
        const auto& query_vec = query._vec;
        auto& knn = knn_results[q2_indexes[i]];

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
