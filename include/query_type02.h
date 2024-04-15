#pragma once

#include "data_format.h"
#include "hnswlib/rangehnswalg.h"

void SolveQueryType02(
        DataSet& data_set,
        QuerySet& query_set,
        std::vector<std::vector<uint32_t>>& knn_results) {
    // build index
    base_hnsw::L2Space space(VEC_DIMENSION);
    std::unique_ptr<base_hnsw::RangeHierarchicalNSW<float>> single_hnsw = std::make_unique<base_hnsw::RangeHierarchicalNSW<float>>(
            &space, data_set.size(), M_Q02, EF_CONSTRUCTION_Q02);
    auto s_index02 = std::chrono::system_clock::now();
#pragma omp parallel for schedule(dynamic, CHUNK_SIZE)
    for (uint32_t i = 0; i < data_set.size(); i++) {
        single_hnsw->addPoint(data_set._vecs[i].data(), i, data_set._timestamps[i]);
    }
    auto e_index02 = std::chrono::system_clock::now();
    std::cout << "build index 02 cost: " << time_cost(s_index02, e_index02) << " (ms)\n";

    // solve query type0 (ANN)
    single_hnsw->setEf(EF_SEARCH_Q0);
    auto s_q0 = std::chrono::system_clock::now();
    auto &q0_indexes = query_set._type_index[0];
#pragma omp parallel for schedule(dynamic, CHUNK_SIZE)
    for (uint32_t i = 0; i < q0_indexes.size(); i++)  {
        const auto& query = query_set._queries[q0_indexes[i]];
        const auto& query_vec = query._vec;
        auto& knn = knn_results[q0_indexes[i]];
        std::priority_queue<std::pair<float, base_hnsw::labeltype>> result;
        result = single_hnsw->searchKnn(query_vec.data(), 100, 0, 1);
        while (knn.size() < K) {
            if (result.empty()) {
                knn.push_back(0);
                continue;
            }
            knn.push_back(result.top().second);
            result.pop();
        }
    }
    auto e_q0 = std::chrono::system_clock::now();
    std::cout << "search query 0 cost: " << time_cost(s_q0, e_q0) << " (ms)\n";

    // solve query type2 (Range-ANN)
    single_hnsw->setEf(EF_SEARCH_Q2);
    auto s_q2 = std::chrono::system_clock::now();
    auto &q2_indexes = query_set._type_index[2];
    std::vector<int32_t> data_time_index(data_set.size());
    std::iota(data_time_index.begin(), data_time_index.end(), 0);
    std::sort(data_time_index.begin(), data_time_index.end(), [&](const auto lhs, const auto rhs) {
        return data_set._timestamps[lhs] < data_set._timestamps[rhs];
    });
#pragma omp parallel for schedule(dynamic, CHUNK_SIZE)
    for (uint32_t i = 0; i < q2_indexes.size(); i++)  {
        const auto& query = query_set._queries[q2_indexes[i]];
        const float l = query._l;
        const float r = query._r;
        const auto& query_vec = query._vec;
        auto& knn = knn_results[q2_indexes[i]];

        int st_pos = std::lower_bound(data_time_index.begin(), data_time_index.end(), l, [&](const auto x, const float ti) {
            return data_set._timestamps[x] < ti;
        }) - data_time_index.begin();
        int en_pos = std::lower_bound(data_time_index.begin(), data_time_index.end(), r, [&](const auto x, const float ti) {
            return data_set._timestamps[x] <= ti;
        }) - data_time_index.begin();
        int range_cnt = en_pos - st_pos;

        std::priority_queue<std::pair<float, base_hnsw::labeltype>> result;
        if (range_cnt <= RANGE_BF_THRASHOLD_Q2) {
            for (int j = st_pos; j < en_pos; ++j) {
                auto id = data_time_index[j];
                #if defined(USE_AVX)
                    float dist = base_hnsw::HybridSimd(data_set._vecs[id].data(),query_vec.data(),&VEC_DIMENSION);
                #else
                    float dist = EuclideanDistanceSquare(data_set._vecs[id], query_vec);
                #endif
                result.push(std::make_pair(-dist, id));
            }
        } else {
            result = single_hnsw->searchKnn(query_vec.data(), 100, l, r);
        }

        while (knn.size() < K) {
            if (result.empty()) {
                knn.push_back(0);
                continue;
            }
            knn.push_back(result.top().second);
            result.pop();
        }
    }
    auto e_q2 = std::chrono::system_clock::now();
    std::cout << "search query 2 cost: " << time_cost(s_q2, e_q2) << " (ms)\n";
}
