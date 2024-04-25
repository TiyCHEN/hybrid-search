#pragma once

#include "core.h"
#include "util.h"
#include "data_format.h"
#include "rangehnswalg.h"

void SolveQueryType02(
    base_hnsw::L2Space& space,
    DataSet& data_set,
    QuerySet& query_set,
    std::vector<std::vector<uint32_t>>& knn_results) {
    // build index02
    auto s_index02 = std::chrono::system_clock::now();
    std::vector<int32_t> data_time_index(data_set.size());
    std::iota(data_time_index.begin(), data_time_index.end(), 0);
    std::sort(data_time_index.begin(), data_time_index.end(), [&](const auto lhs, const auto rhs) {
        return data_set._timestamps[lhs] < data_set._timestamps[rhs];
    });
    auto cur_hnsw = std::make_unique<base_hnsw::RangeHierarchicalNSW<float>>(&space, data_set.size(), M_Q02, EF_CONSTRUCTION_Q02);
    std::vector<std::unique_ptr<base_hnsw::RangeHierarchicalNSW<float>>> partial_hnsw;
    std::vector<std::pair<int, int>> partial_range;
    const int data_size = data_set.size();
    const int data_half = data_set.size() / 2;
    for (int i = 0, st_pos = 0; i < SEGEMENTS_Q2.size(); ++i) {
        int en_pos = data_half * SEGEMENTS_Q2[i];
        #pragma omp parallel for schedule(dynamic, CHUNK_SIZE)
        for (int j = st_pos; j < en_pos; ++j) {
            // it's ok since data_size % 2 == 0
            cur_hnsw->addPoint(data_set._vecs[data_time_index[j + data_half]].data(), data_time_index[j + data_half],
                                data_set._timestamps[data_time_index[j + data_half]]);
            cur_hnsw->addPoint(data_set._vecs[data_time_index[data_half - 1 - j]].data(), data_time_index[data_half - 1 - j],
                                data_set._timestamps[data_time_index[data_half - 1 - j]]);
        }
        partial_range.emplace_back(data_half - en_pos, data_half + en_pos - 1);
        st_pos = en_pos;
        if (en_pos != data_half) {
            // auto st_copy = std::chrono::system_clock::now();
            partial_hnsw.push_back(std::move(std::make_unique<base_hnsw::RangeHierarchicalNSW<float>>(*cur_hnsw.get())));
            // auto en_copy = std::chrono::system_clock::now();
            // std::cout << "copy cost: " << time_cost(st_copy, en_copy) << " (ms)\n";
        } else {
            partial_hnsw.push_back(std::move(cur_hnsw));
        }
    }
    auto e_index02 = std::chrono::system_clock::now();
    std::cout << "I02: " << time_cost(s_index02, e_index02) << " (ms)\n";

    // solve query type2 (Range-ANN)
    auto s_q2 = std::chrono::system_clock::now();
    auto &q2_indexes = query_set._type_index[2];
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
        if (range_cnt < RANGE_BF_THRESHOLD_Q2) {
            for (int j = st_pos; j < en_pos; ++j) {
                auto id = data_time_index[j];
                #if defined(USE_AVX)
                    float dist = space.get_dist_func()(data_set._vecs[id].data(), query_vec.data(), &VEC_DIMENSION);
                #else
                    float dist = EuclideanDistanceSquare(data_set._vecs[id], query_vec);
                #endif
                result.push(std::make_pair(dist, id));
                if (result.size() > K) {
                    result.pop();
                }
            }
        } else {
            for (int j = 0; j < partial_range.size(); ++j) {
                if (partial_range[j].first <= st_pos && partial_range[j].second >= en_pos) {
                    int sz = partial_range[j].second - partial_range[j].first + 1;
                    result = partial_hnsw[j]->searchKnnWithRange(query_vec.data(), 100, l, r, EFS_Q2_BASE + EFS_Q2_K / sz * range_cnt);
                    break;
                }
            }
        }
        while (knn.size() < K) {
            if (result.empty()) {
                knn.push_back(0);
                continue;
            }
            knn.push_back(result.top().second);
            result.pop();
        }
        #if defined(CLOSE_RESULT_Q2)
        std::fill(knn.begin(), knn.end(), I32_MAX);
        #endif
    }
    auto e_q2 = std::chrono::system_clock::now();
    std::cout << "Q2:  " << time_cost(s_q2, e_q2) << " (ms)\n";

    // solve query type0 (ANN)
    auto &whole_hnsw = partial_hnsw.back();
    auto s_q0 = std::chrono::system_clock::now();
    auto &q0_indexes = query_set._type_index[0];
    #pragma omp parallel for schedule(dynamic, CHUNK_SIZE)
    for (uint32_t i = 0; i < q0_indexes.size(); i++)  {
        const auto& query = query_set._queries[q0_indexes[i]];
        const auto& query_vec = query._vec;
        auto& knn = knn_results[q0_indexes[i]];
        std::priority_queue<std::pair<float, base_hnsw::labeltype>> result;
        result = whole_hnsw->searchKnn(query_vec.data(), 100, EFS_Q0_BASE);
        while (knn.size() < K) {
            if (result.empty()) {
                knn.push_back(0);
                continue;
            }
            knn.push_back(result.top().second);
            result.pop();
        }
        #if defined(CLOSE_RESULT_Q0)
        std::fill(knn.begin(), knn.end(), I32_MAX);
        #endif
    }
    auto e_q0 = std::chrono::system_clock::now();
    std::cout << "Q0:  " << time_cost(s_q0, e_q0) << " (ms)\n";
}