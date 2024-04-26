#pragma once

#include "core.h"
#include "util.h"
#include "data_format.h"
#include "hybird_hnswalg.h"

void SolveQueryType0123(
    base_hnsw::L2Space& space,
    DataSet& data_set,
    QuerySet& query_set,
    std::vector<std::vector<uint32_t>>& knn_results) {
    // build index02
    auto s_index0123 = std::chrono::system_clock::now();
    std::vector<int32_t> data_time_index(data_set.size());
    std::iota(data_time_index.begin(), data_time_index.end(), 0);
    std::sort(data_time_index.begin(), data_time_index.end(), [&](const auto lhs, const auto rhs) {
        return data_set._timestamps[lhs] < data_set._timestamps[rhs];
    });
    auto& data_label_index = data_set._label_index;
    for (auto& [_, index] : data_label_index) {
        std::sort(index.begin(), index.end(), [&](const auto lhs, const auto rhs) {
            return data_set._timestamps[lhs] < data_set._timestamps[rhs];
        });
    }
    auto cur_hnsw = std::make_unique<base_hnsw::HybidHierarchicalNSW<float>>(&space, data_set.size(), M_Q0123, EF_CONSTRUCTION_Q0123);
    std::vector<std::unique_ptr<base_hnsw::HybidHierarchicalNSW<float>>> partial_hnsw;
    std::vector<int> partial_range; // {{[0, x1), [0, x2), ..., [0, xi)} | x1 < x2 < ... < xi}
    for (int i = 0, st_pos = 0; i < SEGEMENTS.size(); ++i) {
        int en_pos = data_set.size() * SEGEMENTS[i];
        #pragma omp parallel for schedule(dynamic, CHUNK_SIZE)
        for (int j = st_pos; j < en_pos; ++j) {
            cur_hnsw->addPoint(data_set._vecs[data_time_index[j]].data(), data_time_index[j],
                                data_set._timestamps[data_time_index[j]], data_set._labels[data_time_index[j]]);
        }
        partial_range.emplace_back(en_pos);
        st_pos = en_pos;
        if (en_pos != data_set.size()) {
            // auto st_copy = std::chrono::system_clock::now();
            partial_hnsw.push_back(std::move(std::make_unique<base_hnsw::HybidHierarchicalNSW<float>>(*cur_hnsw.get())));
            // auto en_copy = std::chrono::system_clock::now();
            // std::cout << "copy cost: " << time_cost(st_copy, en_copy) << " (ms)\n";
        } else {
            partial_hnsw.push_back(std::move(cur_hnsw));
        }
    }
    auto e_index0123 = std::chrono::system_clock::now();
    std::cout << "I0123: " << time_cost(s_index0123, e_index0123) << " (ms)\n";


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


    // solve query1 (Filter-ANN)
    auto s_q1 = std::chrono::system_clock::now();
    auto &q1_indexes = query_set._type_index[1];
    #pragma omp parallel for schedule(dynamic, CHUNK_SIZE)
    for (uint32_t i = 0; i < q1_indexes.size(); i++)  {
        const auto& query = query_set._queries[q1_indexes[i]];
        const int32_t label = query._label;
        const auto& query_vec = query._vec;
        auto& knn = knn_results[q1_indexes[i]];

        if (!data_label_index.count(label)) {
            while (knn.size() < K) {
                knn.push_back(0);
            }
           continue;
        }
        std::priority_queue<std::pair<float, base_hnsw::labeltype>> result;
        auto& index = data_label_index[label];
        if (index.size() < BF_THRESHOLD_Q1) {
            for (auto id : index) {
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
            result = partial_hnsw.back()->searchKnnWithAttribute(query_vec.data(), 100, label,
                     EFS_Q1_BASE + EFS_Q1_K * index.size() / data_set.size());
        }
        while (knn.size() < K) {
            if (result.empty()) {
                knn.push_back(0);
                continue;
            }
            knn.push_back(result.top().second);
            result.pop();
        }
        #if defined(CLOSE_RESULT_Q1)
        std::fill(knn.begin(), knn.end(), I32_MAX);
        #endif
    }
    auto e_q1 = std::chrono::system_clock::now();
    std::cout << "Q1:  " << time_cost(s_q1, e_q1) << " (ms)\n";


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
        int candidate_cnt = en_pos - st_pos;

        std::priority_queue<std::pair<float, base_hnsw::labeltype>> result;
        if (candidate_cnt < BF_THRESHOLD_Q2) {
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
                if (partial_range[j] >= en_pos) {
                    int sz = partial_range[j];
                    result = partial_hnsw[j]->searchKnnWithRange(query_vec.data(), 100, l, r, EFS_Q2_BASE + EFS_Q2_K * candidate_cnt / sz);
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


    // solve query3 (Filter-Range-ANN)
    auto s_q3 = std::chrono::system_clock::now();
    auto &q3_indexes = query_set._type_index[3];
    #pragma omp parallel for schedule(dynamic, CHUNK_SIZE)
    for (uint32_t i = 0; i < q3_indexes.size(); i++)  {
        const auto& query = query_set._queries[q3_indexes[i]];
        const int32_t label = query._label;
        const float l = query._l;
        const float r = query._r;
        const auto& query_vec = query._vec;
        auto& knn = knn_results[q3_indexes[i]];

        if (!data_label_index.count(label)) {
            while (knn.size() < K) {
                knn.push_back(0);
            }
          continue;
        }
        std::priority_queue<std::pair<float, base_hnsw::labeltype>> result;
        auto& index = data_label_index[label];
        int st_pos = std::lower_bound(index.begin(), index.end(), l, [&](const auto x, const float ti) {
            return data_set._timestamps[x] < ti;
        }) - index.begin();
        int en_pos = std::lower_bound(index.begin(), index.end(), r, [&](const auto x, const float ti) {
            return data_set._timestamps[x] <= ti;
        }) - index.begin();
        int candidate_cnt = en_pos - st_pos;
        if (candidate_cnt <= BF_THRESHOLD_Q3) {
            for (int j = st_pos; j < en_pos; ++j) {
                auto id = index[j];
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
            int now_en_pos = std::lower_bound(data_time_index.begin(), data_time_index.end(), r, [&](const auto x, const float ti) {
                return data_set._timestamps[x] <= ti;
            }) - data_time_index.begin();
            for (int j = 0; j < partial_range.size(); ++j) {
                if (partial_range[j] >= now_en_pos) {
                    int sz = partial_range[j];
                    result = partial_hnsw[j]->searchKnnWithRangeAttribute(query_vec.data(), 100, l, r, label, EFS_Q3_BASE + EFS_Q3_K * candidate_cnt / sz);
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
        #if defined(CLOSE_RESULT_Q3)
        std::fill(knn.begin(), knn.end(), I32_MAX);
        #endif
    }
    auto e_q3 = std::chrono::system_clock::now();
    std::cout << "Q3:  " << time_cost(s_q3, e_q3) << " (ms)\n";
}