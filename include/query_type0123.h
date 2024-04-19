#pragma once

#include "core.h"
#include "util.h"
#include "data_format.h"
#include "hnswlib/rangehnswalg.h"

void SolveQueryType0123(
    DataSet& data_set,
    QuerySet& query_set,
    std::vector<std::vector<uint32_t>>& knn_results) {
    auto& data_label_index = data_set._label_index;
    // build index
    std::unordered_map<int, std::unique_ptr<base_hnsw::RangeHierarchicalNSW<float>>> label_hnsw;
    // build hnsw for large label vecs
    auto s_index13 = std::chrono::system_clock::now();
    int32_t other_hnsw_size = 0;
    int32_t biggest_label = -1;
    size_t biggest_label_size = 0;
    for (auto& [label, index] : data_label_index) {
        if (index.size() > biggest_label_size) {
            biggest_label_size = index.size();
            biggest_label = label;
        }
        if (index.size() < HNSW_MERGE_THRASHOLD) {
            other_hnsw_size += index.size();
        }
    }
    for (auto& [label, index] : data_label_index) {
        if (index.size() >= HNSW_BUILD_THRASHOLD) {
            base_hnsw::L2Space space(VEC_DIMENSION);
            if (label != biggest_label) {
                label_hnsw[label] = std::move(std::make_unique<base_hnsw::RangeHierarchicalNSW<float>>(
                                              &space, index.size(), M_Q13, EF_CONSTRUCTION_Q13));
            } else {
                label_hnsw[label] = std::move(std::make_unique<base_hnsw::RangeHierarchicalNSW<float>>(
                                              &space, std::min(data_set.size(), index.size() + other_hnsw_size),
                                              M_Q13, EF_CONSTRUCTION_Q13));
            }
            #pragma omp parallel for schedule(dynamic, CHUNK_SIZE)
            for (uint32_t i = 0; i < index.size(); i++) {
                label_hnsw[label]->addPoint(data_set._vecs[index[i]].data(), index[i], data_set._timestamps[index[i]]);
            }
        }
    }
    auto e_index13 = std::chrono::system_clock::now();
    std::cout << "build index 13 cost: " << time_cost(s_index13, e_index13) << " (ms)\n";

    // solve query1 (Filter-ANN)
    for (auto &[_, hnsw] : label_hnsw) {
        hnsw->setEf(EF_SEARCH_Q1);
    }
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
        if (index.size() < HNSW_BUILD_THRASHOLD) {
            for (auto id : index) {
                #if defined(USE_AVX)
                    float dist = base_hnsw::HybridSimd(data_set._vecs[id].data(), query_vec.data(), &VEC_DIMENSION);
                #else
                    float dist = EuclideanDistanceSquare(data_set._vecs[id], query_vec);
                #endif
                result.push(std::make_pair(dist, id));
                if (result.size() > K) {
                    result.pop();
                }
            }
        } else {
            result = label_hnsw[label]->searchKnn(query_vec.data(), 100);
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
    std::cout << "search query 1 cost: " << time_cost(s_q1, e_q1) << " (ms)\n";

    // solve query3 (Filter-Range-ANN)
    for (auto &[_, hnsw] : label_hnsw) {
        hnsw->setEf(EF_SEARCH_Q3);
    }
    for (auto& [_, index] : data_label_index) {
        std::sort(index.begin(), index.end(), [&](const auto lhs, const auto rhs) {
            return data_set._timestamps[lhs] < data_set._timestamps[rhs];
        });
    }
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
        if (index.size() < HNSW_BUILD_THRASHOLD) {
            for (auto id : index) {
                if (data_set._timestamps[id] < l || data_set._timestamps[id] > r) {
                    continue;
                }
                #if defined(USE_AVX)
                    float dist = base_hnsw::HybridSimd(data_set._vecs[id].data(), query_vec.data(), &VEC_DIMENSION);
                #else
                    float dist = EuclideanDistanceSquare(data_set._vecs[id], query_vec);
                #endif
                result.push(std::make_pair(dist, id));
                if (result.size() > K) {
                    result.pop();
                }
            }
        } else {
            int st_pos = std::lower_bound(index.begin(), index.end(), l, [&](const auto x, const float ti) {
                return data_set._timestamps[x] < ti;
            }) - index.begin();
            int en_pos = std::lower_bound(index.begin(), index.end(), r, [&](const auto x, const float ti) {
                return data_set._timestamps[x] <= ti;
            }) - index.begin();
            int range_cnt = en_pos - st_pos;
            if (range_cnt <= RANGE_BF_THRASHOLD_Q3) {
                for (int j = st_pos; j < en_pos; ++j) {
                    auto id = index[j];
                    #if defined(USE_AVX)
                        float dist = base_hnsw::HybridSimd(data_set._vecs[id].data(), query_vec.data(), &VEC_DIMENSION);
                    #else
                        float dist = EuclideanDistanceSquare(data_set._vecs[id], query_vec);
                    #endif
                    result.push(std::make_pair(dist, id));
                    if (result.size() > K) {
                        result.pop();
                    }
                }
            } else {
                result = label_hnsw[label]->searchKnnWithRange(query_vec.data(), 100, l, r);
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
    std::cout << "search query 3 cost: " << time_cost(s_q3, e_q3) << " (ms)\n";


    // build index02
    std::vector<std::unique_ptr<base_hnsw::RangeHierarchicalNSW<float>>> merged_hnsw;
    merged_hnsw.push_back(std::move(label_hnsw[biggest_label]));
    auto s_index02 = std::chrono::system_clock::now();
    for (auto& [label, index] : data_label_index) {
        if (label == biggest_label) continue;
        if (index.size() >= HNSW_MERGE_THRASHOLD) {
            merged_hnsw.push_back(std::move(label_hnsw[label]));
        } else {
            auto& other_hnsw = merged_hnsw.front();
            #pragma omp parallel for schedule(dynamic, CHUNK_SIZE)
            for (uint32_t i = 0; i < index.size(); i++) {
                other_hnsw->addPoint(data_set._vecs[index[i]].data(), index[i], data_set._timestamps[index[i]]);
            }
        }
    }
    auto e_index02 = std::chrono::system_clock::now();
    std::cout << "build index 02 cost: " << time_cost(s_index02, e_index02) << " (ms)\n";

    // solve query type0 (ANN)
    for (auto& hnsw : merged_hnsw) {
        hnsw->setEf(EF_SEARCH_Q0);
    }
    auto s_q0 = std::chrono::system_clock::now();
    auto &q0_indexes = query_set._type_index[0];
    #pragma omp parallel for schedule(dynamic, CHUNK_SIZE)
    for (uint32_t i = 0; i < q0_indexes.size(); i++)  {
        const auto& query = query_set._queries[q0_indexes[i]];
        const auto& query_vec = query._vec;
        auto& knn = knn_results[q0_indexes[i]];
        std::priority_queue<std::pair<float, base_hnsw::labeltype>> result;
        bool is_first = true;
        for (auto& hnsw : merged_hnsw) {
            if (is_first) {
                result = hnsw->searchKnn(query_vec.data(), 100);
                is_first = false;
                continue;
            }
            auto partial_result = hnsw->searchKnn(query_vec.data(), 100);
            while (!partial_result.empty()) {
                result.push(partial_result.top());
                partial_result.pop();
                if (result.size() > K) {
                    result.pop();
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
        #if defined(CLOSE_RESULT_Q0)
        std::fill(knn.begin(), knn.end(), I32_MAX);
        #endif
    }
    auto e_q0 = std::chrono::system_clock::now();
    std::cout << "search query 0 cost: " << time_cost(s_q0, e_q0) << " (ms)\n";

    // solve query type2 (Range-ANN)
    for (auto& hnsw : merged_hnsw) {
        hnsw->setEf(EF_SEARCH_Q2);
    }
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
                    float dist = base_hnsw::HybridSimd(data_set._vecs[id].data(), query_vec.data(), &VEC_DIMENSION);
                #else
                    float dist = EuclideanDistanceSquare(data_set._vecs[id], query_vec);
                #endif
                result.push(std::make_pair(dist, id));
                if (result.size() > K) {
                    result.pop();
                }
            }
        } else {
            bool is_first = true;
            for (auto& hnsw : merged_hnsw) {
                if (is_first) {
                    result = hnsw->searchKnnWithRange(query_vec.data(), 100, l, r);
                    is_first = false;
                    continue;
                }
                auto partial_result = hnsw->searchKnnWithRange(query_vec.data(), 100, l, r);
                while (!partial_result.empty()) {
                    result.push(partial_result.top());
                    partial_result.pop();
                    if (result.size() > K) {
                        result.pop();
                    }
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
    std::cout << "search query 2 cost: " << time_cost(s_q2, e_q2) << " (ms)\n";
}