#pragma once

#include "core.h"
#include "util.h"
#include "data_format.h"
#include "rangehnswalg.h"

void SolveQueryType13(
    base_hnsw::InnerProductSpace& space,
    DataSet& data_set,
    QuerySet& query_set,
    std::vector<std::vector<uint32_t>>& knn_results) {
    // base_hnsw::L2Space t_space(VEC_DIMENSION);
    auto& data_label_index = data_set._label_index;
    // build index
    std::unordered_map<int, std::unique_ptr<base_hnsw::RangeHierarchicalNSW<float>>> label_hnsw;
    std::unordered_map<int, std::vector<std::unique_ptr<base_hnsw::RangeHierarchicalNSW<float>>>> label_partial_hnsw;
    std::unordered_map<int, std::vector<int>> label_partial_range;
    auto s_index13 = std::chrono::system_clock::now();
    for (auto& [_, index] : data_label_index) {
        std::sort(index.begin(), index.end(), [&](const auto lhs, const auto rhs) {
            return data_set._timestamps[lhs] < data_set._timestamps[rhs];
        });
    }
    // build hnsw for large label vecs
    for (auto& [label, index] : data_label_index) {
        if (index.size() >= HNSW_BUILD_THRESHOLD) {
            if (index.size() >= HNSW_PARTIAL_BUILD_THRESHOLD) {
                auto cur_hnsw = std::make_unique<base_hnsw::RangeHierarchicalNSW<float>>(&space, data_set.size(), M_Q13, EF_CONSTRUCTION_Q13);
                // label_partial_hnsw[label] = {};
                label_partial_range[label] = {};
                for (int i = 0, st_pos = 0; i < SEGEMENTS_Q3.size(); ++i) {
                    int en_pos = SEGEMENTS_Q3[i] * index.size();
                    #pragma omp parallel for schedule(dynamic, CHUNK_SIZE)
                    for (int j = st_pos; j < en_pos; ++j) {
                        cur_hnsw->addPoint(data_set._vecs[index[j]].data(), index[j], data_set._sum_of_squares[index[j]], data_set._timestamps[index[j]]);
                    }
                    label_partial_range[label].push_back(en_pos - 1);
                    st_pos = en_pos;
                    if (en_pos != index.size()) {
                        // auto st_copy = std::chrono::system_clock::now();
                        label_partial_hnsw[label].push_back(std::move(std::make_unique<base_hnsw::RangeHierarchicalNSW<float>>(*cur_hnsw.get())));
                        // auto en_copy = std::chrono::system_clock::now();
                        // std::cout << "copy cost: " << time_cost(st_copy, en_copy) << " (ms)\n";
                    } else {
                        label_partial_hnsw[label].push_back(std::move(cur_hnsw));
                    }
                }
            } else {
                label_hnsw[label] = std::move(std::make_unique<base_hnsw::RangeHierarchicalNSW<float>>(
                                    &space, index.size(), M_Q13, EF_CONSTRUCTION_Q13));
                #pragma omp parallel for schedule(dynamic, CHUNK_SIZE)
                for (uint32_t i = 0; i < index.size(); i++) {
                    label_hnsw[label]->addPoint(data_set._vecs[index[i]].data(), index[i], data_set._sum_of_squares[index[i]], data_set._timestamps[index[i]]);
                }
            }
        }
    }
    auto e_index13 = std::chrono::system_clock::now();
    std::cout << "I13: " << time_cost(s_index13, e_index13) << " (ms)\n";

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
        if (index.size() < HNSW_BUILD_THRESHOLD) {
            for (auto id : index) {
                if (data_set._timestamps[id] < l || data_set._timestamps[id] > r) {
                    continue;
                }
                #if defined(USE_AVX)
                    float dist = data_set._sum_of_squares[id] + query._sum_of_square + (-2) * space.get_dist_func()(data_set._vecs[id].data(), query_vec.data(), &VEC_DIMENSION);
                    // float dist = t_space.get_dist_func()(data_set._vecs[id].data(), query_vec.data(), &VEC_DIMENSION);
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
            if (range_cnt <= RANGE_BF_THRESHOLD_Q3) {
                for (int j = st_pos; j < en_pos; ++j) {
                    auto id = index[j];
                    #if defined(USE_AVX)
                        float dist = data_set._sum_of_squares[id] + query._sum_of_square + (-2) * space.get_dist_func()(data_set._vecs[id].data(), query_vec.data(), &VEC_DIMENSION);
                        // float dist = t_space.get_dist_func()(data_set._vecs[id].data(), query_vec.data(), &VEC_DIMENSION);
                    #else
                        float dist = EuclideanDistanceSquare(data_set._vecs[id], query_vec);
                    #endif
                    result.push(std::make_pair(dist, id));
                    if (result.size() > K) {
                        result.pop();
                    }
                }
            } else {
                if (index.size() >= HNSW_PARTIAL_BUILD_THRESHOLD) {
                    for (int j = 0; j < label_partial_range[label].size(); ++j) {
                        if (label_partial_range[label][j] >= en_pos - 1) {
                            result = label_partial_hnsw[label][j]->searchKnnWithRange(query_vec.data(), 100, l, r, query._sum_of_square,
                                     EFS_Q3_BASE + EFS_Q3_K * range_cnt / label_partial_range[label][j]);
                            break;
                        }
                    }
                } else {
                    result = label_hnsw[label]->searchKnnWithRange(query_vec.data(), 100, l, r, query._sum_of_square,
                             EFS_Q3_BASE + EFS_Q3_K * range_cnt / index.size());
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
        if (index.size() < HNSW_BUILD_THRESHOLD) {
            for (auto id : index) {
                #if defined(USE_AVX)
                    float dist = data_set._sum_of_squares[id] + query._sum_of_square + (-2) * space.get_dist_func()(data_set._vecs[id].data(), query_vec.data(), &VEC_DIMENSION);
                    // float dist = t_space.get_dist_func()(data_set._vecs[id].data(), query_vec.data(), &VEC_DIMENSION);
                #else
                    float dist = EuclideanDistanceSquare(data_set._vecs[id], query_vec);
                #endif
                result.push(std::make_pair(dist, id));
                if (result.size() > K) {
                    result.pop();
                }
            }
        } else {
            if (index.size() >= HNSW_PARTIAL_BUILD_THRESHOLD) {
                result = label_partial_hnsw[label].back()->searchKnn(query_vec.data(), 100, query._sum_of_square, EFS_Q1_BASE + EFS_Q1_K * std::sqrt(1.0 * index.size() / data_set.size()));
            } else {
                result = label_hnsw[label]->searchKnn(query_vec.data(), 100, query._sum_of_square, EFS_Q1_BASE + EFS_Q1_K * std::sqrt(1.0 * index.size() / data_set.size()));
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
        #if defined(CLOSE_RESULT_Q1)
        std::fill(knn.begin(), knn.end(), I32_MAX);
        #endif
    }
    auto e_q1 = std::chrono::system_clock::now();
    std::cout << "Q1:  " << time_cost(s_q1, e_q1) << " (ms)\n";
}