#pragma once

#include "core.h"
#include "util.h"
#include "data_format.h"
#include "rangehnswalg.h"

#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>

void SolveQueryType02(
    DataSet& data_set,
    QuerySet& query_set,
    std::vector<std::vector<uint32_t>>& knn_results) {

    std::string ground_truth_path = "../test/groundtruth-1m.bin";
    std::vector<std::vector<uint32_t>> truth(query_set.size());
    ReadKNN(truth, ground_truth_path);
    std::cerr << "finish read gt" << std::endl;

    base_hnsw::L2Space space(VEC_DIMENSION);
    std::unique_ptr<base_hnsw::RangeHierarchicalNSW<float>> hnsw[2];
    
    // sort data_set
    std::vector<int32_t> data_time_index(data_set.size());
    std::iota(data_time_index.begin(), data_time_index.end(), 0);
    std::sort(data_time_index.begin(), data_time_index.end(), [&](const auto lhs, const auto rhs) {
        return data_set._timestamps[lhs] < data_set._timestamps[rhs];
    });
    uint32_t iter_l = data_set.size() / 2, iter_r = data_set.size() / 2;
    uint32_t back_up_l = data_set.size() / 4, back_up_r = data_set.size() / 4 * 3;

    auto s_i2 = std::chrono::system_clock::now();
    hnsw[0] = std::move(std::make_unique<base_hnsw::RangeHierarchicalNSW<float>>(&space, back_up_r - back_up_l + 1, M_Q02, EF_CONSTRUCTION_Q02));
    // std::cerr << data_set.size();
    // test - decreasing_junnjo
// #pragma omp parallel for schedule(dynamic, CHUNK_SIZE)
    // for (uint32_t i = data_set.size() - 1; i >= 0; --i) {
    //     hnsw[0]->addPoint(data_set._vecs[i].data(), i, data_set._timestamps[i]);
    // }
    // int num = 1;
    // hnsw[0] = std::move(std::make_unique<base_hnsw::RangeHierarchicalNSW<float>>(&space, data_set.size(), M_Q02, EF_CONSTRUCTION_Q02));
    int id = data_time_index[iter_l];
    hnsw[0]->addPoint(data_set._vecs[id].data(), id, data_set._timestamps[id]);
#pragma omp parallel for schedule(dynamic, CHUNK_SIZE)
    for (uint32_t i = back_up_l; i <= iter_l; ++i) {
        id = data_time_index[i];
        hnsw[0]->addPoint(data_set._vecs[id].data(), id, data_set._timestamps[id]);
    }
#pragma omp parallel for schedule(dynamic, CHUNK_SIZE)
    for (uint32_t i = iter_r; i <= back_up_r; ++i) {
        id = data_time_index[i];
        hnsw[0]->addPoint(data_set._vecs[id].data(), id, data_set._timestamps[id]);
    }
    // 1/2
    // hnsw[1] = hnsw[0];
    hnsw[1] = std::make_unique<base_hnsw::RangeHierarchicalNSW<float>>(*hnsw[0]);
    // hnsw[0]->check_linklist("statistic-0.txt");
    // hnsw[1]->check_linklist("statistic-1.txt");
    hnsw[0]->resizeIndex(data_set.size());
    // all 
#pragma omp parallel for schedule(dynamic, CHUNK_SIZE)
    for (uint32_t i = 0; i < back_up_l; ++i) {
        id = data_time_index[i];
        hnsw[0]->addPoint(data_set._vecs[id].data(), id, data_set._timestamps[id]);
    }
#pragma omp parallel for schedule(dynamic, CHUNK_SIZE)
    for (uint32_t i = back_up_r; i < data_set.size(); ++i) {
        id = data_time_index[i];
        hnsw[0]->addPoint(data_set._vecs[id].data(), id, data_set._timestamps[id]);
    }
    auto e_i2 = std::chrono::system_clock::now();
    std::cout << "I2:  " << time_cost(s_i2, e_i2) << " (ms)\n";
    
    std::ofstream file("statistic.txt");
    auto &q2_indexes = query_set._type_index[2];
    hnsw[0]->setEf(EF_SEARCH_Q2);
    hnsw[1]->setEf(EF_SEARCH_Q2);
    auto s_q2 = std::chrono::system_clock::now();
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
        if (0 && range_cnt <= RANGE_BF_THRASHOLD_Q2) {
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
            if (back_up_l <= st_pos && en_pos <= back_up_r) {
                result = hnsw[1]->searchKnnWithRange(query_vec.data(), 100, l, r, 128 + (512.0 + 256 + 128) / data_set.size() * range_cnt);
            } else {
                result = hnsw[0]->searchKnnWithRange(query_vec.data(), 100, l, r, 128 + (512.0 + 256 + 128) / data_set.size() * range_cnt);
            }
        }
        int tmp = result.size();
        while (knn.size() < K) {
            if (result.empty()) {
                knn.push_back(0);
                continue;
            }
            knn.push_back(result.top().second);
            result.pop();
        }

        auto& truth_knn = truth[q2_indexes[i]];
        std::sort(knn.begin(), knn.end());
        std::sort(truth_knn.begin(), truth_knn.end());
        std::vector<uint32_t> intersection;
        std::set_intersection(knn.begin(), knn.end(), truth_knn.begin(), truth_knn.end(), std::back_inserter(intersection));
        file << q2_indexes[i] << " " << (back_up_l <= st_pos && en_pos <= back_up_r) << " " << tmp << " " << static_cast<int>(intersection.size()) << "\n";
    }
    auto e_q2 = std::chrono::system_clock::now();
    std::cout << "Q2:  " << time_cost(s_q2, e_q2) << " (ms)\n";
}