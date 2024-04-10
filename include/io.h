/**
 *  Example code for IO, read binary data vectors and save KNNs to path.
 *
 */

#pragma once
#include "core.h"
#include "data_format.h"

/// @brief Save knng in binary format (uint32_t) with name "output.bin"
/// @param knn a (N * 100) shape 2-D vector
/// @param path target save path, the output knng should be named as
/// "output.bin" for evaluation
void SaveKNN(const std::vector<std::vector<uint32_t>>& knns,
             const std::string& path = "output.bin") {
    std::ofstream ofs(path, std::ios::out | std::ios::binary);
    const int K = 100;
    const uint32_t N = knns.size();
    assert(knns.front().size() == K);
    for (unsigned i = 0; i < N; ++i) {
        auto const& knn = knns[i];
        ofs.write(reinterpret_cast<char const*>(&knn[0]), K * sizeof(uint32_t));
    }
    ofs.close();
}

/// @brief Reading binary data vectors. Raw data store as a (N x dim)
/// @param file_path file path of binary data
/// @param data returned 2D data vectors
void ReadData(const std::string& file_path,
              const int num_dimensions,
              DataSet& data_set) {
    std::ifstream ifs;
    ifs.open(file_path, std::ios::binary);
    assert(ifs.is_open());
    uint32_t N;  // num of points
    ifs.read((char*)&N, sizeof(uint32_t));
    data_set.reserve(N);
    std::vector<float> buff(num_dimensions);
    int id = 0;
    auto& node_label_index = data_set._label_index;
    while (ifs.read((char*)buff.data(), num_dimensions * sizeof(float))) {
        auto label = static_cast<float>(buff[0]);
        data_set._timestamps.push_back(static_cast<float>(buff[1]));
        data_set._vecs.emplace_back();
        data_set._vecs.back().reserve(num_dimensions - 2);
        for (int d = 2; d < num_dimensions; ++d) {
            data_set._vecs.back().push_back(static_cast<float>(buff[d]));
        }
        if (!node_label_index.count(label)) {
            node_label_index[label] = {id};
        } else {
            node_label_index[label].push_back(id);
        }
        id++;
    }
    ifs.close();
}

/// @brief Reading binary data vectors. Raw data store as a (N x dim)
/// @param file_path file path of binary data
/// @param data returned 2D data vectors
void ReadQuery(const std::string& file_path,
               const int num_dimensions,
               QuerySet& query_set) {
    std::ifstream ifs;
    ifs.open(file_path, std::ios::binary);
    assert(ifs.is_open());
    uint32_t N;  // num of points
    ifs.read((char*)&N, sizeof(uint32_t));
    query_set.reserve(N);
    std::vector<float> buff(num_dimensions);
    int id = 0;
    while (ifs.read((char*)buff.data(), num_dimensions * sizeof(float))) {
        query_set._queries.emplace_back();
        auto& now_query = query_set._queries.back();
        int32_t type = static_cast<float>(buff[0]);
        now_query._label = static_cast<float>(buff[1]);
        now_query._l = static_cast<float>(buff[2]);
        now_query._r = static_cast<float>(buff[3]);
        now_query._vec.resize(num_dimensions - 4);
        for (int d = 4; d < num_dimensions; ++d) {
            now_query._vec[d - 4] = static_cast<float>(buff[d]);
        }
        query_set._type_index[type].push_back(id++);
    }
    ifs.close();
    std::sort(query_set._type_index[1].begin(), query_set._type_index[1].end(), [&](const auto lhs, const auto rhs) {
        return query_set._queries[lhs]._label < query_set._queries[rhs]._label;
    });
    std::sort(query_set._type_index[3].begin(), query_set._type_index[3].end(), [&](const auto lhs, const auto rhs) {
        return query_set._queries[lhs]._label < query_set._queries[rhs]._label;
    });
}

void ReadKNN(std::vector<std::vector<uint32_t>>& knns,
             const std::string& path) {
    std::cout << "Reading Oputput: " << path << std::endl;
    std::ifstream ifs;
    ifs.open(path, std::ios::binary);
    assert(ifs.is_open());
    const int K = 100;
    std::vector<uint32_t> buff(K);
    int counter = 0;
    while (ifs.read((char*)buff.data(), K * sizeof(uint32_t))) {
        std::vector<uint32_t> knn(K);
        for (uint32_t i = 0; i < K; ++i) {
            knn[i] = static_cast<uint32_t>(buff[i]);
        }
        for (uint32_t i = 0; i < K; ++i) {
          knns[counter].push_back(knn[i]);
        }
        counter++;
        
    }
    ifs.close();
    std::cout << "Finish Reading Output\n";
}

void Recall(const std::string& path,
            const std::string& truth_path, QuerySet& query_set) {
    const int query_num = 10000;
    std::vector<std::vector<uint32_t>> knns(query_num);
    std::vector<std::vector<uint32_t>> truth(query_num);          
    ReadKNN(knns, path);
    ReadKNN(truth, truth_path);
    const int K = 100;
    const uint32_t N = knns.size();
    int hit = 0;
    for (unsigned i = 0; i < N; ++i) {
        auto &knn = knns[i];
        auto &truth_knn = truth[i];
        std::sort(knn.begin(), knn.end());
        std::sort(truth_knn.begin(), truth_knn.end());
        // std::cout << knn[0] << " " << truth_knn[0] << std::endl;
        std::vector<uint32_t> intersection;
        std::set_intersection(knn.begin(), knn.end(), truth_knn.begin(), truth_knn.end(),
                              std::back_inserter(intersection));

        hit += static_cast<int>(intersection.size());
    }
    float recall = static_cast<float>(hit) / (N * K);
    std::cout << "Overall Recall: " << recall << std::endl;
    
    for (int i = 0; i < 4; i++) {
      auto cur_query_index = query_set._type_index[i];
      int hit = 0;
      for (auto index : cur_query_index) {
        auto &knn = knns[index];
        auto &truth_knn = truth[index];
        std::sort(knn.begin(), knn.end());
        std::sort(truth_knn.begin(), truth_knn.end());
        std::vector<uint32_t> intersection;
        std::set_intersection(knn.begin(), knn.end(), truth_knn.begin(), truth_knn.end(),
                              std::back_inserter(intersection));
        hit += static_cast<int>(intersection.size());
      }
      float recall = static_cast<float>(hit) / (cur_query_index.size() * K);
      std::cout << "Recall for Query Type " << i << ": " << recall << std::endl;    
    }
}