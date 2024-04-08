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
void ReadNode(const std::string& file_path,
              const int num_dimensions,
              std::vector<Node>& nodes) {
    std::ifstream ifs;
    ifs.open(file_path, std::ios::binary);
    assert(ifs.is_open());
    uint32_t N;  // num of points
    ifs.read((char*)&N, sizeof(uint32_t));
    nodes.reserve(N);
    std::vector<float> buff(num_dimensions);
    int id = 0;
    while (ifs.read((char*)buff.data(), num_dimensions * sizeof(float))) {
        nodes.emplace_back();
        auto& now_node = nodes.back();
        now_node._id = id++;
        now_node._label = static_cast<float>(buff[0]);
        now_node._timestamp = static_cast<float>(buff[1]);
        #if defined(USE_AVX)
            now_node._vec.resize(num_dimensions - 2 + ALIGN_SIMD_AVX);
        #else
            now_node._vec.resize(num_dimensions - 2);
        #endif
        
        for (int d = 2; d < num_dimensions; ++d) {
            now_node._vec[d - 2] = static_cast<float>(buff[d]);
        }
    }
    ifs.close();
}

/// @brief Reading binary data vectors. Raw data store as a (N x dim)
/// @param file_path file path of binary data
/// @param data returned 2D data vectors
void ReadQuery(const std::string& file_path,
               const int num_dimensions,
               std::vector<Query>& queries,
               std::vector<std::vector<int32_t>> &type_index) {
    std::ifstream ifs;
    ifs.open(file_path, std::ios::binary);
    assert(ifs.is_open());
    uint32_t N;  // num of points
    ifs.read((char*)&N, sizeof(uint32_t));
    queries.reserve(N);
    std::vector<float> buff(num_dimensions);
    int counter = 0;
    while (ifs.read((char*)buff.data(), num_dimensions * sizeof(float))) {
        queries.emplace_back();
        auto& now_query = queries.back();
        now_query._type = static_cast<float>(buff[0]);
        now_query._label = static_cast<float>(buff[1]);
        now_query._l = static_cast<float>(buff[2]);
        now_query._r = static_cast<float>(buff[3]);
        #if defined(USE_AVX)
            now_query._vec.resize(num_dimensions - 4 + ALIGN_SIMD_AVX);
        #else
            now_query._vec.resize(num_dimensions - 4);
        #endif
        
        for (int d = 4; d < num_dimensions; ++d) {
            now_query._vec[d - 4] = static_cast<float>(buff[d]);
        }
        type_index[now_query._type].push_back(counter++);
    }
    ifs.close();
}

void ReadKNN(std::vector<std::vector<uint32_t>>& knns,
             const std::string& path = "output.bin") {
    std::cout << "Reading Oputput: " << path << std::endl;
    std::ifstream ifs;
    ifs.open(path, std::ios::binary);
    assert(ifs.is_open());
    const int K = 100;
    std::vector<uint32_t> buff(K);
    int counter = 0;
    while (ifs.read((char*)buff.data(), K * sizeof(uint32_t))) {
        std::vector<uint32_t> knn;
        for (uint32_t i = 0; i < K; ++i) {
            knn[i] = static_cast<uint32_t>(buff[i]);
        }
        knns[counter++] = std::move(knn);
    }
    ifs.close();
    std::cout << "Finish Reading Output\n";
}