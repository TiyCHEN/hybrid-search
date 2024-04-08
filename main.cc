#include "io.h"
#include "util.h"
#include "hnsw_simd_dist_func.h"
#include "query_type02.h"
#include "query_type13.h"

int main(int argc, char** argv) {
    SetSIMDFunc();
    std::string source_path = "../data/dummy-data.bin";
    std::string query_path = "../data/dummy-queries.bin";
    std::string knn_save_path = "../output.bin";

    // Also accept other path for source data
    if (argc > 1) {
        source_path = std::string(argv[1]);
        query_path = std::string(argv[2]);
    }
    std::cout << "NUM_THREAD: " << NUM_THREAD << '\n';
    //  read process
    auto s_read = std::chrono::system_clock::now();
    // Read data points
    std::vector<Node> nodes;
    ReadNode(source_path, NODE_DIMENSION, nodes);
    std::sort(nodes.begin(), nodes.end(), [&](const Node& a, const Node& b){
        return a._timestamp < b._timestamp;
    });
    // Read queries
    std::vector<Query> queries;
    // classify nodes by type
    std::vector<std::vector<int32_t>> query_type_index(QUERY_TYPE_SIZE);
    ReadQuery(query_path, QUERY_DIMENTION, queries, query_type_index);
    auto e_read = std::chrono::system_clock::now();
    std::cout << "read cost " << time_cost(s_read, e_read) << " (ms)\n";

    uint32_t n = nodes.size();
    uint32_t nq = queries.size();
    std::cout << "# data points:  " << n << "\n";
    std::cout << "# queries:      " << nq << "\n";

    // pre-processing
    // classify nodes by label
    std::unordered_map<int32_t, std::vector<int32_t>> node_label_index;
    for (int32_t i = 0; i < nodes.size(); ++i) {
        const auto& node = nodes[i];
        auto label = node._label;
        if (!node_label_index.count(label)) {
            node_label_index[label] = {i};
        } else {
            node_label_index[label].push_back(i);
        }
    }

    // start processing
    std::vector<std::vector<uint32_t>> knn_results(queries.size());
    for (auto& knns : knn_results) {
        knns.reserve(K);
    }

    auto s02 = std::chrono::system_clock::now();
    solve_query_type02(nodes, queries, query_type_index, knn_results);
    auto e02 = std::chrono::system_clock::now();
    std::cout << "solve query02 cost " << time_cost(s02, e02) << " (ms)\n";

    auto s13 = std::chrono::system_clock::now();
    solve_query_type13(nodes, queries, node_label_index, query_type_index, knn_results);
    auto e13 = std::chrono::system_clock::now();
    std::cout << "solve query13 cost " << time_cost(s13, e13) << " (ms)\n";

    // solve type 0 query
//    auto s0 = std::chrono::system_clock::now();
//    solve_query_type0(nodes, queries, query_type_index[0], knn_results);
//    auto e0 = std::chrono::system_clock::now();
//    std::cout << "solve query0 cost " << time_cost(s0, e0) << " (ms)\n";

    // solve type 1 query
//    auto s1 = std::chrono::system_clock::now();
//    solve_query_type1(nodes, queries, node_label_index, query_type_index[1], knn_results);
//    auto e1 = std::chrono::system_clock::now();
//    std::cout << "solve query1 cost " << time_cost(s1, e1) << " (ms)\n";

    // solve type 2 query
//    auto s2 = std::chrono::system_clock::now();
//    solve_query_type2(nodes, queries, node_label_index, query_type_index[2], knn_results);
//    auto e2 = std::chrono::system_clock::now();
//    std::cout << "solve query2 cost " << time_cost(s2, e2) << " (ms)\n";

    // solve type 3 query
//    auto s3 = std::chrono::system_clock::now();
//    solve_query_type3(nodes, queries, node_label_index, query_type_index[3], knn_results);
//    auto e3 = std::chrono::system_clock::now();
//    std::cout << "solve query3 cost " << time_cost(s3, e3) << " (ms)\n";

    // // save the results
    SaveKNN(knn_results, knn_save_path);
    return 0;
}
