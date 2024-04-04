#include "io.h"
#include "util.h"
#include "query_type0.h"
#include "query_type1.h"
#include "query_type2.h"
#include "query_type3.h"

int main(int argc, char** argv) {
    std::string source_path = "dummy-data.bin";
    std::string query_path = "dummy-queries.bin";
    std::string knn_save_path = "output.bin";

    // Also accept other path for source data
    if (argc > 1) {
        source_path = std::string(argv[1]);
    }

    // Read data points
    std::vector<Node> nodes;
    ReadNode(source_path, NODE_DIMENSION, nodes);

    // Read queries
    std::vector<Query> queries;
    // classify nodes by type
    std::vector<std::vector<int32_t>> query_type_index(QUERY_TYPE_SIZE);
    ReadQuery(query_path, QUERY_DIMENTION, queries, query_type_index);

    uint32_t n = nodes.size();
    uint32_t nq = queries.size();
    std::cout << "# data points:  " << n << "\n";
    std::cout << "# queries:      " << nq << "\n";

    // pre processing
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

    // solve type 0 query
    // TODO:
    solve_query_type0(nodes, queries, query_type_index[0], knn_results);
    // solve type 1 query
    solve_query_type1(nodes, queries, node_label_index, query_type_index[1], knn_results);

    // solve type 2 query
    // TODO:
    solve_query_type2(nodes, queries, node_label_index, query_type_index[2], knn_results);

    // solve type 3 query
    // TODO:
    solve_query_type3(nodes, queries, node_label_index, query_type_index[3], knn_results);

    // // save the results
    SaveKNN(knn_results, knn_save_path);
    return 0;
}
