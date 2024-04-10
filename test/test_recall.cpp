#include "io.h"
#include "util.h"

int main(int argc, char** argv) {
    std::string source_path = "../data/contest-data-release-1m.bin";
    std::string query_path = "../data/contest-queries-release-1m.bin";
    std::string knn_save_path = "../output.bin";
    std::string ground_truth_path = "../test/groundtruth-1m.bin";

    // Also accept other path for source data
    if (argc > 1) {
        source_path = std::string(argv[1]);
        query_path = std::string(argv[2]);
    }

    //  read process
    DataSet data_set;
    ReadData(source_path, NODE_DIMENSION, data_set);
    QuerySet query_set;
    ReadQuery(query_path, QUERY_DIMENTION, query_set);


    Recall(knn_save_path, ground_truth_path, query_set);
    return 0;
}
