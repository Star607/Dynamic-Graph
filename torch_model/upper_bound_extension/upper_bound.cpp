#include <torch/extension.h>

#include <iostream>
#include <thread>
#include <vector>
// #define _OPENMP
// #include <ATen/ParallelOpenMP.h>

torch::Tensor upper_bound(const torch::Tensor &bucket_adj);
torch::Tensor upper_bound_par(const torch::Tensor &bucket_adj);
void sub_upper_bound(const torch::Tensor &bucket_adj, torch::Tensor &ans,
                     const int row);
torch::Tensor upper_bound_full(const torch::Tensor &nodes,
                               const torch::Tensor &etime,
                               const torch::Tensor &degrees);
torch::Tensor upper_bound_full_par(const torch::Tensor &nodes,
                                   const torch::Tensor &etime,
                                   const torch::Tensor &degrees);
void sub_upper_bound_full(const torch::Tensor &etime, torch::Tensor &ans,
                          const int deg, const int idx);
torch::Tensor reference_lower_bound(const torch::Tensor &refer_adj,
                                    const torch::Tensor &bucket_adj);

torch::Tensor upper_bound(const torch::Tensor &bucket_adj) {
    /*  First of all, the last dimension of bucket_adj must be
     * non-decreasing. We compute the upper_bound of each element along the
     * last dimension. The return result will be the same shape as input
     * `bucket_adj`.
     */
    using namespace torch::indexing;
    // We assume the bucket_adj is 2D tensor.
    auto sizes = bucket_adj.sizes();
    auto dim = sizes[1];
    // Default upper_bound is the array length, which means itself is the
    // largest element.
    auto ans = torch::full(sizes, dim, torch::kInt64);
    for (auto i = 0; i < sizes[0]; i++) {
        auto ub = 1;
        for (auto j = 0; j < dim && ub < dim; j++) {
            // Attention! If use `bucket_adj[i][ub].item<float>()` to get
            // float value, the float precision cannot be guaranteed, so we
            // directly get .item<bool>().
            while (ub < dim &&
                   (bucket_adj[i][ub] <= bucket_adj[i][j]).item<bool>()) {
                ub += 1;
            }
            ans.index_put_({i, j}, ub);
        }
    }
    return ans;
}

torch::Tensor upper_bound_par(const torch::Tensor &bucket_adj) {
    // We assume the bucket_adj is 2D tensor.
    auto sizes = bucket_adj.sizes();
    auto dim = sizes[1];
    // Default upper_bound is the array length, which means itself is the
    // largest element. auto processor_count =
    // std::thread::hardware_concurrency(); if (processor_count == 0)
    //     processor_count = 1;
    // auto ans = torch::full(sizes, dim, torch::kInt64);
    // at::parallel_for(0, sizes[0], 1, [&](int64_t begin, int64_t end) {
    //     // auto dim = bucket_adj.sizes(1);
    //     for (auto row = begin; row < end; row++)
    //     {
    //         auto ub = 1;
    //         for (auto j = 0; j < dim && ub < dim; j++)
    //         {
    //             // Attention! If use `bucket_adj[row][ub].item<float>()`
    //             to get float value, the float precision cannot be
    //             guaranteed, so we directly get .item<bool>(). while (ub <
    //             dim && (bucket_adj[row][ub] <=
    //             bucket_adj[row][j]).item<bool>())
    //             {
    //                 ub += 1;
    //             }
    //             ans.index_put_({row, j}, ub);
    //         }
    //     }
    // });
    std::vector<std::thread> workers;
    auto ans = torch::full(sizes, dim, torch::kInt64);
    for (auto i = 0; i < sizes[0]; i++) {
        if (i % 128 == 0) { // Set maximum threads as 128.
            for (std::thread &t : workers) {
               if (t.joinable())
                   t.join();
            }
            workers.clear();
        }
        workers.push_back(std::thread(sub_upper_bound, std::ref(bucket_adj),
                                      std::ref(ans), i));
        // sub_upper_bound(bucket_adj, ans, i);
    }
    for (std::thread &t : workers) {
        if (t.joinable())
            t.join();
    }
    return ans;
}

void sub_upper_bound(const torch::Tensor &bucket_adj, torch::Tensor &ans,
                     const int row) {
    auto dim = bucket_adj.sizes()[1];
    auto ub = 1;
    for (auto j = 0; j < dim && ub < dim; j++) {
        // Attention! If use `bucket_adj[row][ub].item<float>()` to get
        // float value, the float precision cannot be guaranteed, so we
        // directly get .item<bool>().
        while (ub < dim &&
               (bucket_adj[row][ub] <= bucket_adj[row][j]).item<bool>()) {
            ub += 1;
        }
        ans.index_put_({row, j}, ub);
    }
}

#ifndef NDEBUG
#define ASSERT(condition, message)                                             \
    do {                                                                       \
        if (!(condition)) {                                                    \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__   \
                      << " line " << __LINE__ << ": " << message << std::endl; \
            std::terminate();                                                  \
        }                                                                      \
    } while (false)
#else
#define ASSERT(condition, message)                                             \
    do {                                                                       \
    } while (false)
#endif

torch::Tensor upper_bound_full(const torch::Tensor &nodes,
                               const torch::Tensor &etime,
                               const torch::Tensor &degrees) {
    auto size = nodes.sizes()[0];
    ASSERT((size == etime.sizes()[0]), "The reference node Tensor must be "
                                       "the same size as the edge Tensor.");
    ASSERT((size == degrees.sum().item<int>()),
           "The reference node Tensor must be equal to the degree Tensor "
           "sum.");
    auto num_node = degrees.sizes()[0];
    auto ans = torch::full(nodes.sizes(), size, torch::kInt64);
    auto idx = 0;
    for (auto i = 0; i < num_node; i++) {
        auto ub = 1;
        auto deg = degrees[i].item<int>();
        for (auto j = 0; j < deg; j++) {
            // Attention! If use `bucket_adj[row][ub].item<float>()` to get
            // float value, the float precision cannot be guaranteed, so we
            // directly get .item<bool>().
            while (ub < deg &&
                   (etime[idx + ub] <= etime[idx + j]).item<bool>()) {
                ub += 1;
            }
            ans.index_put_({idx + j}, ub);
        }
        idx += degrees[i].item<int>();
    }
    return ans;
}

torch::Tensor upper_bound_full_par(const torch::Tensor &nodes,
                                   const torch::Tensor &etime,
                                   const torch::Tensor &degrees) {
    // auto size = nodes.sizes()[0];
    // ASSERT((size == etime.sizes()[0]), "The reference node Tensor must be
    // the same size as the edge Tensor."); ASSERT((size ==
    // degrees.sum().item<int>()), "The reference node Tensor must be equal
    // to the degree Tensor sum.");
    auto size = etime.sizes()[0];
    auto num_node = degrees.sizes()[0];
    auto ans = torch::full(etime.sizes(), size, torch::kInt64);
    auto idx = 0;
    std::vector<std::thread> workers;

    for (auto i = 0; i < num_node; i++) {
        auto deg = degrees[i].item<int>();
        workers.push_back(std::thread(sub_upper_bound_full, std::ref(etime),
                                      std::ref(ans), deg, idx));
        idx += deg;
    }
    for (std::thread &t : workers) {
        if (t.joinable())
            t.join();
    }
    return ans;
}

void sub_upper_bound_full(const torch::Tensor &etime, torch::Tensor &ans,
                          const int deg, const int idx) {
    auto ub = 1;
    for (auto j = 0; j < deg; j++) {
        while (ub < deg && (etime[idx + ub] <= etime[idx + j]).item<bool>()) {
            ub += 1;
        }
        ans.index_put_({idx + j}, ub);
    }
}

torch::Tensor refer_latest_edge(
    const torch::Tensor &refer_nodes, const torch::Tensor &refer_degrees,
    const torch::Tensor &refer_eids, const torch::Tensor &refer_etime,
    const torch::Tensor &self_nodes, const torch::Tensor &self_degrees,
    const torch::Tensor &self_eids, const torch::Tensor &self_etime) {
    /* For each edge in self_eids, we find the corresponding latest node
     * interaction edge in refer_eids.We assume that `refer_nodes` and
     * `self_nodes` share the same node ids, and `refer_etime` and
     * `self_etime` are non-descending for each node `u`. So that we can
     * calculate the latest refer_eid cumulatively with respect to self_eid.
     * Detailedly, for `(u, t_i)` and `(u, t_{i+1})`, we have that `eid(u, t_i)
     * <= eid(u, t_{i+1})` because `t_i <= t_{i+1}`. All eids are initialized as
     * the first eid of node `u`.
     *
     * Parameters
     * ----------
     * refer_nodes: 1D tensor, shape (N), stores the node ids.
     * refer_degrees : 1D tensor, shape (N), stores the corresponding node
     *      degree for each node. refer_degrees.sum() == refer_eids.sizes()[0].
     * refer_eids : 1D tensor, stores the cumulative edge ids in DGLGraph for
     *      each node.
     * refer_etime : 1D tensor, stores the corresponding edge timestamp.
     */
    using namespace torch::indexing;
    // We assume the bucket_adj is 2D tensor.
    auto num_node = self_nodes.sizes()[0];
    auto num_eids = self_eids.sizes()[0];
    auto ans = torch::zeros({num_eids}, torch::kInt64);
    auto ridx = 0, sidx = 0;
    for (auto i = 0; i < num_node; i++) {
        auto rdeg = refer_degrees[i].item<int>();
        auto sdeg = self_degrees[i].item<int>();
        auto lb = 0;
        for (auto j = 0; j < sdeg; j++) {
            while (lb + 1 < rdeg &&
                   (refer_etime[ridx + lb + 1] < self_etime[sidx + j])
                       .item<bool>()) {
                // We define the lower_bound of self_etime[sidx +j] as the
                // largest refer_etime[ridx + lb] so that `refer_etime[ridx +
                // lb] < self_etime[sidx + j]`.
                lb += 1;
            }
            ans.index_put_({sidx + j}, refer_eids[ridx + lb]);
            // if (self_eids[sidx + j].item<int>() == 2738) {
            //     std::cout << "self id: " << sidx + j << " refer to "
            //               << refer_eids[ridx + lb].item<double>() <<
            //               std::endl;
            //     std::cout << self_etime[sidx + j].item<double>() << " > "
            //               << refer_etime[ridx + lb].item<double>()
            //               << (self_etime[sidx + j] > refer_etime[ridx + lb])
            //                      .item<bool>()
            //               << std::endl;
            // }
        }
        ridx += rdeg;
        sidx += sdeg;
    }
    return ans;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("upper_bound", &upper_bound, "Upper Bound");
    m.def("upper_bound_par", &upper_bound_par, "Upper Bound Parallel");
    m.def("upper_bound_full", &upper_bound_full, "Upper Bound for all nodes.");
    m.def("upper_bound_full_par", &upper_bound_full_par,
          "Upper Bound Parallel for all nodes.");
    m.def("refer_latest_edge", &refer_latest_edge, "Refer Latest Edge.");
}
