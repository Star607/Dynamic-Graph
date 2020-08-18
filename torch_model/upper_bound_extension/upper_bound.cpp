#include <torch/extension.h>
#include <iostream>
#include <thread>
#include <vector>
// #define _OPENMP
// #include <ATen/ParallelOpenMP.h>

torch::Tensor upper_bound(const torch::Tensor &bucket_adj);
torch::Tensor upper_bound_par(const torch::Tensor &bucket_adj);
void sub_upper_bound(const torch::Tensor &bucket_adj, torch::Tensor &ans, const int row);
torch::Tensor upper_bound_full(const torch::Tensor &nodes, const torch::Tensor &etime);
torch::Tensor reference_lower_bound(const torch::Tensor &refer_adj, const torch::Tensor &bucket_adj);

torch::Tensor upper_bound(const torch::Tensor &bucket_adj)
{
    /*  First of all, the last dimension of bucket_adj must be non-decreasing. We compute the upper_bound of each element along the last dimension.
    *   The return result will be the same shape as input `bucket_adj`.
    */
    using namespace torch::indexing;
    // We assume the bucket_adj is 2D tensor.
    auto sizes = bucket_adj.sizes();
    auto dim = sizes[1];
    // Default upper_bound is the array length, which means itself is the largest element.
    auto ans = torch::full(sizes, dim, torch::kInt64);
    for (auto i = 0; i < sizes[0]; i++)
    {
        auto ub = 1;
        for (auto j = 0; j < dim && ub < dim; j++)
        {
            // Attention! If use `bucket_adj[i][ub].item<float>()` to get float value, the float precision cannot be guaranteed, so we directly get .item<bool>().
            while (ub < dim && (bucket_adj[i][ub] <= bucket_adj[i][j]).item<bool>())
            {
                ub += 1;
            }
            ans.index_put_({i, j}, ub);
        }
    }
    return ans;
}

torch::Tensor upper_bound_par(const torch::Tensor &bucket_adj)
{
    // We assume the bucket_adj is 2D tensor.
    auto sizes = bucket_adj.sizes();
    auto dim = sizes[1];
    // Default upper_bound is the array length, which means itself is the largest element.
    // auto processor_count = std::thread::hardware_concurrency();
    // if (processor_count == 0)
    //     processor_count = 1;
    // auto ans = torch::full(sizes, dim, torch::kInt64);
    // at::parallel_for(0, sizes[0], 1, [&](int64_t begin, int64_t end) {
    //     // auto dim = bucket_adj.sizes(1);
    //     for (auto row = begin; row < end; row++)
    //     {
    //         auto ub = 1;
    //         for (auto j = 0; j < dim && ub < dim; j++)
    //         {
    //             // Attention! If use `bucket_adj[row][ub].item<float>()` to get float value, the float precision cannot be guaranteed, so we directly get .item<bool>().
    //             while (ub < dim && (bucket_adj[row][ub] <= bucket_adj[row][j]).item<bool>())
    //             {
    //                 ub += 1;
    //             }
    //             ans.index_put_({row, j}, ub);
    //         }
    //     }
    // });
    std::vector<std::thread> workers;
    auto ans = torch::full(sizes, dim, torch::kInt64);
    for (auto i = 0; i < sizes[0]; i++)
    {
        workers.push_back(std::thread(sub_upper_bound, std::ref(bucket_adj), std::ref(ans), i));
        // sub_upper_bound(bucket_adj, ans, i);
    }
    for (std::thread &t : workers)
    {
        if (t.joinable())
            t.join();
    }
    return ans;
}

void sub_upper_bound(const torch::Tensor &bucket_adj, torch::Tensor &ans, const int row)
{
    auto dim = bucket_adj.sizes()[1];
    auto ub = 1;
    for (auto j = 0; j < dim && ub < dim; j++)
    {
        // Attention! If use `bucket_adj[row][ub].item<float>()` to get float value, the float precision cannot be guaranteed, so we directly get .item<bool>().
        while (ub < dim && (bucket_adj[row][ub] <= bucket_adj[row][j]).item<bool>())
        {
            ub += 1;
        }
        ans.index_put_({row, j}, ub);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("upper_bound", &upper_bound, "Upper Bound");
    m.def("upper_bound_par", &upper_bound_par, "Upper Bound Parallel");
}
