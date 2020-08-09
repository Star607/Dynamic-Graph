#include <torch/extension.h>
#include <iostream>

torch::Tensor upper_bound(torch::Tensor bucket_adj) {
    /*  First of all, the last dimension of bucket_adj must be non-decreasing. We compute the upper_bound of each element along the last dimension.
    *   The return result will be the same shape as input `bucket_adj`.
    */
    using namespace torch::indexing;
    // We assume the bucket_adj is 2D tensor.
    auto sizes = bucket_adj.sizes();
    auto dim = sizes[1];
    // Default upper_bound is the array length, which means it is the largest element.
    auto ans = torch::full(sizes, dim, torch::kInt64);
    for (auto i = 0; i < sizes[0]; i++) {
        auto ub = 1;
        for (auto j = 0; j < dim && ub < dim; j++) {
            // Attention! If use `bucket_adj[i][ub].item<float>()` to get float value, the float precision cannot be guaranteed, so we directly get .item<bool>().
            while (ub < dim && (bucket_adj[i][ub] <= bucket_adj[i][j]).item<bool>()) {
                ub += 1;
            } 
            ans.index_put_({i, j}, ub);
        }
    }
    return ans;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &upper_bound, "Upper Bound");
}
