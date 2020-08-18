import dgl
import torch

from model.utils import timeit
import upper_bound_cpp


@timeit
def _prepare_deg_indices(g):
    """Compute on CPU devices."""
    def _group_func_wrapper(groupby):
        def _compute_deg_indices(edges):
            buc, deg, dim = edges.src["nfeat"].shape
            t = edges.data["timestamp"].view(buc, deg, 1)
            # It doesn't change the behavior but saves to the 1/deg memory use.
            indices = upper_bound_cpp.upper_bound(t.squeeze(-1)).add_(-1)
            # indices = (t.permute(0, 2, 1) <= t).sum(dim=-1).add_(-1)
            # assert torch.all(torch.eq(another_indices, indices))
            return {f"{groupby}_deg_indices": indices}
        return _compute_deg_indices
    g = g.local_var()
    g.edata["timestamp"] = g.edata["timestamp"].cpu()

    src_deg_indices = _group_func_wrapper(groupby="src")
    dst_deg_indices = _group_func_wrapper(groupby="dst")
    g.group_apply_edges(group_by="src", func=src_deg_indices)
    g.group_apply_edges(group_by="dst", func=dst_deg_indices)
    return {"src_deg_indices": g.edata["src_deg_indices"],
            "dst_deg_indices": g.edata["dst_deg_indices"]}


@timeit
def _par_deg_indices(g):
    """Compute on CPU devices."""
    def _group_func_wrapper(groupby):
        def _compute_deg_indices(edges):
            buc, deg, dim = edges.src["nfeat"].shape
            t = edges.data["timestamp"].view(buc, deg, 1)
            # It doesn't change the behavior but saves to the 1/deg memory use.
            indices = upper_bound_cpp.upper_bound_par(t.squeeze(-1)).add_(-1)
            # indices = (t.permute(0, 2, 1) <= t).sum(dim=-1).add_(-1)
            # assert torch.all(torch.eq(another_indices, indices))
            return {f"{groupby}_deg_indices": indices}
        return _compute_deg_indices
    g = g.local_var()
    g.edata["timestamp"] = g.edata["timestamp"].cpu()

    src_deg_indices = _group_func_wrapper(groupby="src")
    dst_deg_indices = _group_func_wrapper(groupby="dst")
    g.group_apply_edges(group_by="src", func=src_deg_indices)
    g.group_apply_edges(group_by="dst", func=dst_deg_indices)
    return {"src_deg_indices": g.edata["src_deg_indices"],
            "dst_deg_indices": g.edata["dst_deg_indices"]}


@timeit
def _deg_indices_full(g):
    u, v, eids = g.out_edges(g.nodes(), 'all')
    etime = g.edata["timestamp"][eids].cpu()
    degs = g.out_degrees()
    src_deg_indices = upper_bound_cpp.upper_bound_full(u, etime, degs)
    # eids is a permutation of torch.arange(g.number_of_edges())
    # we can reverse the permutation by torch.argsort: eids[torch.argsort(eids)] == torch.arange(g.numer_of_edges())
    src_deg_indices = src_deg_indices[torch.argsort(eids)]
    u, v, eids = g.in_edges(g.nodes(), 'all')
    etime = g.edata["timestamp"][eids].cpu()
    degs = g.out_degrees()
    dst_deg_indices = upper_bound_cpp.upper_bound_full(v, etime, degs)
    dst_deg_indices = dst_deg_indices[torch.argsort(eids)]
    return {"src_deg_indices": src_deg_indices.add_(-1),
            "dst_deg_indices": dst_deg_indices.add(-1)}


@timeit
def _par_deg_indices_full(g):
    u, v, eids = g.out_edges(g.nodes(), 'all')
    etime = g.edata["timestamp"][eids].cpu()
    degs = g.out_degrees()
    src_deg_indices = upper_bound_cpp.upper_bound_full_par(u, etime, degs)
    # eids is a permutation of torch.arange(g.number_of_edges())
    # we can reverse the permutation by torch.argsort: eids[torch.argsort(eids)] == torch.arange(g.numer_of_edges())
    src_deg_indices = src_deg_indices[torch.argsort(eids)]
    u, v, eids = g.in_edges(g.nodes(), 'all')
    etime = g.edata["timestamp"][eids].cpu()
    degs = g.out_degrees()
    dst_deg_indices = upper_bound_cpp.upper_bound_full_par(v, etime, degs)
    dst_deg_indices = dst_deg_indices[torch.argsort(eids)]
    return {"src_deg_indices": src_deg_indices.add_(-1),
            "dst_deg_indices": dst_deg_indices.add(-1)}
