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


@timeit
def _latest_edge(g, u, t, mode="in"):
    # Assume the timestamp is non-descending for each node, otherwise the cpp
    # extension will give wrong answers.
    assert torch.all(t[1:] - t[:-1] >=
                     0), "Timestamp tensor is not non-descending."
    sg = dgl.DGLGraph()
    sg.add_nodes(g.number_of_nodes())
    sg.add_edges(u, u)  # each edge represents a query for (u, t)
    sg.edata["timestamp"] = t.cpu().clone().detach()

    nodes = u.unique()
    degs = g.in_degrees(nodes)
    eids = g.in_edges(
        nodes, 'eid') if mode == "in" else g.out_edges(nodes, 'eid')
    etime = g.edata["timestamp"][eids].to(sg.edata["timestamp"])

    sdegs = sg.in_degrees(nodes)
    seids = sg.in_edges(nodes, 'eid')
    setime = sg.edata["timestamp"][seids]
    refer_eids = upper_bound_cpp.refer_latest_edge(
        nodes, degs, eids, etime, nodes, sdegs, seids, setime)
    return refer_eids[torch.argsort(seids)]


@timeit
def LatestNodeInteractionFinder(g, u, t, mode="in"):
    """for each `(u, t)`, find the latest in/out interaction of `u` in graph `g`.

    Returns
    ----------
    eids : tensor
    """
    eids = torch.full((u.shape[0],), -1, dtype=torch.int64)
    g = g.local_var()
    # Attention! When `ts` is a float tensor, `ti` is a double scalar, `ts < ti` will cast `ti` to `float`, resulting in precision loss. So we use `ts.to(t)` in the first.
    ts = g.edata["timestamp"].cpu().to(t)
    for i, (ui, ti) in enumerate(zip(u, t)):
        if mode == "in":
            edges = g.in_edges(ui, 'eid')
        else:
            edges = g.out_edges(ui, 'eid')
        mask = torch.where(ts[edges] < ti)[0]
        if mask.shape[0] > 0:
            eids[i] = torch.max(edges[mask])
        else:
            eids[i] = edges[0]
    return eids
