from multiprocessing import Process, Queue
import queue
import threading
import numpy as np
from numpy.core.shape_base import block
import torch
from torch.utils.data import Dataset, DataLoader
import networkx as nx
from numba import jit, prange
import bisect


@jit
def find_before_nb(src_idx, cut_time, node_idx_l, node_ts_l, edge_idx_l,
                   off_set_l):
    """

    Params
    ------
    src_idx: int
    cut_time: float
    """

    neighbors_idx = node_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
    neighbors_ts = node_ts_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
    neighbors_e_idx = edge_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]

    if len(neighbors_idx) == 0 or len(neighbors_ts) == 0:
        return neighbors_idx, neighbors_e_idx, neighbors_ts

    left = 0
    right = len(neighbors_idx) - 1

    while left + 1 < right:
        mid = (left + right) // 2
        curr_t = neighbors_ts[mid]
        if curr_t < cut_time:
            left = mid
        else:
            right = mid
    if neighbors_ts[right] < cut_time:
        return neighbors_idx[:right], neighbors_e_idx[:right], neighbors_ts[:right]
    else:
        return neighbors_idx[:left], neighbors_e_idx[:left], neighbors_ts[:left]


@jit
def get_temporal_neighbor_nb(src_idx_l, cut_time_l, node_idx_l, node_ts_l,
                             edge_idx_l, off_set_l, num_neighbors, uniform):
    """
    Params
    ------
    src_idx_l: List[int]
    cut_time_l: List[float],
    num_neighbors: int
    """
    out_ngh_node_batch = np.zeros(
        (len(src_idx_l), num_neighbors)).astype(np.int32)
    out_ngh_t_batch = np.zeros(
        (len(src_idx_l), num_neighbors)).astype(np.float32)
    out_ngh_eidx_batch = np.zeros(
        (len(src_idx_l), num_neighbors)).astype(np.int32)

    for i, (src_idx, cut_time) in enumerate(zip(src_idx_l, cut_time_l)):
        # use np.searchsorted
        neighbors_idx = node_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        neighbors_ts = node_ts_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        neighbors_e_idx = edge_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        left = np.searchsorted(neighbors_ts, cut_time, side="left")
        ngh_idx, ngh_eidx, ngh_ts = neighbors_idx[:left], \
                                    neighbors_e_idx[:left], neighbors_ts[:left]

        # ngh_idx, ngh_eidx, ngh_ts = find_before_nb(src_idx, cut_time,
        #                                            node_idx_l, node_ts_l,
        #                                            edge_idx_l, off_set_l)

        if len(ngh_idx) <= 0:
            continue
        # if uniform:
        #     sampled_idx = np.random.randint(0, len(ngh_idx), num_neighbors)
        #     # sampled_idx = np.sort(sampled_idx)

        #     # out_ngh_node_batch[i, :] = ngh_idx[sampled_idx]
        #     # out_ngh_t_batch[i, :] = ngh_ts[sampled_idx]
        #     # out_ngh_eidx_batch[i, :] = ngh_eidx[sampled_idx]

        # else:
        #     prob = ngh_ts - cut_time
        #     prob = np.exp(prob - np.max(prob)) # avoid single zero, or overflow
        #     # assert np.all(prob <= 0)
        #     # prob = np.exp(prob)
        #     prob = prob / prob.sum()
        #     sampled_idx = np.random.choice(len(ngh_idx), size=num_neighbors, p=prob)

        # sampled_idx = np.sort(sampled_idx)
        # out_ngh_node_batch[i, :] = ngh_idx[sampled_idx]
        # out_ngh_t_batch[i, :] = ngh_ts[sampled_idx]
        # out_ngh_eidx_batch[i, :] = ngh_eidx[sampled_idx]

        if uniform:
            sampled_idx = np.random.randint(0, len(ngh_idx), num_neighbors)
            sampled_idx = np.sort(sampled_idx)

            out_ngh_node_batch[i, :] = ngh_idx[sampled_idx]
            out_ngh_t_batch[i, :] = ngh_ts[sampled_idx]
            out_ngh_eidx_batch[i, :] = ngh_eidx[sampled_idx]
        else:
            ngh_ts = ngh_ts[-num_neighbors:]
            ngh_idx = ngh_idx[-num_neighbors:]
            ngh_eidx = ngh_eidx[-num_neighbors:]

            assert (len(ngh_idx) <= num_neighbors)
            assert (len(ngh_ts) <= num_neighbors)
            assert (len(ngh_eidx) <= num_neighbors)

            out_ngh_node_batch[i, num_neighbors - len(ngh_idx):] = ngh_idx
            out_ngh_t_batch[i, num_neighbors - len(ngh_ts):] = ngh_ts
            out_ngh_eidx_batch[i, num_neighbors - len(ngh_eidx):] = ngh_eidx

    return out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch


# ！
class NeighborFinder:
    def __init__(self, adj_list, uniform=False):
        """
        Params
        ------
        node_idx_l: List[int]
        node_ts_l: List[int]
        off_set_l: List[int], such that node_idx_l[off_set_l[i]:off_set_l[i + 1]] = adjacent_list[i]
        """

        node_idx_l, node_ts_l, edge_idx_l, off_set_l = self.init_off_set(
            adj_list)
        self.node_idx_l = node_idx_l
        self.node_ts_l = node_ts_l
        self.edge_idx_l = edge_idx_l

        self.off_set_l = off_set_l

        self.uniform = uniform

    def init_off_set(self, adj_list):
        """
        Params
        ------
        adj_list: List[List[int]]
        
        """
        n_idx_l = []
        n_ts_l = []
        e_idx_l = []
        off_set_l = [0]
        for i in range(len(adj_list)):
            curr = adj_list[i]
            curr = sorted(curr, key=lambda x: x[1])
            n_idx_l.extend([x[0] for x in curr])
            e_idx_l.extend([x[1] for x in curr])
            n_ts_l.extend([x[2] for x in curr])

            off_set_l.append(len(n_idx_l))
        n_idx_l = np.array(n_idx_l)
        n_ts_l = np.array(n_ts_l)
        e_idx_l = np.array(e_idx_l)
        off_set_l = np.array(off_set_l)

        assert (len(n_idx_l) == len(n_ts_l))
        assert (off_set_l[-1] == len(n_ts_l))

        return n_idx_l, n_ts_l, e_idx_l, off_set_l

    def find_before(self, src_idx, cut_time):
        """
    
        Params
        ------
        src_idx: int
        cut_time: float
        """
        return find_before_nb(src_idx, cut_time, self.node_idx_l,
                              self.node_ts_l, self.edge_idx_l, self.off_set_l)

    def get_temporal_neighbor(self, src_idx_l, cut_time_l, num_neighbors=20):
        """
        Params
        ------
        src_idx_l: List[int]
        cut_time_l: List[float],
        num_neighbors: int
        """
        assert (len(src_idx_l) == len(cut_time_l))

        return get_temporal_neighbor_nb(src_idx_l,
                                        cut_time_l,
                                        self.node_idx_l,
                                        self.node_ts_l,
                                        self.edge_idx_l,
                                        self.off_set_l,
                                        num_neighbors=num_neighbors,
                                        uniform=self.uniform)

    def find_k_hop(self, k, src_idx_l, cut_time_l, num_neighbors=20):
        """Sampling the k-hop sub graph
        """
        # x, y, z = self.get_temporal_neighbor(src_idx_l, cut_time_l,
        #                                      num_neighbors)
        node_records = [src_idx_l]
        eidx_records = [np.ones_like(src_idx_l)]
        t_records = [cut_time_l]
        for _ in range(k):
            ngn_node_est, ngh_t_est = node_records[-1], t_records[
                -1]  # [N, *([num_neighbors] * (k - 1))]
            orig_shape = ngn_node_est.shape
            ngn_node_est = ngn_node_est.flatten()
            ngn_t_est = ngh_t_est.flatten()
            out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch = self.get_temporal_neighbor(
                ngn_node_est, ngn_t_est, num_neighbors)
            out_ngh_node_batch = out_ngh_node_batch.reshape(
                *orig_shape, num_neighbors)  # [N, *([num_neighbors] * k)]
            out_ngh_eidx_batch = out_ngh_eidx_batch.reshape(
                *orig_shape, num_neighbors)
            out_ngh_t_batch = out_ngh_t_batch.reshape(*orig_shape,
                                                      num_neighbors)

            node_records.append(out_ngh_node_batch)
            eidx_records.append(out_ngh_eidx_batch)
            t_records.append(out_ngh_t_batch)
        return node_records, eidx_records, t_records


@jit
def bisampling_nb(src_idx_l, cut_time_l, node_idx_l, node_ts_l, edge_idx_l,
                  off_set_l, num_neighbors):
    assert num_neighbors % 2 == 0

    out_ngh_node_batch = np.zeros(
        (len(src_idx_l), num_neighbors)).astype(np.int32)
    out_ngh_t_batch = np.zeros(
        (len(src_idx_l), num_neighbors)).astype(np.float32)
    out_ngh_eidx_batch = np.zeros(
        (len(src_idx_l), num_neighbors)).astype(np.int32)

    for i, (src_idx, cut_time) in enumerate(zip(src_idx_l, cut_time_l)):
        # use np.searchsorted
        neighbors_idx = node_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        neighbors_ts = node_ts_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        neighbors_e_idx = edge_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        left = np.searchsorted(neighbors_ts, cut_time, side="left")
        ngh_idx, ngh_eidx, ngh_ts = neighbors_idx[:left], \
                                    neighbors_e_idx[:left], neighbors_ts[:left]

        if len(ngh_idx) <= 0:
            continue

        half_neighbors = num_neighbors // 2
        # uniform sampling
        # sampled_idx = np.random.randint(0, len(ngh_idx), half_neighbors)
        # sampled_idx = np.sort(sampled_idx)
        # print(i,':',sampled_idx)

        # exp sampling
        x = np.exp(-(cut_time - ngh_ts)) + 1
        probabilities = x / np.sum(x)
        elements = np.array(range(0, len(ngh_idx)))  # !
        sampled_idx = np.random.choice(elements, half_neighbors, replace=True, p=probabilities)
        sampled_idx = np.sort(sampled_idx)
        # print(i, ':', sampled_idx)

        out_ngh_node_batch[i, :half_neighbors] = ngh_idx[sampled_idx]
        out_ngh_t_batch[i, :half_neighbors] = ngh_ts[sampled_idx]
        out_ngh_eidx_batch[i, :half_neighbors] = ngh_eidx[sampled_idx]
        ngh_ts = ngh_ts[-half_neighbors:]
        ngh_idx = ngh_idx[-half_neighbors:]
        ngh_eidx = ngh_eidx[-half_neighbors:]

        assert (len(ngh_idx) <= half_neighbors)
        assert (len(ngh_ts) <= half_neighbors)
        assert (len(ngh_eidx) <= half_neighbors)

        out_ngh_node_batch[i, -len(ngh_idx):] = ngh_idx
        out_ngh_t_batch[i, -len(ngh_idx):] = ngh_ts
        out_ngh_eidx_batch[i, -len(ngh_idx):] = ngh_eidx

    return out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch


# ！
class BiSamplingNFinder(NeighborFinder):
    def __init__(self, adj_list) -> None:
        # binary sampling: first half consists of uniform sampling, and the
        # second hanlf consists of inverse temporal sampling.
        super(BiSamplingNFinder, self).__init__(adj_list, uniform=False)

    def get_temporal_neighbor(self, src_idx_l, cut_time_l, num_neighbors):
        assert (len(src_idx_l) == len(cut_time_l))

        return bisampling_nb(src_idx_l,
                             cut_time_l,
                             self.node_idx_l,
                             self.node_ts_l,
                             self.edge_idx_l,
                             self.off_set_l,
                             num_neighbors=num_neighbors)


# @jit
def global_anchors(edges, n_anchors=20, metric="pagerank"):
    mg = nx.MultiGraph()
    mg.add_edges_from(edges, weight=1.0)
    if metric == "pagerank":
        scores = nx.pagerank_scipy(mg)
    elif metric == "degree":
        scores = nx.degree_centrality(mg)
    elif metric == "closeness":
        scores = nx.closeness_centrality(mg)
    elif metric == "betweenness":
        scores = nx.betweenness_centrality(mg)
    else:
        raise NotImplementedError(metric)
    keys = np.array(list(scores.keys()))
    vals = np.array(list(scores.values()))
    return keys[vals.argsort()[-n_anchors:]]


class NeighborProcess(Process):
    def __init__(self, node_generator, ts_generator, ngh_finder, k, q) -> None:
        super(NeighborProcess, self).__init__()
        self.node_generator = node_generator
        self.ts_generator = ts_generator
        self.ngh_finder = ngh_finder
        self.k = k
        self.q = q
        # self.stream = torch.cuda.Stream()

    def run(self):
        # with torch.cuda.stream(self.stream):
        for src_nodes, ts_l in zip(self.node_generator, self.ts_generator):
            ngh_nodes, ngh_eids, ngh_t = self.ngh_finder.find_k_hop(
                self.k, src_nodes, ts_l)
            self.q.put((ngh_nodes, ngh_eids, ngh_t), block=True)


class Wrapper(object):
    def __init__(self, nodes, batch_size) -> None:
        self.nodes = nodes
        self.batch_size = batch_size
        self.batch_num = 0

    def __iter__(self):
        self.batch_num = 0
        return self

    def __next__(self):
        if self.batch_num * self.batch_size >= len(self.nodes):
            raise StopIteration
        else:
            start = self.batch_num * self.batch_size
            batch = self.nodes[start:start + self.batch_size]
            self.batch_num += 1
            return batch


class NeighborStream(threading.Thread):
    """Transform the numpy arrays to pytorch tensors on the specified device.
    """

    def __init__(self, queues, device, batch_num, maxsize=20) -> None:
        super(NeighborStream, self).__init__()
        self.cpu_queues = queues
        self.device_queues = [queue.Queue(maxsize=maxsize) for _ in queues]
        self.device = device
        self.batch_num = batch_num
        self.stream = torch.cuda.Stream(device)

    def get_queues(self):
        return self.device_queues

    def run(self):
        with torch.cuda.stream(self.stream):
            for _ in range(self.batch_num):
                batch = [q.get(block=True) for q in self.cpu_queues]
                # device_batch = []
                for el, q in zip(batch, self.device_queues):
                    nodes, eids, tbatch = el
                    ngh_nodes_th = [
                        torch.from_numpy(n.flatten()).long().to(self.device)
                        for n in nodes
                    ]
                    ngh_eids_th = [
                        torch.from_numpy(e.flatten()).long().to(self.device)
                        for e in eids
                    ]
                    ngh_t_th = [
                        torch.from_numpy(t.flatten()).float().to(self.device)
                        for t in tbatch
                    ]
                    q.put((ngh_nodes_th, ngh_eids_th, ngh_t_th), block=True)
                    # device_batch.append((ngh_nodes_th, ngh_eids_th, ngh_t_th))


class NeighborLoader(object):
    """Given the ngh_finder and source nodes, `NeighborLoader` provides batch-wise `k`-hop neighbors of a batch of source nodes.
    """

    def __init__(self,
                 ngh_finder,
                 num_layer,
                 src_nodes,
                 ts_list,
                 device,
                 batch_size=128,
                 shuffle=False,
                 gpu_stream=False) -> None:
        self.ngh_finder = ngh_finder
        self.num_layer = num_layer
        self.src_nodes = src_nodes  # a list of source nodes, destination nodes
        self.ts_list = ts_list

        self.device = device
        self.batch_size = batch_size
        self.idx_list = np.arange(len(ts_list))
        self.num = len(self.src_nodes[0])
        assert np.all([len(nodes) == self.num for nodes in src_nodes])

        self.batch_num = 0
        self.shuffle = shuffle
        self.ques = []
        self.gpu_stream = gpu_stream
        self.anchors = None

    def set_anchors(self, anchors):
        self.anchors = anchors

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.idx_list)
            self.src_nodes = [nodes[self.idx_list] for nodes in self.src_nodes]
            self.ts_list = self.ts_list[self.idx_list]
        self.batch_num = 0

        maxsize = 100
        cpu_ques = [Queue(maxsize=maxsize) for _ in range(len(self.src_nodes))]
        for i, nodes in enumerate(self.src_nodes):
            node_generator = Wrapper(nodes, self.batch_size)
            ts_generator = Wrapper(self.ts_list, self.batch_size)
            s = NeighborProcess(node_generator, ts_generator, self.ngh_finder,
                                self.num_layer, cpu_ques[i])
            s.start()

        if self.anchors is not None:
            cpu_ques.append(Queue(maxsize=maxsize))
            anchors = np.expand_dims(self.anchors, axis=0)
            anchors = anchors.repeat(self.num, axis=0).flatten()
            anchor_generator = Wrapper(anchors, self.batch_size * len(self.anchors))

            ts_list = self.ts_list.repeat(len(self.anchors))
            ts_generator = Wrapper(ts_list, self.batch_size * len(self.anchors))
            s = NeighborProcess(anchor_generator, ts_generator, self.ngh_finder,
                                self.num_layer - 1, cpu_ques[-1])
            s.start()

        if self.gpu_stream:
            stream = NeighborStream(cpu_ques,
                                    self.device,
                                    len(self),
                                    maxsize=maxsize)
            self.ques = stream.get_queues()
            stream.start()
        else:
            self.ques = cpu_ques

    def __len__(self):
        r = 1 if len(self.ts_list) % self.batch_size > 0 else 0
        return len(self.ts_list) // self.batch_size + r

    def end(self):
        return self.batch_num * self.batch_size >= self.num

    def next_batch(self):
        if self.end():
            raise StopIteration
        # Here we wait for the queue productions.
        batch = [q.get(block=True) for q in self.ques]
        if self.gpu_stream:
            gpu_batch = batch
        else:
            gpu_batch = []
            for el in batch:
                nodes, eids, tbatch = el
                ngh_nodes_th = [
                    torch.from_numpy(n.flatten()).long().to(self.device)
                    for n in nodes
                ]
                ngh_eids_th = [
                    torch.from_numpy(e.flatten()).long().to(self.device)
                    for e in eids
                ]
                ngh_t_th = [
                    torch.from_numpy(t.flatten()).float().to(self.device)
                    for t in tbatch
                ]
                gpu_batch.append((ngh_nodes_th, ngh_eids_th, ngh_t_th))
        self.batch_num += 1

        if self.anchors is not None:
            self.batch_anchor = gpu_batch[-1]
            return gpu_batch[:-1]
        else:
            return gpu_batch
