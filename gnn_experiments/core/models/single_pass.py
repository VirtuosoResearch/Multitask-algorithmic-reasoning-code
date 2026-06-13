"""Single-pass trace model.

A drop-in alternative to `EncodeProcessDecode` that does NOT unroll the
algorithm. Instead it runs a fixed number of message-passing rounds once and
then predicts, per node / per edge, the *entire* label trajectory (all
``TRACE_LEN`` time steps) in a single shot -- i.e. the DFS labelling problem
treated as plain multi-label node classification.

Because there is no recurrence over the trajectory, there is no backprop
through ~3n steps, so memory is O(rounds) instead of O(trajectory length).

The forward signature and the shapes of the returned ``output`` / ``hints``
dicts match `EncodeProcessDecode` exactly, so `CLRSLoss`, the metrics and the
`SALSACLRSModel` training loop are reused unchanged. Each predicted label has
the same shape `stack_hints` would have produced in the recurrent model:

  * node scalar  (d, f)              -> (N, T)
  * node mask_one (s, u, v, s_last)  -> (N, T)   log-softmax over nodes/graph
  * node categorical (color)         -> (N, C, T) log-softmax over C
  * edge pointer (pi_h, s_prev)      -> (E, T)   log-softmax over out-edges
  * graph scalar (time)              -> (G, T)
  * output pointer (pi)              -> (E,)     log-softmax over out-edges
"""

import torch
import torch.nn as nn
import torch_scatter
from torch_geometric.nn import global_mean_pool

from .encoder import Encoder
from .processor import Processor


class _TraceHead(nn.Module):
    """Predicts the full ``width``-step trajectory of one label from node
    embeddings, normalised the same way as the corresponding CLRS decoder.

    The head *kind* is derived from (loc, type_) the same way `_DECODER_MAP`
    does -- note in particular that ``pointer`` labels, although the spec marks
    them ``node``-located, are decoded on edges (per-source out-edge softmax),
    exactly like `NodePointerDecoder`.
    """

    def __init__(self, loc, type_, cat_dim, hidden_dim, width):
        super().__init__()
        self.type_ = type_
        self.cat_dim = cat_dim if type_ == "categorical" else 1
        self.width = width  # number of time steps (T for hints, 1 for outputs)

        if type_ == "pointer":
            self.kind = "edge"
        elif loc == "graph":
            self.kind = "graph"
        else:  # node-located scalar / mask / mask_one / categorical
            self.kind = "node"

        if self.kind == "edge":
            self.source_lin = nn.Linear(hidden_dim, hidden_dim)
            self.target_lin = nn.Linear(hidden_dim, hidden_dim)
            self.edge_lin = nn.Linear(hidden_dim, width)
        else:
            self.lin = nn.Linear(hidden_dim, self.cat_dim * width)

    def forward(self, hidden, edge_index, batch_assignment):
        if self.kind == "edge":  # pointer: score out-edges, softmax per source
            zs = self.source_lin(hidden)
            zt = self.target_lin(hidden)
            out = self.edge_lin(zs[edge_index[0]] * zt[edge_index[1]])  # (E, width)
            out = torch_scatter.scatter_log_softmax(out, edge_index[0], dim=0)
            return out.squeeze(-1) if self.width == 1 else out

        if self.kind == "graph":
            out = global_mean_pool(self.lin(hidden), batch_assignment)  # (G, width)
            return out.squeeze(-1) if self.width == 1 else out

        # node-located
        out = self.lin(hidden)
        if self.type_ == "categorical":
            out = out.view(-1, self.cat_dim, self.width)  # (N, C, T)
            return torch.log_softmax(out, dim=1)
        if self.type_ == "mask_one":  # which node, per step -> softmax over nodes/graph
            return torch_scatter.scatter_log_softmax(out, batch_assignment, dim=0)
        return out.squeeze(-1) if self.width == 1 else out  # scalar / mask


class TraceDecoder(nn.Module):
    """Builds one `_TraceHead` per output/hint label."""

    def __init__(self, specs, hidden_dim, trace_len):
        super().__init__()
        self.heads = nn.ModuleDict()
        self.stage_keys = {"outputs": [], "hints": []}
        for key, (stage, loc, type_, cat_dim) in specs.items():
            if stage == "input":
                continue
            width = trace_len if stage == "hint" else 1
            self.heads[key] = _TraceHead(loc, type_, cat_dim, hidden_dim, width)
            self.stage_keys["outputs" if stage == "output" else "hints"].append(key)

    def forward(self, hidden, batch, stage):
        out = {}
        for key in getattr(batch, stage):
            out[key] = self.heads[key](hidden, batch.edge_index, batch.batch)
        return out


class SinglePassTraceModel(nn.Module):
    """Encode -> a few message-passing rounds -> predict the whole trace."""

    def __init__(self, specs, cfg):
        super().__init__()
        self.cfg = cfg
        self.specs = specs
        self.has_randomness = "randomness" in specs

        self.encoder = Encoder(specs, cfg.MODEL.HIDDEN_DIM)
        self.processor = nn.ModuleList(
            [Processor(cfg, self.has_randomness) for _ in range(cfg.MODEL.MSG_PASSING_STEPS)])

        if cfg.MODEL.TRACE_LEN <= 0:
            raise ValueError("MODEL.TRACE_LEN must be set to the trajectory length T "
                             "for the single-pass model.")
        self.decoder = TraceDecoder(specs, cfg.MODEL.HIDDEN_DIM, cfg.MODEL.TRACE_LEN)

    def forward(self, batch):
        input_hidden, randomness = self.encoder(batch)
        hidden = input_hidden
        for k in range(self.cfg.MODEL.MSG_PASSING_STEPS):
            hidden = self.processor[k](
                input_hidden, hidden, hidden,
                batch_assignment=batch.batch, edge_index=batch.edge_index,
                randomness=randomness[:, k] if randomness is not None else None)

        output = self.decoder(hidden, batch, "outputs")
        hints = self.decoder(hidden, batch, "hints")
        # Heads emit a fixed TRACE_LEN width; graphs in this batch only run for
        # batch.length.max() steps, so slice the (last) time axis to match the
        # (batch-padded) targets. The per-graph loss mask handles the rest.
        max_len = int(batch.length.max().item())
        if max_len < self.cfg.MODEL.TRACE_LEN:
            hints = {k: v[..., :max_len] for k, v in hints.items()}
        return output, hints, hidden
