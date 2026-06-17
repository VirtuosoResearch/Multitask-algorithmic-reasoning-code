"""Star-graph DFS generalization experiment.

A fixed GNN is trained on MANY star-graph instances to predict depth-first
search, then evaluated on UNSEEN star graphs (the standard neural-algorithmic-
reasoning train/test setup):

  * Train : many random star graphs (random size n in [min_nodes, max_nodes],
            random center). For each, the model predicts DFS's intermediate
            labels (the hints) and final output (the parent pointer pi).
  * Test  : freshly sampled, unseen star graphs (same size range by default, or
            a larger range for size generalization).

The prediction is treated as multi-label node classification (every DFS label is
node-/edge-located): encode the graph, run a *fixed* number of message-passing
rounds once, then predict -- per node / per edge -- the whole DFS label
trajectory (all T time steps of color, d, f, pi_h, s_prev, s, u, v, s_last,
time) plus the output pointer pi. This runs `SALSACLRSModel` with
MODEL.SINGLE_PASS=True, which uses `SinglePassTraceModel` instead of the
recurrent algorithm-executor -- so there is no backprop-through-time and memory
is O(message-passing rounds), not O(trajectory length).

Graphs vary in size, so their trajectory lengths T=3n differ. The decoder heads
are sized to the largest T (MODEL.TRACE_LEN) and predictions are sliced to each
batch's trajectory length; the per-graph loss mask handles shorter graphs.

Almost everything is reused from gnn_experiments: `clrs.algorithms.dfs` +
the CLRS sampler generate ground-truth probes, `data_utils.utils.to_data` builds
the PyG graph, `CLRSDataModule` batches, `SALSACLRSModel` / `CLRSLoss` / the
metrics / `train.train` are unchanged.

Example:
    python star_graph_experiment.py --cfg ./configs/GIN.yml \
        --min_nodes 8 --max_nodes 16 --num_train 2000 --num_test 256 \
        --gnn_layers 4 --epochs 100 --devices 0
    # size generalization: train small, test larger
    #   --test_min_nodes 24 --test_max_nodes 32
"""

import os
import csv
import copy
import argparse

import numpy as np
import torch
from loguru import logger
try:
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl

import clrs
from clrs._src.samplers import DfsSampler

from core.config import load_cfg
from core.module import SALSACLRSModel
from data_utils.utils import to_data
from data_utils.data_loader import CLRSDataModule
from train import train  # reuse the existing train / validate / test loop

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)


# --------------------------------------------------------------------------- #
# Star graph + DFS sample generation
# --------------------------------------------------------------------------- #
def make_star_adjacency(n: int, center: int) -> np.ndarray:
    """Adjacency matrix of a star graph on ``n`` nodes with the given center.

    ``center`` is connected to every leaf in both directions (a symmetric star)
    so DFS can traverse from the center out to the leaves and back.
    """
    if not 0 <= center < n:
        raise ValueError(f"center {center} out of range for n={n}")
    A = np.zeros((n, n), dtype=np.float32)
    leaves = [i for i in range(n) if i != center]
    A[center, leaves] = 1.0
    A[leaves, center] = 1.0
    return A


class _OneStarDFSSampler(DfsSampler):
    """Generates the DFS probes for a single given star adjacency.

    ``num_samples=1`` (a one-element pool) means the probes keep this graph's
    true trajectory length T -- nothing pads it to a larger batch/pool maximum.
    Reuses the CLRS ``DfsSampler`` for running ``dfs`` and splitting probes.
    """

    def __init__(self, adjacency: np.ndarray, seed=0):
        self._adjacency = adjacency
        super().__init__(clrs.algorithms.dfs, clrs.SPECS["dfs"], num_samples=1,
                         seed=seed)

    def _sample_data(self, length=None, p=(0.5,)):  # signature matches base
        return [self._adjacency.copy()]


def apply_node_index_encoding(data, node_index_encoding):
    """Control whether the model sees CLRS's node-order/index feature."""
    if node_index_encoding == "clrs_pos":
        return data
    if node_index_encoding == "constant":
        data.pos = torch.zeros_like(data.pos)
        return data
    raise ValueError(f"Unknown node_index_encoding: {node_index_encoding}")


def make_star_dfs_data(n, rng, shuffle_labels=True, use_hints=True,
                       use_complete_graph=False, node_index_encoding="clrs_pos"):
    """Build one `CLRSData` graph: a star + its DFS probes (true length T).

    The star is built canonically (center = node 0). If ``shuffle_labels``, the
    node labels are permuted at random -- the adjacency is relabelled by a random
    permutation and DFS is recomputed on the relabelled graph -- so the
    center/leaf identities (and the `pos` ordering DFS breaks ties on) are not
    tied to fixed node indices.
    """
    A = make_star_adjacency(n, center=0)
    if shuffle_labels:
        perm = rng.permutation(n)
        A = A[np.ix_(perm, perm)]
    sampler = _OneStarDFSSampler(A)
    feedback = sampler.next(batch_size=1)
    data = to_data(
        feedback.features.inputs,
        feedback.features.hints,
        feedback.outputs,
        use_hints=use_hints,
        use_complete_graph=use_complete_graph,
    )
    return apply_node_index_encoding(data, node_index_encoding)


def build_specs(raw_specs, sample):
    """CLRS spec -> gnn spec, adding the categorical dimension.

    Mirrors `CLRSDataset._update_specs`: keep only probes that ended up in the
    graph and record the number of categories for categorical probes (color).
    """
    present = set(sample.keys())
    specs = {}
    for key, (stage, location, type_) in raw_specs.items():
        if key not in present:
            continue
        if type_ == clrs.Type.CATEGORICAL:
            specs[key] = (stage, location, clrs.Type.CATEGORICAL, sample[key].shape[-1])
        else:
            specs[key] = (stage, location, type_, None)
    return specs


# --------------------------------------------------------------------------- #
# Dataset: many random star graphs
# --------------------------------------------------------------------------- #
class StarDFSDataset(torch.utils.data.Dataset):
    """In-memory dataset of random star-graph DFS instances.

    Each item is a distinct star (random size in [min_nodes, max_nodes]) with,
    by default, randomly shuffled node labels. Different ``seed`` values give
    disjoint (unseen) splits. Exposes the ``.specs`` / ``.nickname`` attributes
    `CLRSDataModule` expects.
    """

    def __init__(self, num_graphs, min_nodes, max_nodes, seed,
                 shuffle_labels=True, use_complete_graph=False,
                 node_index_encoding="clrs_pos", nickname=None):
        rng = np.random.default_rng(seed)
        self.data_list = []
        for _ in range(num_graphs):
            n = int(rng.integers(min_nodes, max_nodes + 1))
            self.data_list.append(
                make_star_dfs_data(n, rng, shuffle_labels=shuffle_labels,
                                   use_complete_graph=use_complete_graph,
                                   node_index_encoding=node_index_encoding))
        self.nickname = nickname
        self.node_index_encoding = node_index_encoding
        self.specs = build_specs(clrs.SPECS["dfs"], self.data_list[0])
        self.max_length = max(int(d.length) for d in self.data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # Return a copy: the collater pads hints in place, which would otherwise
        # corrupt the stored graph across batches/epochs.
        return copy.deepcopy(self.data_list[idx])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", type=str, required=True, help="Path to config file")

    # star graphs / dataset
    parser.add_argument("--min_nodes", type=int, default=8,
                        help="Min nodes per (train/val) star graph")
    parser.add_argument("--max_nodes", type=int, default=16,
                        help="Max nodes per (train/val) star graph")
    parser.add_argument("--test_min_nodes", type=int, default=None,
                        help="Min nodes per test star graph (default: same as train)")
    parser.add_argument("--test_max_nodes", type=int, default=None,
                        help="Max nodes per test star graph (default: same as train)")
    parser.add_argument("--num_train", type=int, default=2000)
    parser.add_argument("--num_val", type=int, default=256)
    parser.add_argument("--num_test", type=int, default=256)
    parser.add_argument("--shuffle_labels", action=argparse.BooleanOptionalAction, default=True,
                        help="Randomly permute each graph's node labels (default on). "
                             "Off => canonical stars with center = node 0.")
    parser.add_argument("--node_index_encoding", choices=("clrs_pos", "constant"),
                        default="clrs_pos",
                        help="Use CLRS pos node-order input or replace it with a constant.")
    parser.add_argument("--use_complete_graph", action="store_true",
                        help="Message-pass on the complete graph instead of the star edges")

    # training
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--loss_weight_output", type=float, default=1.0,
                        help="Weight on the output (pi) loss")
    parser.add_argument("--loss_weight_hint", type=float, default=1.0,
                        help="Weight on the intermediate (hint) loss; >0 learns the DFS hints")
    parser.add_argument("--loss_weight_hidden", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)

    # model
    parser.add_argument(
        "--execution_mode",
        choices=("recurrent", "direct_output", "single_pass"),
        default="single_pass",
        help="Recurrently execute DFS, predict only pi, or predict the full trace once",
    )
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--gnn_layers", type=int, default=4,
                        help="Number of message-passing steps/layers")
    parser.add_argument("--enable_gru", action="store_true")

    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--devices", type=int, nargs="+", default=[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name_suffix", type=str, default=None,
                        help="Optional suffix used to keep checkpoints/results distinct")
    return parser.parse_args()


def main():
    args = parse_args()
    test_min_nodes = args.test_min_nodes if args.test_min_nodes is not None else args.min_nodes
    test_max_nodes = args.test_max_nodes if args.test_max_nodes is not None else args.max_nodes

    # ---- config (mirrors train.py) ----
    cfg = load_cfg(args.cfg)
    cfg.ALGORITHM = "dfs_star"
    cfg.TRAIN.OPTIMIZER.NAME = args.optimizer
    cfg.TRAIN.OPTIMIZER.LR = args.lr
    cfg.TRAIN.LOSS.OUTPUT_LOSS_WEIGHT = args.loss_weight_output
    cfg.TRAIN.LOSS.HINT_LOSS_WEIGHT = args.loss_weight_hint
    cfg.TRAIN.LOSS.HIDDEN_LOSS_WEIGHT = args.loss_weight_hidden
    cfg.TRAIN.BATCH_SIZE = args.batch_size
    cfg.TRAIN.MAX_EPOCHS = args.epochs
    cfg.MODEL.EXECUTION_MODE = args.execution_mode
    pos_suffix = args.node_index_encoding.replace("clrs_pos", "clrs")
    cfg.RUN_NAME += f"-star-dfs-{args.execution_mode.replace('_', '-')}-pos-{pos_suffix}"
    if args.run_name_suffix:
        cfg.RUN_NAME += f"-{args.run_name_suffix}"

    cfg.MODEL.HIDDEN_DIM = args.hidden_dim
    cfg.MODEL.MSG_PASSING_STEPS = args.gnn_layers
    cfg.MODEL.SINGLE_PASS = args.execution_mode == "single_pass"
    if args.execution_mode == "direct_output":
        cfg.TRAIN.LOSS.HINT_LOSS_WEIGHT = 0.0
        cfg.MODEL.GRU.ENABLE = False
    else:
        cfg.MODEL.GRU.ENABLE = args.enable_gru
    if args.use_complete_graph:
        cfg.MODEL.PROCESSOR.KWARGS[0].update({"edge_dim": 128})

    logger.info("Generating star-graph DFS datasets...")
    torch.set_float32_matmul_precision("medium")

    # Distinct seeds -> disjoint (unseen) train / val / test instances.
    train_ds = StarDFSDataset(args.num_train, args.min_nodes, args.max_nodes,
                              seed=args.seed, shuffle_labels=args.shuffle_labels,
                              use_complete_graph=args.use_complete_graph,
                              node_index_encoding=args.node_index_encoding,
                              nickname="train")
    val_ds = StarDFSDataset(args.num_val, args.min_nodes, args.max_nodes,
                            seed=args.seed + 1, shuffle_labels=args.shuffle_labels,
                            use_complete_graph=args.use_complete_graph,
                            node_index_encoding=args.node_index_encoding,
                            nickname="val")
    test_ds = StarDFSDataset(args.num_test, test_min_nodes, test_max_nodes,
                             seed=args.seed + 2, shuffle_labels=args.shuffle_labels,
                             use_complete_graph=args.use_complete_graph,
                             node_index_encoding=args.node_index_encoding,
                             nickname=f"test_n{test_min_nodes}-{test_max_nodes}")
    specs = train_ds.specs
    first_pos = train_ds.data_list[0].pos
    logger.info(
        "Node index encoding: {}; first train pos unique values: {}; min: {:.4f}; max: {:.4f}",
        args.node_index_encoding,
        int(torch.unique(first_pos).numel()),
        float(first_pos.min()),
        float(first_pos.max()),
    )

    if args.execution_mode == "single_pass":
        cfg.MODEL.TRACE_LEN = max(
            train_ds.max_length, val_ds.max_length, test_ds.max_length
        )
        logger.info(f"Max DFS trajectory length: {cfg.MODEL.TRACE_LEN}")
    logger.info(f"Execution mode: {args.execution_mode}")
    logger.info(f"Specs: {specs}")
    print("Using processor", cfg.MODEL.PROCESSOR.NAME)

    metrics = {}
    rng = np.random.default_rng(args.seed)
    for run in range(args.runs):
        run_seed = int(rng.integers(0, 10000))
        pl.seed_everything(run_seed)
        logger.info(f"Run {run}, seed {run_seed}")

        datamodule = CLRSDataModule(
            train_dataset=train_ds, val_datasets=val_ds, test_datasets=test_ds,
            batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.TRAIN.NUM_WORKERS,
            test_batch_size=cfg.TEST.BATCH_SIZE)
        model = SALSACLRSModel(specs=specs, cfg=cfg)

        results = train(model, datamodule, cfg, specs, seed=run_seed,
                        checkpoint_dir="./saved/", devices=args.devices,
                        run_name=cfg.RUN_NAME + f"-run{run}")
        for key in results:
            metrics.setdefault(key, []).append(results[key])

    for key in metrics:
        logger.info("{}: {:.4f} +/- {:.4f}".format(
            key, np.mean(metrics[key]), np.std(metrics[key])))

    logger.info("Saving results...")
    results_dir = f"results/{cfg.ALGORITHM}/{cfg.RUN_NAME}"
    os.makedirs(results_dir, exist_ok=True)
    metrics = {k: float(np.mean(v)) for k, v in metrics.items()}
    with open(os.path.join(results_dir, "results.csv"), "w") as f:
        writer = csv.DictWriter(f, metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)


if __name__ == "__main__":
    main()
