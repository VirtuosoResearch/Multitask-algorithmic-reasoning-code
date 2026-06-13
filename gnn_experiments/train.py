import os
import sys
from typing import Optional, Any, Dict
from collections import defaultdict
import csv

import torch
from loguru import logger
import lightning.pytorch as pl
import argparse
import os
import math

from core.module import SALSACLRSModel
from core.config import load_cfg
from core.utils import NaNException
from data_utils.data_loader import CLRSData, CLRSDataset, CLRSDataModule
from data_utils.utils import GRAPH_TOPOLOGIES
import numpy as np

logger.remove()
logger.add(sys.stderr, level="INFO")

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

''' 
TODO:
- Why the NodePointerDecoder is defined on the edges 
- Why the Depth first search has no input of the source nodes: 's_prev': ('hint', 'node', 'pointer'), 's': ('hint', 'node', 'mask_one'), 'u': ('hint', 'node', 'mask_one'), 'v': ('hint', 'node', 'mask_one'), 's_last': ('hint', 'node', 'mask_one')}
- Calculate loss on multiple outputs
'''


def train(model, datamodule, cfg, specs, seed=42, checkpoint_dir=None, devices=[0], run_name=None):
    callbacks = []
    # checkpointing
    if checkpoint_dir is not None:
        ckpt_cbk = pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join("./checkpoints", str(cfg.ALGORITHM), run_name), 
            monitor="val_node_accuracy", mode="max", filename=f'seed{seed}-{{epoch}}-{{step}}', save_top_k=1, verbose=True)
        callbacks.append(ckpt_cbk)

    # early stopping
    early_stop_cbk = pl.callbacks.EarlyStopping(monitor="val_node_accuracy", patience=cfg.TRAIN.EARLY_STOPPING_PATIENCE, mode="max", verbose=True)
    callbacks.append(early_stop_cbk)

    # Setup trainer
    trainer = pl.Trainer(
        devices=devices,
        enable_checkpointing=True,
        callbacks=[ckpt_cbk, early_stop_cbk],
        max_epochs=cfg.TRAIN.MAX_EPOCHS,
        logger=None,
        accelerator="gpu",
        log_every_n_steps=5,
        gradient_clip_val=cfg.TRAIN.GRADIENT_CLIP_VAL,
        reload_dataloaders_every_n_epochs=datamodule.reload_every_n_epochs,
        precision= cfg.TRAIN.PRECISION,
    )

    # Load checkpoint
    if cfg.TRAIN.LOAD_CHECKPOINT is not None:
        logger.info(f"Loading checkpoint from {cfg.TRAIN.LOAD_CHECKPOINT}")
        model = SALSACLRSModel.load_from_checkpoint(cfg.TRAIN.LOAD_CHECKPOINT, cfg=cfg, specs=specs)

    # Train
    if cfg.TRAIN.ENABLE:
        try:
            logger.info("Starting training...")
            trainer.fit(model, datamodule=datamodule)
        except NaNException:
            logger.info(f"NaN detected, trying to recover from {ckpt_cbk.best_model_path}...")
            try:
                trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_cbk.best_model_path)
            except NaNException:
                logger.info("Recovery failed, stopping training...")

    # Load best model
    if cfg.TRAIN.LOAD_CHECKPOINT is None and cfg.TRAIN.ENABLE:
        logger.info(f"Best model path: {ckpt_cbk.best_model_path}")
        model = SALSACLRSModel.load_from_checkpoint(ckpt_cbk.best_model_path)

    # Test
    logger.info("Testing best model...")
    results = trainer.validate(model, datamodule=datamodule)[0]
    test_results = trainer.test(model, datamodule=datamodule)[0]
    results.update(test_results)
    print(results)
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="Path to config file")
    # datasets
    parser.add_argument("--algorithm", type=str, required=True)
    parser.add_argument("--graph_topology",       type=str,  choices=GRAPH_TOPOLOGIES,  default="dataset")
    parser.add_argument("--star_center", type=int, default=0)
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=None,
        help="Generate star-graph samples with exactly this many nodes",
    )
    parser.add_argument(
        "--shuffle_node_labels",
        action="store_true",
        help="Apply a new random node-label permutation to every star sample",
    )
    parser.add_argument("--use_complete_graph", action="store_true")
    # training
    parser.add_argument("--optimizer", type=str, default="adamw", help="Optimizer")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--loss_weight_output", type=float, default=1.0, help="Output loss weight.")
    parser.add_argument("--loss_weight_hint", type=float, default=1.0, help="Hint loss weight.")
    parser.add_argument("--loss_weight_hidden", type=float, default=0.01, help="KL loss weight.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    # model
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--gnn_layers", type=int, default=2, help="Message passing steps")
    parser.add_argument("--enable_gru", action="store_true", help="Enable GRU")

    parser.add_argument("--runs", type=int, default=1, help="Number of runs")
    parser.add_argument("--devices", type=int, nargs="+", default=[0], help="Devices to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    if args.use_complete_graph:
        if args.graph_topology != "dataset":
            parser.error(
                "--use_complete_graph cannot be combined with --graph_topology"
            )
        args.graph_topology = "complete"
    if args.num_nodes is not None:
        if args.graph_topology != "star":
            parser.error("--num_nodes requires --graph_topology star")
        if args.num_nodes < 2:
            parser.error("--num_nodes must be at least 2")
        if not 0 <= args.star_center < args.num_nodes:
            parser.error(
                "--star_center must be between 0 and --num_nodes - 1"
            )
    if args.shuffle_node_labels and args.num_nodes is None:
        parser.error("--shuffle_node_labels requires --num_nodes")

    # load config
    cfg = load_cfg(args.cfg)
    cfg.ALGORITHM = args.algorithm
    cfg.TRAIN.OPTIMIZER.NAME = args.optimizer
    cfg.TRAIN.OPTIMIZER.LR = args.lr
    cfg.TRAIN.LOSS.OUTPUT_LOSS_WEIGHT = args.loss_weight_output
    cfg.TRAIN.LOSS.HINT_LOSS_WEIGHT = args.loss_weight_hint
    cfg.TRAIN.LOSS.HIDDEN_LOSS_WEIGHT = args.loss_weight_hidden
    cfg.TRAIN.BATCH_SIZE = args.batch_size
    cfg.TRAIN.MAX_EPOCHS = args.epochs
    cfg.RUN_NAME = cfg.RUN_NAME+"-hints"
    if args.graph_topology != "dataset":
        cfg.RUN_NAME += f"-{args.graph_topology}"
        if args.graph_topology == "star":
            cfg.RUN_NAME += f"-center{args.star_center}"
            if args.num_nodes is not None:
                cfg.RUN_NAME += f"-n{args.num_nodes}"
            if args.shuffle_node_labels:
                cfg.RUN_NAME += "-shuffled"

    # update model configs 
    cfg.MODEL.HIDDEN_DIM = args.hidden_dim
    cfg.MODEL.MSG_PASSING_STEPS = args.gnn_layers
    cfg.MODEL.GRU.ENABLE = args.enable_gru
    if args.graph_topology == "complete":
        cfg.MODEL.PROCESSOR.KWARGS[0].update({"edge_dim": 128})
    
    logger.info("Starting run...")
    torch.set_float32_matmul_precision('medium')
    logger.info(
        f"Using {args.graph_topology} input graph topology"
        + (
            f" with center node {args.star_center}"
            + (
                f" and {args.num_nodes} nodes"
                if args.num_nodes is not None
                else ""
            )
            + (
                " with labels reshuffled per sample"
                if args.shuffle_node_labels
                else ""
            )
            if args.graph_topology == "star"
            else ""
        )
    )

    # load datasets
    dataset_kwargs = {
        "algorithm": args.algorithm,
        "graph_topology": args.graph_topology,
        "star_center": args.star_center,
        "num_nodes": args.num_nodes,
        "shuffle_node_labels": args.shuffle_node_labels,
    }
    train_ds = CLRSDataset(
        split="train",
        num_samples=1000,
        seed=args.seed,
        **dataset_kwargs,
    )
    val_ds = CLRSDataset(
        split="val",
        num_samples=32,
        seed=args.seed + 1,
        **dataset_kwargs,
    )
    test_ds = CLRSDataset(
        split="test",
        num_samples=32,
        seed=args.seed + 2,
        **dataset_kwargs,
    )
    specs = train_ds.specs
    is_weighted = hasattr(train_ds[0], "weights")
    if is_weighted and cfg.MODEL.PROCESSOR.NAME == "GINConv":
        cfg.MODEL.PROCESSOR.NAME = "GINEConv"
    print("Using processor", cfg.MODEL.PROCESSOR.NAME)
    
    # load model
    metrics = {}; rng = np.random.default_rng(args.seed)
    for run in range(args.runs):
        # set seed
        run_seed = rng.integers(0, 10000)
        pl.seed_everything(run_seed)
        logger.info(f"Using seed {run_seed}")

        datamodule = CLRSDataModule(train_dataset=train_ds, val_datasets=val_ds, test_datasets=test_ds, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.TRAIN.NUM_WORKERS, test_batch_size=cfg.TEST.BATCH_SIZE)
        datamodule.val_dataloader()
        model = SALSACLRSModel(specs=train_ds.specs, cfg=cfg)

        ckpt_dir = "./saved/"
        results = train(model, datamodule, cfg, train_ds.specs, seed = run_seed, checkpoint_dir=ckpt_dir, devices=args.devices, run_name=cfg.RUN_NAME + f"-run{run}")

        for key in results:
            if key not in metrics:
                metrics[key] = []
            metrics[key].append(results[key])
    # Log results
    for key in metrics:
        logger.info("{}: {:.4f} +/- {:.4f}".format(key, np.mean(metrics[key]), np.std(metrics[key])))
    
    logger.info("Saving results...")
    results_dir = f"results/{cfg.ALGORITHM}/{cfg.RUN_NAME}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    # write csv
    metrics = {k: np.mean(v) for k, v in metrics.items()}
    with open(os.path.join(results_dir, f"results.csv"), "w") as f:
        writer = csv.DictWriter(f, metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)
