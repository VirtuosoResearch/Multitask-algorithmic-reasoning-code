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
from core.models import EncodeProcessDecode, MultitaskEncodeProcessDecode, MMOE_EncodeProcessDecode, BranchedMTL_EncodeProcessDecode
import numpy as np

from data_utils.multitask_data_loader import MultiCLRSDataModule
from data_utils.multitask_data_loader_sample_complexity import MultiCLRSDataModuleSampleComplexity
from core.mtl_module import MultiCLRSModel

logger.remove()
logger.add(sys.stderr, level="INFO")

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def train(model, datamodule, cfg, seed=42, checkpoint_dir=None, devices=[0], algorithms=[], run_name=None, node=None, num_samples=None, use_wandb=False):
    callbacks = []
    algorithm_str = "_".join(algorithms)[:100]
    if use_wandb:
        wandblogger = pl.loggers.WandbLogger(project=cfg.LOGGING.WANDB.PROJECT, entity=cfg.LOGGING.WANDB.ENTITY, group=cfg.LOGGING.WANDB.GROUP, name=f"{run_name}-{algorithm_str}-{node}-{num_samples}")
    else:
        wandblogger = None
    # checkpointing
    if checkpoint_dir is not None:
        ckpt_cbk = pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join("./checkpoints", "_".join(algorithms)[:100], run_name), 
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
        logger=wandblogger,
        accelerator="gpu",
        log_every_n_steps=5,
        gradient_clip_val=cfg.TRAIN.GRADIENT_CLIP_VAL,
        # reload_dataloaders_every_n_epochs=datamodule.reload_every_n_epochs,
        precision= cfg.TRAIN.PRECISION,
    )

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

    # save the checkpoint as a .pt file
    from lightning_fabric.utilities.cloud_io import _load as pl_load
    checkpoint = pl_load(ckpt_cbk.best_model_path, map_location=model.device)
    state_dict = checkpoint["state_dict"]
    state_dict = {k[6:]: v for k, v in state_dict.items()}
    torch.save(state_dict, ckpt_cbk.best_model_path.replace(".ckpt", ".pt"))

    # Load best model
    if cfg.TRAIN.LOAD_CHECKPOINT is None and cfg.TRAIN.ENABLE:
        logger.info(f"Best model path: {ckpt_cbk.best_model_path}")
        mtl_model = model.model
        model = MultiCLRSModel.load_from_checkpoint(ckpt_cbk.best_model_path, 
                                                    task_to_specs=datamodule.task_to_specs, cfg=cfg, model=mtl_model)

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
    parser.add_argument("--algorithms", type=str, nargs="+", default=["bfs"], help="Algorithms")
    parser.add_argument("--use_complete_graph", action="store_true", help="Use complete graph")
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
    parser.add_argument("--enbale_gru_task_wise", action="store_true", help="Enable GRU task wise")

    parser.add_argument("--runs", type=int, default=1, help="Number of runs")
    parser.add_argument("--devices", type=int, nargs="+", default=[0], help="Devices to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--train_mmoe", action="store_true", help="Train MMoE model")
    parser.add_argument("--num_experts", type=int, default=4, help="Number of experts")

    parser.add_argument("--train_branched_network", action="store_true", help="Train branched network")
    parser.add_argument("--tree_config_dir", type=str, default=None, help="Tree config directory")
    parser.add_argument("--branch_layer", type=int, default=-1, help="Branch layer") # only used for pairwise tasks

    # checkpointing 
    parser.add_argument("--load_checkpoint_dir", type=str, default=None, help="Load checkpoint")
    parser.add_argument("--train_layer", type=int, default=0, help="Load layer")
    # load specific layers
    parser.add_argument("--load_layer_checkpoint_dirs", type=str, nargs="+", default=None, help="Load layer checkpoint directories")
    parser.add_argument("--load_layers", type=int, nargs="+", default=None, help="Load layers")

    parser.add_argument("--save_name", type=str, default="none")

    parser.add_argument("--data_dir", type=str, help="Path to data directory")
    parser.add_argument("--enable_wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="mtl_graph", help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default="", help="Wandb entity name")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples per task")
    parser.add_argument("--node", type=int, default=4, help="Number of nodes in the graph")
    parser.add_argument("--graph_batch_dir", type=str, default="graph_batches", help="Graph batch directory")
    args = parser.parse_args()

    # load config
    cfg = load_cfg(args.cfg)
    cfg.TRAIN.OPTIMIZER.NAME = args.optimizer
    cfg.TRAIN.OPTIMIZER.LR = args.lr
    cfg.TRAIN.LOSS.OUTPUT_LOSS_WEIGHT = args.loss_weight_output
    cfg.TRAIN.LOSS.HINT_LOSS_WEIGHT = args.loss_weight_hint
    cfg.TRAIN.LOSS.HIDDEN_LOSS_WEIGHT = args.loss_weight_hidden
    cfg.TRAIN.BATCH_SIZE = args.batch_size
    cfg.TRAIN.MAX_EPOCHS = args.epochs
    cfg.RUN_NAME = cfg.RUN_NAME+"-hints"
    cfg.LOGGING.WANDB.PROJECT = args.wandb_project
    cfg.LOGGING.WANDB.ENTITY = args.wandb_entity

    # update model configs 
    cfg.MODEL.HIDDEN_DIM = args.hidden_dim
    cfg.MODEL.MSG_PASSING_STEPS = args.gnn_layers
    cfg.MODEL.GRU.ENABLE = args.enable_gru
    cfg.MODEL.GRU.TASK_WISE = args.enbale_gru_task_wise
    if args.use_complete_graph:
        cfg.MODEL.PROCESSOR.KWARGS[0].update({"edge_dim": 128})
    
    logger.info("Starting run...")
    torch.set_float32_matmul_precision('medium')

    # load datasets
    algorithm_str = "_".join(args.algorithms)[:100]
    # Replaced this to use randomly generated graphs
    # data_module = MultiCLRSDataModule(algorithms=args.algorithms, batch_size=cfg.TRAIN.BATCH_SIZE)
    data_module = MultiCLRSDataModuleSampleComplexity(algorithms=args.algorithms, data_dir=args.data_dir, num_samples=args.num_samples, node=args.node, 
                                                      use_complete_graph=args.use_complete_graph, batch_size=cfg.TRAIN.BATCH_SIZE, graph_batch_dir="./shared_graph_cache")
    data_module.setup()
    task_to_specs = data_module.task_to_specs
    print("Using processor", cfg.MODEL.PROCESSOR.NAME)
    
    # load model
    metrics = {}; rng = np.random.default_rng(args.seed)
    for run in range(args.runs):
        # set seed
        run_seed = rng.integers(0, 10000)
        pl.seed_everything(run_seed)
        logger.info(f"Using seed {run_seed}")
        
        if args.train_mmoe:
            mtl_model = MMOE_EncodeProcessDecode(task_to_specs, cfg, args.num_experts)
        elif args.train_branched_network:
            mtl_model = BranchedMTL_EncodeProcessDecode(task_to_specs, cfg)
            if args.tree_config_dir is not None:
                tree_config_dir = os.path.join("tree_configs", args.tree_config_dir)
                with open(tree_config_dir, "r") as f:
                    for line in f.readlines():
                        layer, tasks = line.split(":")
                        tasks = tasks.strip().split(" ")
                        layer = int(layer)
                        mtl_model.branch_layers(layer, tasks)
            elif len(args.algorithms) == 2 and args.branch_layer != -1:
                mtl_model.branch_layers(args.branch_layer, args.algorithms[1])
            print(mtl_model)
            for i in range(len(mtl_model.task_to_processor_indexes)):
                print(mtl_model.task_to_processor_indexes[i])
        else:
            mtl_model = MultitaskEncodeProcessDecode(task_to_specs, cfg)
        model = MultiCLRSModel(task_to_specs, cfg=cfg, model=mtl_model)

        if args.load_checkpoint_dir is not None and os.path.exists(os.path.join("./checkpoints", args.load_checkpoint_dir)):
            state_dict = torch.load(os.path.join("./checkpoints", args.load_checkpoint_dir), map_location=model.device)

            mtl_model.load_state_dict(state_dict, strict=False)

            def filter_layer(state_dict, layer):
                new_state_dict = {}
                for key in state_dict:
                    if key.startswith(f"processor.{layer}."):
                        new_state_dict[key] = state_dict[key]
                return new_state_dict

            if args.load_layer_checkpoint_dirs is not None:
                for checkpoint_dir, layer in zip(args.load_layer_checkpoint_dirs, args.load_layers):
                    tmp_state_dict = torch.load(os.path.join("./checkpoints", checkpoint_dir), map_location=model.device)
                    tmp_state_dict = filter_layer(tmp_state_dict, layer)
                    mtl_model.load_state_dict(tmp_state_dict, strict=False)

            for name, param in mtl_model.named_parameters():
                if name.startswith("processor"):
                    layer = int(name.split(".")[1])
                    if layer >= args.train_layer:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False 

        for name, param in mtl_model.named_parameters():
            print(name, param.requires_grad)

        ckpt_dir = "./saved/"
        results = train(model, data_module, cfg, seed = run_seed, checkpoint_dir=ckpt_dir, devices=args.devices, algorithms=args.algorithms,
                        run_name=cfg.RUN_NAME + f"-run{run}" + (f"-{args.save_name}" if args.save_name != "none" else ""), node=args.node, num_samples=args.num_samples, use_wandb=args.enable_wandb)

        for key in results:
            if key not in metrics:
                metrics[key] = []
            metrics[key].append(results[key])
    # Log results
    for key in metrics:
        logger.info("{}: {:.4f} +/- {:.4f}".format(key, np.mean(metrics[key]), np.std(metrics[key])))
    
    logger.info("Saving results...")
    results_dir = f"results/{algorithm_str}/{cfg.RUN_NAME}_{args.save_name}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    # write csv
    metrics = {k: np.mean(v) for k, v in metrics.items()}
    with open(os.path.join(results_dir, f"results.csv"), "w") as f:
        writer = csv.DictWriter(f, metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)