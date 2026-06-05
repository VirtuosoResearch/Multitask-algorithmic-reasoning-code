# Project code on multitask algorithmic reasoning

This repository contains code for multitask algorithmic reasoning experiments on the CLRS benchmark and text-based graph tasks. We propose branching neural networks for multitask algorithmic reasoning by dividing algorithms into separate branches. This can be applied on top of base models, including GNNs or LLMs with low-rank adapters. 

## Repository Structure

The repository is organized into several main components:

- **clrs_experiments**: GNN-based experiments on the CLRS benchmark
- **text-graph-tasks**: LLM-based experiments on text-based reasoning tasks
- **gnn_experiments**: Additional GNN experiments


## CLRS experiments

This directory contains code for running GNN-based experiments on the CLRS-30 benchmark, focusing on multitask algorithmic reasoning with different graph neural network architectures.

We use 12 graph algorithms from the CLRS benchmark across different categories: `bfs`, `dfs`, `topological_sort`, `articulation_points`, `bridges`, `strongly_connected_components`, `mst_kruskal`, `mst_prim`, `dijkstra`, `bellman_ford`, `dag_shortest_paths`

### Installation

1. Create a conda environment:
```bash
conda create -n clrs python=3.10
conda activate clrs
```

2. Install dependencies:
```bash
cd clrs_experiments
pip install -e .
```


#### Usage

Use `clrs.examples.run` to train and evaluate models on the CLRS benchmark.

```bash
# Train a single algorithm (e.g., Dijkstra):
python -m clrs.examples.run --algorithms dijkstra

# Train multiple algorithms:
python -m clrs.examples.run --algorithms "bfs" "dfs" "dijkstra"

# Specify processor type and model parameters
python -m clrs.examples.run \
  --algorithms "bfs" "dfs" \
  --processor_type "edge_t" \
  --num_layers 5 \
  --hidden_size 192 \
  --use_projection \
  --projection_dim 16
```

Use the following example to train a branching network
```bash
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -m clrs.examples.run \
    --algorithms "bfs","dfs","topological_sort","articulation_points","bridges","strongly_connected_components","mst_kruskal","mst_prim","dijkstra","bellman_ford",'dag_shortest_paths',"floyd_warshall"\
    --use_branching_structure --branching_structure_dir "tree_structure" --processor_type branching_edge_t \
    --num_layers 5 \
    --runs 3 \
    --train_steps 10000 
```

Use `branchnn_search.py` to conduct the search for the branching structures. This file integrates the training,  gradient-based approximation, and clustering steps into a single script. For example: 
```
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python branchnn_search.py \
        --algorithms "bfs","dfs","topological_sort","articulation_points","bridges","strongly_connected_components","mst_kruskal","mst_prim","dijkstra","bellman_ford",'dag_shortest_paths',"floyd_warshall" \
        --processor_type "edge_t" --num_layers 5 --hidden_size 192 \
        --gradient_projection_dim 400 --num_subsets 200 --subset_size 3
```

Available processor types:
- `gat`: Graph Attention Network
- `edge_t`: Edge Transformer
- `mpnn`: Message Passing Neural Network
- `pgn`: Pointer Graph Network
- `branching_edge_t`: Branching Edge Transformers
- `branching_mpnn`: Branching MPNN networks
- `branching_gat` & `branching_gatv2`: Branching GAT networks

Key hyperparameters (see `clrs/examples/run.py` and `clrs/branchnn_search.py` for full list):

- `--batch_size`: Training batch size (default: 4)
- `--train_steps`: Number of training iterations (default: 10000)
- `--learning_rate`: Learning rate (default: 2.5e-4)
- `--hidden_size`: Hidden dimension size (default: 192)
- `--num_layers`: Number of network layers (default: 5)
- `--processor_type`: Type of GNN processor`


## Text-based reasoning tasks

This directory contains code for training LLMs on text-encoded graph reasoning tasks, including CLRS text tasks, GraphWiz, and GraphQA benchmarks.

### Installation

Create a conda environment:
```bash
conda env create -f text-graph-tasks/environment.yml
conda activate llama-env
```

Or manually:
```bash
conda create -n llama-env python=3.10
conda activate llama-env
cd text-graph-tasks
pip install -r requirements.txt
```

### Supported Tasks

Tasks in text versions of the CLRS benchmark:
- Graph algorithms: `bfs`, `dfs`, `topological_sort`, `articulation_points`, `bridges`, `strongly_connected_components`
- Shortest path: `dijkstra`, `bellman_ford`, `dag_shortest_paths`, `floyd_warshall`
- Minimum spanning tree: `mst_kruskal`, `mst_prim`

GraphWiz Tasks
- `connectivity`, `bipartite`, `cycle`, `flow`, `hamilton`, `shortest`, `substructure`, `topology`, `triangle`

GraphQA Tasks 
- `edge_existence`, `node_degree`, `node_count`, `edge_count`, `connected_nodes`, `cycle_check`, `disconnected_nodes`, `reachability`, `shortest_path`, `maximum_flow`, `triangle_counting`, `node_classification`
- Follow instructions in `graph_tasks` to generate task data

### Training

For Text-CLRS datasets, use `train_clrs_text.py`

For GraphWiz datasets, use `train_graphwiz.py`

For GraphQA datasets, use `train_graphqa.py`

Use `fast_estimate_compute_gradients.py` to evaluate gradients on a trained adapter model. 

Use `fast_estimate_linear_regression.py` to perform gradient-based estimation on a trained adapter model over subsets of tasks.

Key parameters for training scripts:
- `--task_names`: List of tasks to train on
- `--model_key`: HuggingFace model identifier
- `--devices`: GPU device IDs to use
- `--batch_size`: Training batch size
- `--max_epochs`: Maximum training epochs
- `--learning_rate`: Learning rate (default: 1e-5)
- `--max_length`: Maximum sequence length
- `--train_multitask`: Enable multi-task training
- `--use_lora`: Use LoRA fine-tuning
- `--use_qlora`: Use QLoRA (4-bit quantization)
- `--train_adapter`: Use adapter-based training
- `--load_branching_config`: Load branching structure from file
- `--task_branching_config_dir`: Directory for branching structure files

## Requirements

Both projects require:
- Python 3.10+
- CUDA-compatible GPU (recommended)
- Sufficient GPU memory (8GB+ for smaller models, 24GB+ for 7B+ LLMs)

## Citation

If you find this repository useful or happen to use it in a research paper, please cite our work with the following BibTeX information.

```
@article{li2026efficiently,
  title={Efficiently Learning Branching Networks for Multitask Algorithmic Reasoning},
  author={Li, Dongyue and Zhang, Zhenshuo and Duan, Minxuan and Dobriban, Edgar and Zhang, Hongyang R},
  journal={SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2026}
}
```
