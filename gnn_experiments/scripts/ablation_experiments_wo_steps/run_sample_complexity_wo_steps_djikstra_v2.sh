SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/.." && pwd)"

# for num_sample in 4000 6000 8000 
# do
# PYTHONPATH="${REPO_ROOT}/clrs${PYTHONPATH:+:${PYTHONPATH}}" python train_mtl.py --algorithms "dijkstra"\
#   --cfg "./configs/dijkstra/GINE.yml" --lr 5e-5 --hidden_dim 128 --gnn_layers 2 --enable_gru --enbale_gru_task_wise --devices 1 --batch_size 400 --epochs 100 --runs 1 --loss_weight_hint 0\
#   --save_name "mtl"\
#   --data_dir data/ --num_samples $num_sample --node 4
# done

# for num_sample in 15000 20000 30000
# do
# PYTHONPATH="${REPO_ROOT}/clrs${PYTHONPATH:+:${PYTHONPATH}}" python train_mtl.py --algorithms "dijkstra"\
#   --cfg "./configs/dijkstra/GINE.yml" --lr 5e-5 --hidden_dim 128 --gnn_layers 2 --enable_gru --enbale_gru_task_wise --devices 1 --batch_size 400 --epochs 100 --runs 1 --loss_weight_hint 0\
#   --save_name "mtl"\
#   --data_dir data/ --num_samples $num_sample --node 8
# done

# for num_sample in 50000 70000 90000
# do
# PYTHONPATH="${REPO_ROOT}/clrs${PYTHONPATH:+:${PYTHONPATH}}" python train_mtl.py --algorithms "dijkstra"\
#   --cfg "./configs/dijkstra/GINE.yml" --lr 5e-5 --hidden_dim 128 --gnn_layers 2 --enable_gru --enbale_gru_task_wise --devices 1 --batch_size 400 --epochs 100 --runs 1 --loss_weight_hint 0\
#   --save_name "mtl"\
#   --data_dir data/ --num_samples $num_sample --node 12
# done

PYTHONPATH="${REPO_ROOT}/clrs${PYTHONPATH:+:${PYTHONPATH}}" python train_mtl.py --algorithms "dijkstra"\
  --cfg "./configs/dijkstra/GINE.yml" --lr 5e-5 --hidden_dim 128 --gnn_layers 2 --enable_gru --enbale_gru_task_wise --devices 1 --batch_size 400 --epochs 100 --runs 1 --loss_weight_hint 2\
  --save_name "mtl"\
  --data_dir data/ --num_samples 1000 --node 4

PYTHONPATH="${REPO_ROOT}/clrs${PYTHONPATH:+:${PYTHONPATH}}" python train_mtl.py --algorithms "dijkstra"\
  --cfg "./configs/dijkstra/GINE.yml" --lr 5e-5 --hidden_dim 128 --gnn_layers 2 --enable_gru --enbale_gru_task_wise --devices 1 --batch_size 400 --epochs 100 --runs 1 --loss_weight_hint 2\
  --save_name "mtl"\
  --data_dir data/ --num_samples 5000 --node 8

PYTHONPATH="${REPO_ROOT}/clrs${PYTHONPATH:+:${PYTHONPATH}}" python train_mtl.py --algorithms "dijkstra"\
  --cfg "./configs/dijkstra/GINE.yml" --lr 5e-5 --hidden_dim 128 --gnn_layers 2 --enable_gru --enbale_gru_task_wise --devices 1 --batch_size 400 --epochs 100 --runs 1 --loss_weight_hint 2\
  --save_name "mtl"\
  --data_dir data/ --num_samples 25000 --node 12

PYTHONPATH="${REPO_ROOT}/clrs${PYTHONPATH:+:${PYTHONPATH}}" python train_mtl.py --algorithms "dijkstra"\
  --cfg "./configs/dijkstra/GINE.yml" --lr 5e-5 --hidden_dim 128 --gnn_layers 2 --enable_gru --enbale_gru_task_wise --devices 1 --batch_size 400 --epochs 1 --runs 1 --loss_weight_hint 2\
  --save_name "mtl"\
  --data_dir data/ --num_samples 50000 --node 16

PYTHONPATH="${REPO_ROOT}/clrs${PYTHONPATH:+:${PYTHONPATH}}" python train_mtl.py --algorithms "dijkstra"\
  --cfg "./configs/dijkstra/GINE.yml" --lr 5e-5 --hidden_dim 128 --gnn_layers 2 --enable_gru --enbale_gru_task_wise --devices 1 --batch_size 400 --epochs 1 --runs 1 --loss_weight_hint 2\
  --save_name "mtl"\
  --data_dir data/ --num_samples 100000 --node 20

PYTHONPATH="${REPO_ROOT}/clrs${PYTHONPATH:+:${PYTHONPATH}}" python train_mtl.py --algorithms "dijkstra"\
  --cfg "./configs/dijkstra/GINE.yml" --lr 5e-5 --hidden_dim 128 --gnn_layers 2 --enable_gru --enbale_gru_task_wise --devices 1 --batch_size 400 --epochs 1 --runs 1 --loss_weight_hint 2\
  --save_name "mtl"\
  --data_dir data/ --num_samples 200000 --node 24