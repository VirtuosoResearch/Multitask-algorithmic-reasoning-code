SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${EXPERIMENT_DIR}/.." && pwd)"

# for num_sample in 3000 6000 9000
# do
# PYTHONPATH="${REPO_ROOT}/clrs${PYTHONPATH:+:${PYTHONPATH}}" python train_mtl.py --algorithms "mst_prim"\
#   --cfg "./configs/mst_prim/GINE.yml" --lr 5e-5 --hidden_dim 128 --gnn_layers 2 --enable_gru --enbale_gru_task_wise --devices 0 --batch_size 400 --epochs 100 --runs 1 --loss_weight_hint 0\
#   --save_name "mtl"\
#   --data_dir data/ --num_samples $num_sample --node 4
# done

# for num_sample in 20000 40000 80000
# do
# PYTHONPATH="${REPO_ROOT}/clrs${PYTHONPATH:+:${PYTHONPATH}}" python train_mtl.py --algorithms "mst_prim"\
#   --cfg "./configs/mst_prim/GINE.yml" --lr 5e-5 --hidden_dim 128 --gnn_layers 2 --enable_gru --enbale_gru_task_wise --devices 0 --batch_size 400 --epochs 100 --runs 1 --loss_weight_hint 0\
#   --save_name "mtl"\
#   --data_dir data/ --num_samples $num_sample --node 8
# done

for num_sample in 40000 80000 150000 300000
do
PYTHONPATH="${REPO_ROOT}/clrs${PYTHONPATH:+:${PYTHONPATH}}" python train_mtl.py --algorithms "mst_prim"\
  --cfg "./configs/mst_prim/GINE.yml" --lr 5e-5 --hidden_dim 128 --gnn_layers 2 --enable_gru --enbale_gru_task_wise --devices 0 --batch_size 400 --epochs 1 --runs 1 --loss_weight_hint 0\
  --save_name "mtl"\
  --data_dir data/ --num_samples $num_sample --node 12
done

for num_sample in 100000 200000 400000 800000
do
PYTHONPATH="${REPO_ROOT}/clrs${PYTHONPATH:+:${PYTHONPATH}}" python train_mtl.py --algorithms "mst_prim"\
  --cfg "./configs/mst_prim/GINE.yml" --lr 5e-5 --hidden_dim 128 --gnn_layers 2 --enable_gru --enbale_gru_task_wise --devices 0 --batch_size 400 --epochs 1 --runs 1 --loss_weight_hint 0\
  --save_name "mtl"\
  --data_dir data/ --num_samples $num_sample --node 16
done

for num_sample in 400000 800000 1600000
do
PYTHONPATH="${REPO_ROOT}/clrs${PYTHONPATH:+:${PYTHONPATH}}" python train_mtl.py --algorithms "mst_prim"\
  --cfg "./configs/mst_prim/GINE.yml" --lr 5e-5 --hidden_dim 128 --gnn_layers 2 --enable_gru --enbale_gru_task_wise --devices 0 --batch_size 400 --epochs 1 --runs 1 --loss_weight_hint 0\
  --save_name "mtl"\
  --data_dir data/ --num_samples $num_sample --node 20
done

for num_sample in 800000 1600000 3200000
do
PYTHONPATH="${REPO_ROOT}/clrs${PYTHONPATH:+:${PYTHONPATH}}" python train_mtl.py --algorithms "mst_prim"\
  --cfg "./configs/mst_prim/GINE.yml" --lr 5e-5 --hidden_dim 128 --gnn_layers 2 --enable_gru --enbale_gru_task_wise --devices 0 --batch_size 400 --epochs 1 --runs 1 --loss_weight_hint 0\
  --save_name "mtl"\
  --data_dir data/ --num_samples $num_sample --node 24
done


# python train_mtl.py --algorithms "mst_prim"\
#   --cfg "./configs/mst_prim/GINE.yml" --lr 5e-5 --hidden_dim 128 --gnn_layers 2 --enable_gru --enbale_gru_task_wise --devices 1 --batch_size 400 --epochs 100 --runs 1 --loss_weight_hint 0\
#   --save_name "mtl"\
#   --data_dir data/ --num_samples 100000 --node 20 --enable_wandb

# python train_mtl.py --algorithms "mst_prim"\
#   --cfg "./configs/mst_prim/GINE.yml" --lr 5e-5 --hidden_dim 128 --gnn_layers 2 --enable_gru --enbale_gru_task_wise --devices 1 --batch_size 400 --epochs 100 --runs 1 --loss_weight_hint 0\
#   --save_name "mtl"\
#   --data_dir data/ --num_samples 100000 --node 20 --enable_wandb
