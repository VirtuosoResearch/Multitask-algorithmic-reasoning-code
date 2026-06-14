# conda activate llama-env
# cd /home/ldy/Multi-CLRS-reasoning-code/gnn_experiments

PYTHONPATH=../clrs python star_graph_experiment.py \
  --cfg ./configs/GIN.yml \
  --min_nodes 8 --max_nodes 16 \
  --num_train 2000 --num_test 256 \
  --gnn_layers 4 --epochs 100 --devices 0
