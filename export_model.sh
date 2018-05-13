CHECKPOINT_FILE=/home/irina/ICNet-tensorflow/snapshots/model.ckpt-42000
TF_PATH=/home/irina/tensorflow
NAME=icnet
export CUDA_VISIBLE_DEVICES=""
CUR_PATH=/home/irina/ICNet-tensorflow

##########################################################################################

# Create unfrozen graph with export_inference_graph.py
python3 export_inference_graph.py --output_file ${NAME}_unfrozen.pb

python3 ${TF_PATH}/tensorflow/python/tools/freeze_graph.py \
--output_node_names="indices,label_names,label_colours,input_size,output_name" \
--input_graph=${NAME}_unfrozen.pb \
--input_checkpoint=${CHECKPOINT_FILE} \
--input_binary=true --output_graph=${NAME}_frozen.pb
#
#${TF_PATH}/bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
#  --in_graph=${CUR_PATH}/${NAME}_frozen.pb \
#  --out_graph=${CUR_PATH}/${NAME}_cleaned.pb \
#  --inputs="input"\
#  --outputs="indices"\
#  --transforms='add_default_attributes
#  strip_unused_nodes
#  remove_nodes(op=Identity, op=CheckNumerics)
#  fold_constants(ignore_errors=true)
#  fold_batch_norms
#  fold_old_batch_norms
#  quantize_weights
#  sort_by_execution_order'

python3 ${TF_PATH}/tensorflow/python/tools/optimize_for_inference.py \
--input=${CUR_PATH}/${NAME}_frozen.pb \
--output=${CUR_PATH}/${NAME}_optimized.pb \
--input_names="input" \
--output_names="indices"


