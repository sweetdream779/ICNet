OUT=/home/irina/Desktop/work/lanes_results
python3 inference_sliding_window.py \
--img-path=/home/irina/Desktop/work/spc_results/panorams \
--save-dir=${OUT} \
--pb-file=/home/irina/Desktop/work/lanes_segmentator_v1.2_cleaned.pb
