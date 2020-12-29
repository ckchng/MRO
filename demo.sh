#!/bin/bash 
cd '/path/to/MRO/build/src'

orb_filename="/path/to/MRO/data/ORBvoc.txt"

MIN_SET=10
winsize=4
img_width=1241
img_height=376

config_filename="/path/to/MRO/data/config.yaml"

sequence_path="/path/to/KITTI/data_odometry_gray/dataset/sequences/00/image_0"

output_path="/path/to/MRO/outputs"

sel_id_output_path="${output_path}/view_graph_sel_id.h5"

conn_output_path="${output_path}/view_graph_connections.h5"

tij_conn_output_path="${output_path}/view_graph_tij_connections.h5"

view_conn_output_path="${output_path}/view_graph_views.h5"

poses_conn_output_path="${output_path}/view_graph_poses.h5"

./irotavg $orb_filename $config_filename $sequence_path $sel_id_output_path $conn_output_path $tij_conn_output_path $view_conn_output_path $poses_conn_output_path $MIN_SET $winsize $img_width $img_height

