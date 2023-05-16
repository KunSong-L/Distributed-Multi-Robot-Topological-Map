#!/usr/bin/python3.8

import subprocess

# 调用可执行文件并等待其完成
executable_file_path = "./pose_graph_2d"
input_file_path = "input_MITb_g2o.g2o"
process = subprocess.Popen([executable_file_path,"--input",input_file_path], stdout=subprocess.PIPE)
output, error = process.communicate()

# 输出结果
print(output.decode())

