import os
import math
import numpy as np

def writeTSPLIBfile_FE(fname_tsp,CostMatrix,user_comment,pwd,tsplib_dir,problemType):
    dims_tsp = len(CostMatrix)
    name_line = 'NAME : ' + fname_tsp + '\n'
    type_line = 'TYPE: TSP' + '\n'
    comment_line = 'COMMENT : ' + user_comment + '\n'
    tsp_line = 'TYPE : ' + problemType + '\n'
    dimension_line = 'DIMENSION : ' + str(dims_tsp) + '\n'
    edge_weight_type_line = 'EDGE_WEIGHT_TYPE : ' + 'EXPLICIT' + '\n' # explicit only
    edge_weight_format_line = 'EDGE_WEIGHT_FORMAT: ' + 'FULL_MATRIX' + '\n'
    display_data_type_line ='DISPLAY_DATA_TYPE: ' + 'NO_DISPLAY' + '\n' # 'NO_DISPLAY'
    edge_weight_section_line = 'EDGE_WEIGHT_SECTION' + '\n'
    eof_line = 'EOF\n'
    Cost_Matrix_STRline = []
    for i in range(0,dims_tsp):
        cost_matrix_strline = ''
        for j in range(0,dims_tsp-1):
            cost_matrix_strline = cost_matrix_strline + str(int(CostMatrix[i][j])) + ' '

        j = dims_tsp-1
        cost_matrix_strline = cost_matrix_strline + str(int(CostMatrix[i][j]))
        cost_matrix_strline = cost_matrix_strline + '\n'
        Cost_Matrix_STRline.append(cost_matrix_strline)
    
    fileID = open((pwd + tsplib_dir + fname_tsp + '.tsp'), "w")
    fileID.write(name_line)
    fileID.write(comment_line)
    fileID.write(tsp_line)
    fileID.write(dimension_line)
    fileID.write(edge_weight_type_line)
    fileID.write(edge_weight_format_line)
    fileID.write(edge_weight_section_line)
    for i in range(0,len(Cost_Matrix_STRline)):
        fileID.write(Cost_Matrix_STRline[i])

    fileID.write(eof_line)
    fileID.close()

    fileID2 = open((pwd + tsplib_dir + fname_tsp + '.par'), "w")

    problem_file_line = 'PROBLEM_FILE = ' + pwd + tsplib_dir + fname_tsp + '.tsp' + '\n' # remove pwd + tsplib_dir
    optimum_line = 'OPTIMUM 378032' + '\n'
    move_type_line = 'MOVE_TYPE = 5' + '\n'
    patching_c_line = 'PATCHING_C = 3' + '\n'
    patching_a_line = 'PATCHING_A = 2' + '\n'
    runs_line = 'RUNS = 100' + '\n'
    tour_file_line = 'TOUR_FILE = ' + pwd + tsplib_dir + fname_tsp + '.txt' + '\n'

    fileID2.write(problem_file_line)
    fileID2.write(optimum_line)
    fileID2.write(move_type_line)
    fileID2.write(patching_c_line)
    fileID2.write(patching_a_line)
    fileID2.write(runs_line)
    fileID2.write(tour_file_line)
    fileID2.close()
    return fileID, fileID2



def solveTSP(CostMatrix,fname_tsp='TSP_file',problemType='ATSP'):
    #CostMatrix[i,j]: the cost from node i to node j
    #预处理CostMatrix
    if len(CostMatrix)==2:
        return [0,1,-1], CostMatrix[0,1] + CostMatrix[1,0]

    k_num = 1000 #由于输入为整数，首先换到毫米的单位
    CostMatrix = CostMatrix*k_num

    user_comment = "a comment by the user"
    # Change these directories based on where you have 
    # a compiled executable of the LKH TSP Solver
    lkh_dir = '/'
    tsplib_dir = '/'
    lkh_cmd = 'LKH'
    pwd = os.path.dirname(os.path.abspath(__file__))
    fileID1,fileID2 = writeTSPLIBfile_FE(fname_tsp,CostMatrix,user_comment,pwd,tsplib_dir,problemType)
    run_lkh_cmd =  pwd + lkh_dir  + lkh_cmd + ' ' + pwd + tsplib_dir + fname_tsp + '.par'
    os.system(f"{run_lkh_cmd} > /dev/null 2>&1")
    #准备读取文件
    # 打开文件
    file = open(os.path.join(pwd , fname_tsp + '.txt'), 'r')  # 'example.txt' 是文件的路径和名称，'r' 表示以只读模式打开文件
    # 读取文件内容
    content = file.read()
    # 关闭文件
    file.close()
    # 打印文件内容
    index = content.find('TOUR_SECTION')
    index_EOF = content.find('EOF')

    # 从 TOUR_SECTION 后面截取字符串
    tour_section = content[index + len('TOUR_SECTION'):index_EOF].strip()

    # 将字符串按行分割，并忽略空行
    lines = tour_section.split('\n')
    lines = [line.strip() for line in lines if line.strip()]

    # 提取数字
    numbers = [int(line) for line in lines]
    
    #提取距离
    length_index = content.find('COMMENT : Length =')
    length = int(content[length_index + len('COMMENT : Length ='):].strip().split('\n')[0])
    

    rm_sol_cmd = 'rm' + ' ' + pwd + tsplib_dir+ fname_tsp + '.txt'
    os.system(rm_sol_cmd) 
    rm_sol_cmd = 'rm' + ' ' + pwd + tsplib_dir+ fname_tsp + '.par'
    os.system(rm_sol_cmd) 
    rm_sol_cmd = 'rm' + ' ' + pwd + tsplib_dir+ fname_tsp + '.tsp'
    os.system(rm_sol_cmd) 

    return numbers,length/k_num

def multi_robot_tsp(points,robots_pose):
    #points: n*3 array
    #robot_pose: m*3 array
    
    n_points = len(points)
    n_robots = len(robots_pose)
    robot_frontier = [[] for i in range(n_robots)]
    

    for point in points:
        distance_list = []
        for i in range(n_robots):
            #对每个机器人做一个TSP
            total_point = [robots_pose[i]] + robot_frontier[i] + [point]
            now_cost_mat = distanc_function(total_point)
            numbers,length = solveTSP(now_cost_mat,'TSP_file','ATSP')
            distance_list.append(length)
        
        min_index = distance_list.index(min(distance_list))
        
        robot_frontier[min_index].append(point)
    
    robot_frontier_sorted = [[] for i in range(n_robots)]
    for i in range(n_robots):
        #对每个机器人做一个TSP
        total_point = [robots_pose[i]] + robot_frontier[i]
        now_cost_mat = distanc_function(total_point)
        numbers,length = solveTSP(now_cost_mat,'TSP_file','ATSP')
        robot_frontier_sorted[i] = [robot_frontier[i][index-2] for index in numbers[1:-1]]
      

    return robot_frontier_sorted


def distanc_function(points):
    n_point = len(points)
    CostMatrix = np.zeros((n_point,n_point))
    for i in range(n_point):
        for j in range(n_point):
            point1= points[i]
            point2 = points[j]
            CostMatrix[i,j] = np.linalg.norm(point1 - point2)
    
    return CostMatrix


from scipy.spatial.distance import cdist
if __name__ == '__main__':

    theta = np.arange(0,2*np.pi,2*np.pi/6)
    points = np.vstack((np.cos(theta),np.sin(theta))).T
    CostMatrix = distanc_function(points)
    numbers,length = solveTSP(CostMatrix,'TSP_file','ATSP')

    print('visit order: ', numbers)
    print('total length of path:', length)

    # multi robot
    robots_pose = np.zeros((2,2))
    robot_frontier = multi_robot_tsp(points,robots_pose)
    print(robot_frontier)