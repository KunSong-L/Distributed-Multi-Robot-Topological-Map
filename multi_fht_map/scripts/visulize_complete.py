import pandas as pd
import matplotlib.pyplot as plt

path = "/home/master/debug/map_complete_data/"
robot_name = "robot1"

# 创建初始图形
fig, ax = plt.subplots()
line, = ax.plot([], [], 'b-')
ax.set_xlabel('Timestamp')
ax.set_ylabel('Zeros Count')
ax.set_xlim(0, 2000)  # 设置x轴的范围
ax.set_ylim(0, 140000)  # 设置y轴的范围

def update_graph():
    global df  # 声明df为全局变量
    # 读取CSV文件并更新DataFrame对象
    new_data = pd.read_csv(path + robot_name + 'map_complete.csv')
    df = pd.concat([new_data], ignore_index=True)
    
    # 更新图形
    line.set_data(df['Timestamp'], df['Zeros Count'])
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()

# 创建空的DataFrame对象
df = pd.DataFrame(columns=['Timestamp', 'Zeros Count'])

# 每隔1秒更新一次图形
while True:
    update_graph()
    plt.pause(3)