"""
记录一些操作关于如何设置命令行参数
"""
import argparse  # 导入解析命令行的模块

# 获得命令行参数解析器
parser = argparse.ArgumentParser()

# 增加参数，使程序能够接受的参数;设置‘--’可以让参数变为可选的
parser.add_argument('--x', type=int)
parser.add_argument('--y', type=str)
# action意味着,如果指定了该选项，则将True赋给args.b。不指定它意味着False。
parser.add_argument('--b', action='store_true')
# 使用’-‘让参数可选
parser.add_argument('-s', type=int, metavar='s')

# 获得解析后的参数
arguments = parser.parse_args()

# 输出参数值
print(arguments.x**2)
print(arguments.y)
if arguments.b:
    print('b turn on!')
print(arguments.s)
