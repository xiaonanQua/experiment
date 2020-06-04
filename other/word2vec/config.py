import os


class Config:
    def __init__(self):

        # -----------数据集配置----------------

        # 数据集路径
        self.dataset_path = {
            'train': 'ptb/ptb.train.txt',
            'val': 'ptb/ptb.val.txt',
            'test': 'ptb/ptb.test.txt'
        }

        # 最大背景窗口大小
        self.max_window_size = 5

        # ---------网络参数配置-----------

        # 运行模式
        self.run_mode = 'train'

print('Hello world')
