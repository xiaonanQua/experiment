# coding:utf8

import visdom
import time
import numpy as np


class Visualizer(object):
    """
    封装了visdom的基本操作
    """
    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        """
        修改visdom的配置
        :param env:
        :param kwargs:
        :return:
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        """
        一次plot多个
        :param d: dict(name,value) 也就是（‘loss', 0.11）
        :return:
        """
        for k,v in d.items():
            self.plot(k,v)

    def image_many(self,d):
        for k, v in d.items():
            self.image(k, v)

    def plot(self, name, y, **kwargs):
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([y]),
                      win=(name),
                      opts=dict(title=name),
                      update=None if x ==0 else 'append',
                      **kwargs)
        self.index[name] = x + 1

    def image(self, name, img_, **kwargs):
        self.vis.images(img_.cpu().numpy(),
                        win=(name),
                        opts=dict(title=name),
                        **kwargs)

    def log(self, info, win='log_text'):
        self.log_text += ('[{time}]{info}<br>'.format(time=time.strftime('%m%d_%H%M%S'), info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        return getattr(self.vis, name)

