import torch
import torch.optim as optim

class WarmupOptimizer(object):
    def __init__(self, lr_base, optimizer, data_size, batch_size, is_warmup=True):
        self.optimizer = optimizer
        self._step = 0
        self.lr_base = lr_base
        self._rate = 0
        self.data_size = data_size
        self.batch_size = batch_size
        self.is_warmup = is_warmup

    def step(self):
        # 统计训练的步骤
        self._step += 1

        # 获得下一阶段的学习率
        rate = self.rate()
        # 更新参数中学习率
        for p in self.optimizer.param_groups:
            p['lr'] = rate

        self._rate = rate
        # 优化器更新梯度
        self.optimizer.step()

    def zero_grad(self):
        # 每次循环开始训练,初始化梯度张量
        self.optimizer.zero_grad()

    def rate(self, step=None):
        # 如果训练步骤为None,则使用之前统计的步骤数
        if step is None:
            step = self._step

        # 在前3周期内分别按[1/4, 2/4, 3/4]每步骤对学习率进行衰减
        if step <= int(self.data_size/self.batch_size*1) and self.is_warmup:
            r = self.lr_base*1/4
        elif step <= int(self.data_size/self.batch_size*2) and self.is_warmup:
            r = self.lr_base*2/4
        elif step <= int(self.data_size/self.batch_size*3) and self.is_warmup:
            r = self.lr_base*3/4
        else:
            # 恢复学习率到初始学习率
            r = self.lr_base
        return r

# 获得经过处理后的优化器
def get_optim(cfg, model, data_size, lr_base=None):
    # 定义Adam优化器
    optimizer = optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=0,
                           betas=cfg.OPT_BETAS, eps=cfg.OPT_EPS)
    warm_optimizer = WarmupOptimizer(lr_base, optimizer, cfg.data_size, cfg.batch_size)
    return warm_optimizer

# 调整学习率
def adjust_lr(optimizer, decay_r):
    optimizer.lr_base *= decay_r