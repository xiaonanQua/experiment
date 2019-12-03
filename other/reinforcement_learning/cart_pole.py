"""
车杆游戏
游戏结束规则：
1、杆的倾斜角度超过15度
2、小车移动超过2.4个单元
3、游戏已经玩过200步
"""
import gym
import time
import random
import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import deque


def play_single_episode():
    """
    玩一场游戏
    :return:
    """
    # 获得游戏环境
    env = gym.make('CartPole-v0')
    # 重置游戏环境，新一局游戏开始
    observation = env.reset()
    print('新一句游戏开始，初始观测：{}'.format(observation))
    for t in range(200):
        # 绘制当前的游戏状态，并显示出来
        env.render()
        # 随机选择当前动作
        action = env.action_space.sample()
        print('{}:动作={}'.format(t, action))
        # 执行动作
        obeservation, reward, done, info = env.step(action)
        # 观测值的信息分布：[小车位置，小车速度，木棒角度，木棒角速度]
        print('{}:观测值：{}，本步得分：{}，结束指示：{}，其他信息：{}'.format(t, observation, reward, done, info))
        # done为True则结束
        if done:
            break
        print(env.observation_space.shape[0])
        # 睡眠一秒
        time.sleep(0.5)
    env.close()


def play_mutli_episode():
    """
    玩多回合游戏
    :return:
    """
    # 绘制当前游戏环境
    env = gym.make('CartPole-v0')
    # 进行游戏的回合数
    num_episode = 30
    # 进行多个回合的迭代
    for i_episode in range(num_episode):
        # 每回合开始重置游戏环境,初始化观测值
        obeservation = env.reset()
        # 回合奖励
        episode_reward = 0
        while True:
            # 绘制当前游戏状态并可视化
            env.render()
            # 随机选择行动
            action = env.action_space.sample()
            # 执行行动
            observation, reward, done, _ = env.step(action)
            # 统计回合奖励
            episode_reward += 1
            # 若done为True则结束
            if done:
                break
            time.sleep(0.5)
        print('第{}局得分：{}'.format(i_episode, episode_reward))
    env.close()


def act(net, state, epsilon, env):
    """
    决策方法使用epsilon贪心探索法
    :param net: 网络
    :param state: 状态
    :param epsilon: 系数，决策最大动作值
    :param env: 游戏环境
    :return:
    """
    # 若随机的值大于epsilon，则选出使得q(s,a)最大的动作a。若随机数小于epsilon，则随机算则一种动作
    if random.random()>epsilon:
        # 将状态转化成float张量，消去0维为1的维度
        state = torch.FloatTensor(state).unsqueeze(0)
        # 获取网络预测的动作
        q_value = net.forward(state)
        # 获取使q_value最大的动作
        action = q_value.max(1)[1].item()
    else:
        # 随机生成小于env.action_space.n的整数值
        action = random.randrange(env.action_space.n)
    return action


def change_epsilon(t, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=500):
    """
    随着训练步长的增长，改变epislin的值
    :param t: 训练步长
    :param epsilon_start: 开始值
    :param epsilon_end: 结束值
    :param epsilon_decay: 衰减值
    :return:
    """
    epsilon = epsilon_end + (epsilon_start - epsilon_end)*math.exp(-1.*t/epsilon_decay)
    return epsilon


class ReplayBuffer(object):
    """
    使用双向队列存储最近的历史，并在最近的历史中采样。
    在该类中存在大小有限的双向队列buffer，它可以不断存储新的历史，并且在满时自动删去最久远的历史
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)  # 设置特定大小的双向队列

    def push(self, state, action, reward, next_state, done):
        """
        将状态、行为、奖励信息送入队列中
        :param state: 状态
        :param action: 行为
        :param reward: 奖励
        :param next_state: 下一个奖励
        :param done: 结束信息
        :return:
        """
        # 给状态添加一个一维维度
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        # 将信息添加到队列的右端
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        从队列里采样相应批次的数据
        :param batch_size: 批次大小
        :return:
        """
        # 从双向队列中采样批次大小的数据
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        # 将状态数据连接起来
        concat_state = np.concatenate(state)
        concat_next_state = np.concatenate(next_state)
        return concat_state, action, reward, concat_next_state, done

    def __len__(self):
        return len(self.buffer)


def train_ai():
    """
    训练AI
    :return:
    """
    # 定义网络训练的超参数
    gamma = 0.99
    batch_size = 64

    # 各回合的得分情况
    episode_rewards = []
    # 训练步骤，用于计算epsilon
    t = 0

    # 构建游戏环境
    env = gym.make('CartPole-v0')

    # 定义Q网络来预测行为动作
    q_net = nn.Sequential(
        # 网络的输入元素个数是观测的元素个数,输入元素是观测值
        nn.Linear(env.observation_space.shape[0], 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        # 网络输出层，根据观测值预测行为
        nn.Linear(128, env.action_space.n)
    )

    # 定义Adam优化器
    optimizer = optim.Adam(q_net.parameters())

    # 实例化缓存对象
    replay_buffer = ReplayBuffer(capacity=1000)

    # 无限循环多轮训练
    while True:
        # 每次循环迭代都开始新的一局
        state = env.reset()
        # 统计这一轮的得分情况
        episode_reward = 0

        # 循环一轮的结果
        while True:
            # 计算epsilon的衰减值
            epsilon = change_epsilon(t)
            # 通过决策函数获得行动
            action = act(q_net, state, epsilon, env)
            # 通过行动获得相关信息
            next_state, reward, done, _ = env.step(action)
            # 将获得新数据送入缓存中
            replay_buffer.push(state, action, reward, next_state, done)

            # 更新状态和奖励值
            state = next_state
            episode_reward += reward

            if len(replay_buffer) > batch_size:
                # 计算时间差分误差
                sample_state, sample_action, sample_reward, sample_next_state, sample_done = \
                replay_buffer.sample(batch_size)

                # 将获得的数据转化成张量
                sample_state = torch.tensor(sample_state, dtype=torch.float32)
                sample_action = torch.tensor(sample_action, dtype=torch.int64)
                sample_reward = torch.tensor(sample_reward, dtype=torch.float32)
                sample_next_state = torch.tensor(sample_next_state, dtype=torch.float32)
                sample_done = torch.tensor(sample_done, dtype=torch.float32)

                # 下一个动作值
                next_qs = q_net(sample_next_state)
                next_q, _ = next_qs.max(dim=1)

                expected_q = sample_reward + gamma * next_q * (1-sample_done)

                # 当前动作值
                qs = q_net(sample_state)
                q = qs.gather(1, sample_action.unsqueeze(1)).squeeze(1)

                td_error = expected_q - q

                # 计算MSE损失
                loss = td_error.pow(2).mean()

                # 根据损失改进网络
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t += 1

            if done:  # 本局结束
                # 统计回合数
                i_episode = len(episode_rewards)
                print('第{}局收益={}'.format(i_episode, episode_reward))
                episode_rewards.append(episode_reward)
                break

        if len(episode_rewards) > 20 and np.mean(episode_rewards[-20:])>195:
            torch.save(q_net.state_dict(), 'cart.pth')
            print('保存模型')
            break  # 结束训练


def play_ai():
    """
    让ai玩游戏
    :return:
    """
    # 玩的轮数
    num_episode = 20
    # 构建游戏环境
    env = gym.make('CartPole-v0')

    # 定义Q网络来预测行为动作
    q_net = nn.Sequential(
        # 网络的输入元素个数是观测的元素个数,输入元素是观测值
        nn.Linear(env.observation_space.shape[0], 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        # 网络输出层，根据观测值预测行为
        nn.Linear(128, env.action_space.n)
    )

    # 加载模型
    q_net.load_state_dict(torch.load('cart.pth'))

    for i_episode in range(num_episode):
        obeservation = env.reset()
        episode_reward = 0

        while True:
            # 绘制当前的游戏状态，并显示出来
            env.render()
            # 随机选择当前动作
            action = act(q_net, obeservation, 0, env)
            # 执行动作
            obeservation, reward, done, info = env.step(action)
            episode_reward += reward
            # done为True则结束
            if done:
                break
            # 睡眠一秒
            # time.sleep(0.5)

        print('第{}局得分={}'.format(i_episode, episode_reward))
    env.close()


if __name__ == '__main__':
    # play_single_episode()
    # play_mutli_episode()
    # print(random.randrange(2))
    # state = torch.randn(30, 4)
    # print(state, state.max(dim=1))
    # train_ai()
    play_ai()