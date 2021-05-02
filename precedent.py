import numpy
import gym
env = gym.make('CartPole-v0')

import numpy as np

dim_action = env.action_space.n
dim_obs = env.observation_space.shape[0]
# 将整数转变成one-hot向量形式
def one_hot(value, dim):
    arr = np.zeros((dim,), dtype=np.float)
    arr[value] = 1
    return arr

# 使用线性方法，估计动作价值函数， 顺便给出梯度。
def Q(obs, action, weight):
    obs = np.expand_dims(obs, 0)
    action = np.expand_dims(one_hot(int(action), dim_action), 1)
    print(action)
    prediction = np.dot(np.dot(obs, weight), action)
    grad = np.dot(obs.T, action.T)
    #print(grad)
    return prediction, grad

# 一个使用epsilon-贪心的软性策略
def pi_soft(obs, weight, epsilon):
    Qs = np.dot(obs, weight)  # Qs = [Q(s, a1), Q(s, a2), ...]
    p = np.zeros((2), dtype=float)
    p[np.argmax(Qs)] = 1 - epsilon
    p += epsilon / dim_action
    return np.random.choice([0, 1], p=p)

# 初始化学习的权重
weight = np.zeros((dim_obs, dim_action), dtype=float)
# 学习的步长
alpha = 0.1

import matplotlib.pyplot as plt

epsilon = 0.01
performance = []

for episode in range(300):  # 循环：每一幕
    
    obs = env.reset()  # 初始状态
    action = pi_soft(obs, weight, epsilon)  # 先选择第一个动作
    print(episode)
    for i in range(1000):  # 循环：每一步
        env.render()
        # 执行动作，观察
        next_obs, reward, done, info = env.step(action)
        
        #往后看：选择下一步动作
        next_action = pi_soft(next_obs, weight, epsilon)
        
        #更新权重。分幕式任务中可认为折后率=1
        q, grad_q = Q(obs, action, weight)
        q_next, _ = Q(obs, next_action, weight)
        weight = weight + alpha * (reward + q_next - q) * grad_q
        
        #保存选择的动作和观察，进入下一轮
        action = next_action
        obs = next_obs
        
        if done:
            performance.append(i + 1)  # 用每一幕的持续时间评判策略的好坏
            break
env.close()
plt.plot(performance)


def test():
    obs = env.reset()
    for i in range(2000):
        action = pi_soft(obs, weight, 0)
        env.render()
        obs, reward, done, info = env.step(action)
        next_action = pi_soft(next_obs, weight, epsilon)
    env.close()

test()
