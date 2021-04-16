# -*- coding: UTF-8 -*-
# based on SERSE method for cartpole control
import numpy as np
from math import pi
import gym
import random
from matplotlib import pyplot as plt
class SARSE():
    #alpha for updating value
    #epsilon for exploring and apply
    #gama for discount parameter
    #num for state-rank actually 2*num+1
    def __init__(self, alpha, epsilon, gama, num):
        self.alpha=alpha
        self.epsilon=epsilon
        self.gama=gama
        #displace,angle,speed,angle speed,action    
        self.state_action_value=np.zeros([2*num+2,2*num+2,2*num+2,2*num+2,2])
        #0 is left,1 is right
        self.policy=np.ones([2*num+2,2*num+2,2*num+2,2*num+2])
        self.amount=np.zeros([2*num+2,2*num+2,2*num+2,2*num+2,2])
        #[[state_action, current reward]...]
        self.num=num
        self.env = gym.make('CartPole-v0')

    def rank(self, observation):
        sample=[]
        #for displace
        displace=observation[0]
        scale=2*2.4/(2*self.num)
        if displace<-2.4:
            rank=0
        elif displace>=2.4:
            rank=2*self.num+1
        else:
            rank=int((displace+2.4)/scale+1)
        sample.extend([rank])
        #for angle
        angle=observation[2]/pi*180
        scale=2*15/(2*self.num)
        if angle<-15:
            rank=0
        elif angle>=15:
            rank=2*self.num+1
        else:
            rank=int((angle+15)/scale+1)
        sample.extend([rank])
        #for speed
        speed=observation[1]
        scale=2*2/(2*self.num)
        if speed<-2:
            rank=0
        elif speed>=2:
            rank=2*self.num+1
        else:
            rank=int((speed+2)/scale+1)
        sample.extend([rank])
        #for angle speed 
        anglespeed=observation[3]/pi*180
        scale=2*5/(2*self.num)
        if anglespeed<-5:
            rank=0
        elif anglespeed>=5:
            rank=2*self.num+1
        else:
            rank=int((anglespeed+5)/scale+1)
        sample.extend([rank])

        return sample

    def reward(self,sample):
        displace_percent=sample[0]/2.4
        angle_percent=sample[1]/pi*180/15
        R=-(0*displace_percent**2+angle_percent**2)
        return R
    
    def episode_generate(self):
        episode=[]
        observation = self.env.reset()
        self.env.render()
        check=1
        for t in range(200):
            sample=self.rank(observation)
            action=self.action_generate(sample)
            self.env.render()
            #action = env.action_space.sample()
            observation, reward, done, info = self.env.step(int(action))
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                check=0
                break
            #reward=self.reward(observation)
            sample.extend([action])
            episode.append([sample,1])
        if check==1:
            sample=self.rank(observation)
            action=-1
            sample.extend([action])
            episode.append([sample,10])
        else:
            sample=self.rank(observation)
            action=-1
            sample.extend([action])
            episode.append([sample,-10])
        return episode,t 
    def update_value(self,episode):
        length=len(episode)
        #value=episode[length-1][1]
        for i in range(length-1,0,-1):
            sample=episode[i][0]
            last_value=self.state_action_value[int(sample[0])][int(sample[1])][int(sample[2])][int(sample[3])][int(sample[4])]
            if i==length-1:
                value=episode[i][1]
            else:
                old_sample=episode[i+1][0]
                old_value=self.state_action_value[int(old_sample[0])][int(old_sample[1])][int(old_sample[2])][int(old_sample[3])][int(old_sample[4])]
                contemporary_value=episode[i][1]
                value=contemporary_value+self.gama*old_value
            amount=self.amount[int(sample[0])][int(sample[1])][int(sample[2])][int(sample[3])][int(sample[4])]
            self.state_action_value[int(sample[0])][int(sample[1])][int(sample[2])][int(sample[3])][int(sample[4])]=(last_value*amount+value)/(amount+1)
            self.amount[int(sample[0])][int(sample[1])][int(sample[2])][int(sample[3])][int(sample[4])]+=1
        
    def update_policy(self):
        for i in range(2*self.num+2):
            for j in range(2*self.num+2):
                for k in range(2*self.num+2):
                    for l in range(2*self.num+2):
                        gen1=random.random()
                        if gen1<self.epsilon:
                            if self.state_action_value[i][j][k][l][0]>=self.state_action_value[i][j][k][l][1]:
                                self.policy[i][j][k][l]=0
                            else:
                                self.policy[i][j][k][l]=1
                        else:
                            gen2=random.random()
                            if gen2<0.5:
                                self.policy[i][j][k][l]=0
                            else:
                                self.policy[i][j][k][l]=1

    def action_generate(self,sample):
        return self.policy[int(sample[0])][int(sample[1])][int(sample[2])][int(sample[3])]
            




if __name__ == "__main__":
    rank=4
    RLcontroler=SARSE(0.01,0,0.6,rank)
    round=400
    score=[]
    for i in range(round):
        #if i>=100:
        #    RLcontroler.epsilon=0.5
        #elif i>300:
        #    RLcontroler.epsilon=0.8
        #elif i>400:
        #    RLcontroler.epsilon=1
        if i<400:
            RLcontroler.epsilon=1/round*i
        else:
            RLcontroler.epsilon=1
        episode,t=RLcontroler.episode_generate()
        RLcontroler.update_value(episode)
        RLcontroler.update_policy()
        score.extend([t])
        print("finished round: {}".format(i))
    time=np.arange(1,round+1)
    series=np.array(score)
        
    plt.title("RLcontroler")
    plt.xlabel("time")
    plt.ylabel("score")
    plt.plot(time,series)
    plt.show()
    
    policy=np.array(RLcontroler.policy)
    policy=policy.reshape(((rank*2+2)**2,(rank*2+2)**2))
    print(policy)
    np.savetxt('policy.txt',policy,fmt='%0.8f')