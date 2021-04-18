import gym
import PID
import numpy as np
import random
class env():
    def __init__(self,end):
        self.time=0
        self.end=end
        self.env=gym.make('CartPole-v1')
    def episode_generate(self):
        self.observation=self.env.reset()
        self.env.render()
    def episode_action(self,action):
        self.last_observation=self.observation
        observation, reward, done, info = self.env.step(int(action))
        self.observation=observation
        self.env.render()
        if done:
            reward=-1
            return  observation, reward, done, self.last_observation
        if self.time==self.end:
            reward=1
            done=1
        else:
            reward=0
        return  observation, reward, done, self.last_observation

class agent():
    def __init__(self,cycle,env,value_network,step):
        self.check=0
        self.cycle=cycle 
        self.env=env
        self.value_network=value_network
        self.aimstep=step
    def openGame(self):
        self.env.episode_generate()
        self.state=self.env.observation
    def execute(self):
        action=self.value_network.policysearch(self.state)
        self.state, reward, self.check, self.last_state=self.env.episode_action(action)
        self.value_network.gradient(self.state, reward, self.last_state,action)
    def training(self):
        cycle=0
        while(cycle!=self.cycle):
            self.openGame()
            for i in range(self.aimstep):
                self.execute()
                if self.check==1:
                    print("Episode finished after {} timesteps".format(i + 1))
                    break
            cycle=cycle+1
        print("training finished!!!")


class value_network():
    def __init__(self,learning_rate,gama,epsilon):
        self.learning_rate=learning_rate
        self.w=np.array([1,1,1,1,1,1])
        self.gama=gama
        self.epsilon=epsilon
        '''
        self.Pidout=PID.PIDcontrol(0.3,1000,5,0,0)
        self.Pidin=PID.PIDcontrol(0.5,1000,0,50,0)
        '''
    def policysearch(self,state):
        # PID for a test
        '''
        out1=self.Pidout.control(state[1])
        self.Pidin.modify(out1)
        out2=self.Pidin.control(state[3])
        action = 1 if out2<=50 else 0
        '''
        # based on epsilon-greedy policy
        value0=self.state_action_valuesearch(state,0)
        value1=self.state_action_valuesearch(state,1)

        gen1=random.random()
        if gen1>=self.epsilon:
            if value0>=value1:
                return 0
            else:
                return 1
        else:
            gen2=random.random()
            if gen2>0.5:
                return 0
            else:
                return 1
    def gradient(self,state,reward,last_state,action):
        next_value=self.state_valuesearch(state)
        sample_value=reward+self.gama*next_value
        evaluate_value=self.state_valuesearch(last_state)

        last_state=np.array(last_state)

        action=np.array(self.action_to_onehot(action))
        state_action=np.hstack([state,action]).T
        delta_w=self.learning_rate*(evaluate_value-sample_value)*state_action
        self.w=self.w-delta_w
        print(self.w.T)
    def state_action_valuesearch(self,state,action):
        state=np.array(state)
        action=np.array(self.action_to_onehot(action))
        state_action=np.hstack([state,action]).T
        return np.matmul(self.w.T,state_action)
    def state_valuesearch(self,state):
        # based on best policy principle
        value0=self.state_action_valuesearch(state,0)
        value1=self.state_action_valuesearch(state,1)
        # print(value0)
        if value0>=value1:
            return value0
        else:
            return value1
    def action_to_onehot(self,action):
        if action==0:
            return [0,1]
        else:
            return [1,0]

if __name__=="__main__":
    goal=300
    trainset=2000
    learning_rate=0.01
    gama=0.5
    epsilon=0.5
    AI_env=env(goal)
    model=value_network(learning_rate,gama,epsilon)
    AI=agent(trainset,AI_env,model,goal)
    AI.training()