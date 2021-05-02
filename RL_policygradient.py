import gym
import PID
import numpy as np
import random
class env():
    def __init__(self,end):
        self.time=0
        self.end=end
        self.env=gym.make('CartPole-v0')
    def episode_generate(self):
        self.time=0
        self.observation=self.env.reset()#np.append(self.env.reset(),[1],axis=0) 
        self.env.render()
    def episode_action(self,action):
        self.last_observation=self.observation
        observation, reward, done, info = self.env.step(int(action))
        observation=observation#np.append(observation,[1],axis=0) 
        self.observation=observation 
        self.env.render()
        self.time=self.time+1
        #if done and self.time!=self.end-1:
        #    reward=-1
        #    return  observation, reward, done, self.last_observation
        if self.time==self.end-1:
            reward=1
            done=1
        else:
            reward=1
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
            self.check=0
            self.openGame()
            # accelerate the train process
            self.value_network.epsilon-=0.002
            if self.value_network.epsilon<=0.1:
                self.value_network.epsilo=0.05
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
        self.w=np.zeros([4,2])
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
            if value0<=value1:
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
        # using TD(0)
        sample_value=reward+self.gama*next_value
        evaluate_value=self.state_valuesearch(last_state)


        last_state=np.mat(last_state).reshape(1,4)
        action=np.mat(self.action_to_onehot(action)).reshape(1,2)
        t=np.asarray(last_state.T*action)
        delta_w=self.learning_rate*(sample_value-evaluate_value)*t
        #print(delta_w)
        self.w=self.w+delta_w
        #print(self.w.T)
    def state_action_valuesearch(self,state,action):
        state=np.array(state)
        action=np.array(self.action_to_onehot(action))
        v1=np.matmul(state,self.w)
        v2=np.matmul(v1,action.T)
        return v2
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
            return [1,0]
        else:
            return [0,1]

if __name__=="__main__":
    goal=200
    trainset=1000
    learning_rate=0.1
    gama=1
    epsilon=1
    # setting training environment
    AI_env=env(goal)
    # setting value iteration model
    model=value_network(learning_rate,gama,epsilon)
    # develop AI engine for env
    AI=agent(trainset,AI_env,model,goal)
    # start training
    AI.training()