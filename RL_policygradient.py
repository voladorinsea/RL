import gym
import PID
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
        self.value_network.gradient(self.state, reward, self.last_state)
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
    def __init__(self):
        self.learning_rate=0.1
        self.Pidout=PID.PIDcontrol(0.3,1000,5,0,0)
        self.Pidin=PID.PIDcontrol(0.5,1000,0,50,0)
    def policysearch(self,state):
        out1=self.Pidout.control(state[1])
        self.Pidin.modify(out1)
        out2=self.Pidin.control(state[3])
        action = 1 if out2<=50 else 0
        return action
    def gradient(self,state,reward,last_state):
        return 0


if __name__=="__main__":
    goal=500
    trainset=20
    AI_env=env(goal)
    model=value_network()
    AI=agent(trainset,AI_env,model,goal)
    AI.training()