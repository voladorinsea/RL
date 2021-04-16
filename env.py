class environment():
    def __init__(self):
        self.env = gym.make('CartPole-v0')
    def observe_generate(self,action):
        return observation, reward, done, info = self.env.step(int(action))

    def env_init(self):
        return observation = self.env.reset()

