# -*- coding: UTF-8 -*-
if __name__ == '__main__':
    print('开始学习')
    import time
    import gym
    import numpy as np
    #based on traditional controler
    import PID
    from matplotlib import pyplot as plt
    Pidout = PID.PIDcontrol(0.3,1000,5,0,0)
    Pidin = PID.PIDcontrol(0.5,1000,0,50,0)
    
    env = gym.make('CartPole-v1')
    check=1
    angle=[]
    for i_episode in range(1):
        observation = env.reset()
        env.render()
        for t in range(500):
            env.render()
            print(observation)
            angle.extend([observation[1]])
            out1=Pidout.control(observation[1])
            Pidin.modify(out1)
            out2=Pidin.control(observation[3])
            action = 1 if out2<=50 else 0
            #action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                check=0
                break
    
    

    #for drawing
    if check==1:
        print("winner!!!")
    t=np.arange(1,t+2)
    series=np.array(angle)
        
    plt.title("Pidcontroler")
    plt.xlabel("time")
    plt.ylabel("angle")
    plt.plot(t,series)
    plt.show()