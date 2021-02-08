from TankEnv import TankEnv
from IndvTankEnv import IndvTankEnv
import numpy as np

env = TankEnv()
ienv = IndvTankEnv(env)
state = ienv.reset()
print(state)
done = False
while not done:
    state, reward, done, info = ienv.step(np.random.rand(5))
    print(state)
state = ienv.reset()
print(state)
ienv.close()
