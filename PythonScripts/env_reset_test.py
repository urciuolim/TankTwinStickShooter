from TankEnv import TankEnv
from IndvTankEnv import IndvTankEnv

env = IndvTankEnv(TankEnv())

for _ in range (1000):
	env.reset()
    
env.close()