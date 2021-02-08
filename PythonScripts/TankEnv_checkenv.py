from TankEnv import TankEnv
from IndvTankEnv import IndvTankEnv
from stable_baselines3.common.env_checker import check_env

env = TankEnv()
ienv = IndvTankEnv(env)
check_env(ienv)
ienv.close()