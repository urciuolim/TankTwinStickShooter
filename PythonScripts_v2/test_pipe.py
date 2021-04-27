from tank_env2 import TankEnv
import time

game_path = "../Builds/Linux_Server_5_x64/game.x86_64"
env = TankEnv(game_path)
for i in range(100):
    print(env.send(i))
env.close()