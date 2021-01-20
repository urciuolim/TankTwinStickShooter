from os.path import isfile
import time

class AgentLogger:
    def __init__(self, agent, log_dir):
        self.agent = agent
        self.name = agent.name + "_Logger"
        self.log_dir = log_dir
        self.open_new_log()
        
    def __del__(self):
        if hasattr(self, "log_file"):
            self.close()
            
    def open_new_log(self, log_dir=None, close_old=True):
        if hasattr(self, "log_file"):
            if close_old:
                self.log_file.close()
            del self.log_file
        if log_dir == None:
            log_dir = self.log_dir
        if log_dir[-1] != '/':
            log_dir = log_dir + '/'
        self.log_file = open(log_dir + self.agent.name + "_" + time.strftime("%Y%m%d%H%M%S") + ".log", 'w')
        
    def close(self):
        if hasattr(self, "log_file"):
            self.log_file.close()
        
    def write_line(self, line):
        if not hasattr(self, "log_file"):
            self.open_new_log()
        self.log_file.write(str(line) + '\n')
        
    def get_action(self, state):
        self.write_line("STATE: " + str(state))
        action = self.agent.get_action(state)
        self.write_line("ACTION: " + str(action))
        return action