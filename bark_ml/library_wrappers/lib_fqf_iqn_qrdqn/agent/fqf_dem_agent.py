from .base_dem_agent import  BaseDemonstratedAgent
from .fqf_agent import FQFAgent

class FQFDemonstratedAgent(BaseDemonstratedAgent, FQFAgent):

    def __init__(self, env, test_env, params, demonstrator, is_online=True):
        if is_online:
            FQFAgent.__init__(env, test_env, params)
        else:
            assert demonstrator is not None, "Demonstrator needed for offline training"
            BaseDemonstratedAgent.__init__(env, test_env, params, demonstrator)

    def load_demonstrations(self, filename):
        # import pickle
        # with open(filename, "r+") as f:
        #     return pickle.load(f)


    def train_episode(self):
        # append self generated samples
        return

    def run(self):
        # fill memory with demonstrator samples
        dem_samples =

        return

    def calculate_large_margin_loss(self):
        return
    
    def calculate_loss(self):
        return
    
    def calculate_l2_reg_loss(self):
        return
