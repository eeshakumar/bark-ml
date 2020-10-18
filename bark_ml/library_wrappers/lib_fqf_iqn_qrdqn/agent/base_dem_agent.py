import torch


class BaseDemonstratedAgent(object):

    def __init__(self, env, test_env, params, demonstrator):
        self._params = params
        self._demonstrator = demonstrator
        self.env = env
        self.test_env = test_env

        self.device = torch.device("cuda" if self._params["ML"]["BaseAgent"][
            "Cuda", "", True] and torch.cuda.is_available() else "cpu")

        self.use_per = True
        if self.use_per:
            beta_steps = (self.num_steps - self.start_steps) / \
                         self.update_interval
            print("Initializing Prioritized experience replay memory for demonstrator actions")
            self.memory = LazyPrioritizedDemMultiStepMemory(
                self._params["ML"]["BaseAgent"]["MemorySize", "", 10 ** 6],
                self.env.observation_space.shape,
                self.device,
                self._params["ML"]["BaseAgent"]["Gamma", "", 0.99],
                self._params["ML"]["BaseAgent"]["Multi_step", "", 1],
                beta_steps=beta_steps,
                epsilon_demo=self._params["ML"]["Demonstrator"]["EpsilonDemo", "", 1.0],
                epsilon_alpha=self._params["ML"]["Demonstrator"]["EpsilonAlpha", "", 0.001],
                per_beta=self._params["ML"]["Demonstrator"][
                    "PerBeta", "This param specifies the importance sampling weight",
                    0.6],
                per_beta_steps=self._params["ML"]["Demonstrator"][
                    "PerBetaSteps", "This param specifies the number of steps to update beta",
                    25000],
                demo_ratio=self._params["ML"]["Demonstrator"][
                    "DemoRatio", "This param specifies what proportion of capacity is for demo samples",
                    0.25])

        # get k: pre-training gradient update steps from params
        self.pre_training_steps = self._params["ML"]["Demonstrator"][
                    "PreTrainingSteps", "This param specifies number of steps to train target network",
                    750000]

    def train_episode(self):
        # train on demonstration memory, no env and step necessary
        # TODO: how many epochs? What about done?
        self.online_net.train()
        self.target_net.train()


        done = False

        for k in self.pre_training_steps:
            # grab n transitions - control with batch_size?
            # train target network with n transitions
            # TODO: How?
            # large margin loss is always 0 here?
            # get action from train network
            # get action from demonstrator
            # calculate large_margin_loss if it is a demo sample
            self.calculate_loss()
            update_params(networks=[self.online_net.dqn_net, self.online_net.cosine_net,
                                    self.online_net.quantile_net])
            # loss on target, update online net
            if k % self.update_interval == 0:
                self.update_target()

        return

    def run(self):
        # create demonstrations
        # TODO: Specify size? If not, prune to memory
        assert self._demonstrator is not None, "No Demonstrator found"
        demonstrations = self._demonstrator.create_demonstrations()
        for i, demo in enumerate(demonstrations):
            if i <= self.memory.capacity:
                state, action, reward, next_state, done, is_demo = demo
                self.memory.append(state, action, reward, next_state, done, is_demo)
            else:
                print("Demonstration capacity full, truncated.")
                break

    def calculate_large_margin_loss(self):
        # TODO: Implement large margin loss, entry from train_episode
        # LML is only on demonstrated data
        return

    def calculate_loss(self):
        """Calculate loss J(Q)"""
        return

    def calculate_l2_reg_loss(self):
        return