import os
import logging

class DemonstrationGenerator(object):

    def __init__(self, env, params, behavior, collector, save_dir):
        self._env = env
        self._params = params
        self._behavior = behavior
        self._collector = collector
        self._save_dir = save_dir
        self._demonstrations = None

    @property
    def demonstrations(self):
        return self._demonstrations

    def generate_demonstrations(self, num_episodes, eval_criteria, save_dir=None,
                                gen_file="generated_demonstrations", use_mp_runner=False, db=None):
        if save_dir is None:
            save_dir = self._save_dir

        demo_path = os.path.join(save_dir, gen_file)
        collection_result = self._collector.CollectDemonstrations(
            self._env, self._behavior, num_episodes, demo_path, use_mp_runner=use_mp_runner,
            runner_init_params= None if use_mp_runner else {"deepcopy": False}, db=db)

        self._collector.ProcessCollectionResult(eval_criteria)
        self._demonstrations = self._collector.GetDemonstrationExperiences()
        return self._demonstrations

    def dump_demonstrations(self, save_dir=None, gen_file="generated_demonstrations"):
        if save_dir is None:
            save_dir = self._save_dir
        demo_path = os.path.join(save_dir, gen_file)
        logging.info(f"Demonstrations will be saved to {demo_path}")
        self._collector.dump(demo_path)
