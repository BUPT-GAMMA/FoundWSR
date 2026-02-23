from ..pipeline import build_pipe
import optuna
from ..utils import set_random_seed


def hpo_experiment(args, pipeline, **kwargs):
    tool = AutoML(args, pipeline, n_trials=args.hpo_trials, search_space=args.hpo_search_space)
    performance, result = tool.run()
    print("[Hyper-parameter optimization] Final results:{}".format(performance))
    print(result)
    # logger.info("[Hyper-parameter optimization] Final results:{}".format(result))
    return performance, result


class AutoML(object):
    """
    Args:
        search_space: function to obtain hyper-parameters to search
    """

    def __init__(self, args, pipeline, n_trials=3, **kwargs):
        self.args = args
        self.pipeline = pipeline
        # self.seed = kwargs.pop("seed") if "seed" in kwargs else [1]
        assert "search_space" in kwargs
        self.search_space = kwargs["search_space"]
        self.n_trials = n_trials
        self.best_score = None
        self.best_result = None
        self.best_params = None
        self.default_params = kwargs

    def _objective(self, trials):
        args = self.args
        cur_params = self.search_space(trials)
        for key, value in cur_params.items():
            args.__setattr__(key, value)
        # Set random seed each time, or the initialization of the weight will be different.
        set_random_seed(args.seed)
        flow = build_pipe(args, self.pipeline)
        result = flow.train()
        score = result['Avg']
        # with open(args.dataset[0] + "trail.json", "a") as f:
        #     import json
        #     f.write(args.model + "\n")
        #     json.dump(result, f, ensure_ascii=False, indent=4)
        #     json.dump(cur_params, f, ensure_ascii=False, indent=4)
        #     f.write("\n")
        
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best_result = result
            self.best_params = cur_params
        return score

    def run(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self._objective, n_trials=self.n_trials, n_jobs=1)
        print("[Hyper-parameter optimization] Best parameter: {}".format(self.best_params))
        return self.best_score, self.best_result