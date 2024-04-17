from copy import deepcopy
from configs.gym.default import default_args


walker2d_medium_args = deepcopy(default_args)
walker2d_medium_args["rollout_length"] = 5
walker2d_medium_args["penalty_coef"] = 0.5
walker2d_medium_args["dynamics_max_epochs"] = 30