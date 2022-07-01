import random
import numpy as np
import numpy.random
from Structure_project.drl_sample_project_python.drl_lib.do_not_touch.contracts import SingleAgentEnv
from Structure_project.drl_sample_project_python.drl_lib.do_not_touch.result_structures import \
    PolicyAndActionValueFunction

def dyna_q(env: SingleAgentEnv, max_iter_count: int = 10000,
               gamma: float = 0.99,
               alpha: float = 0.1,
               epsilon: float = 0.2,
               round: int = 20,
               save_path: str = '../../drl_sample_project_python/drl_lib/to_do/models/dyna_q_model.pkl'):
        q = {}
        Model = {}

        for it in range(max_iter_count):
            if env.is_game_over():
                env.reset()

            s = env.state_id()
            aa = env.available_actions_ids()

            if s not in q:
                q[s] = {}
                for a in aa:
                    q[s][a] = 0.0

            if random.random() <= epsilon:
                a = np.random.choice(aa)
            else:
                a = aa[np.argmax([q[s][a] for a in aa])]

            old_score = env.score()
            env.act_with_action_id(a)
            new_score = env.score()
            r = new_score - old_score

            s_p = env.state_id()
            aa_p = env.available_actions_ids()

            if env.is_game_over():
                q[s][a] += alpha * (r - q[s][a])
            else:
                if s_p not in q:
                    q[s_p] = {}
                    for a_p in aa_p:
                        q[s_p][a_p] = 0.0
                q[s][a] += alpha * (r + gamma * np.max([q[s_p][a_p] for a_p in aa_p]) - q[s][a])

            if s not in Model:
                Model[s] = {a: (s_p, r)}
            else:
                Model[s][a] = (s_p, r)

            for _ in range(round):
                s = np.random.choice(list(Model))
                actions = Model[s]
                a = np.random.choice(list(actions))
                s_p, r = actions[a]
                if s_p not in q:
                    q[s][a] += alpha * (r - q[s][a])
                else:
                    q[s][a] += alpha * (r + gamma * np.max([q[s_p][a_p] for a_p in q[s_p]]) - q[s][a])

        pi = {}
        for (s, a_dict) in q.items():
            pi[s] = {}
            actions = []
            q_values = []
            for (a, q_value) in a_dict.items():
                actions.append(a)
                q_values.append(q_value)

            best_action_idx = actions[np.argmax(q_values)]
            for i in range(len(actions)):
                pi[s][actions[i]] = 1.0 if actions[i] == best_action_idx else 0.0
        return PolicyAndActionValueFunction(q=q, pi=pi)
