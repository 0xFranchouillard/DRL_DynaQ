from Structure_project.drl_sample_project_python.drl_lib.do_not_touch.result_structures import \
    PolicyAndActionValueFunction
from Structure_project.drl_sample_project_python.drl_lib.do_not_touch.single_agent_env_wrapper import Env3
from Structure_project.src.envs import TicTacToe, GridWorld, LineWorld
from Structure_project.src.temporal_difference_learning.q_learning import q_learning
from Structure_project.src.temporal_difference_learning.sarsa import sarsa
from Structure_project.src.temporal_difference_learning.expected_sarsa import expected_sarsa
from Structure_project.src.temporal_difference_learning.dyna_Q import dyna_q
from Structure_project.src.utils import save_model
import numpy as np


def sarsa_on_line_world() -> PolicyAndActionValueFunction:
    """
    Creates a Line World environment (Single player versus Uniform Random Opponent)
    Launches a Monte Carlo ES (Exploring Starts) in order to find the optimal Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    """
    env = LineWorld(7)

    number_of_iterations = np.linspace(100, 100_000, 3).tolist()
    epsilons = np.linspace(0.001, 0.5, 3).tolist()
    alphas = np.linspace(0.01, 0.5, 3).tolist()

    for i, number_of_iteration in enumerate(number_of_iterations):
        print(f"-- Training ~~ Iteration n°{i + 1}/{len(number_of_iterations)} --")
        for epsilon in epsilons:
            for alpha in alphas:
                win_rate = 0.0
                game_played = 0
                model = sarsa(env, max_iter_count=int(number_of_iteration),
                              epsilon=float(epsilon), alpha=float(alpha))
                # Test a model and evaluate it
                print(
                    f"-- Testing start ~~ Iteration n°{i + 1}/{len(number_of_iterations)} -- with number_of_iteration : {int(number_of_iteration)} / epsilon : {int(epsilon)}  / alpha : {float(alpha)}--")
                for game in range(1_000):
                    env.reset()
                    while not env.is_game_over():
                        s = env.state_id()
                        aa = env.available_actions_ids()
                        if s not in model.pi:
                            a = np.random.choice(aa, p=[1 / len(aa) for _ in aa])
                        else:
                            a = np.random.choice(aa, p=[model.pi[s][a_p] for a_p in aa])
                        env.act_with_action_id(a)
                    game_played += 1
                    win_rate += env.score()
                score = round(win_rate / game_played, 4)
                print(
                    f"-- Testing done with a score of {score}% of win rate ~~ Iteration n°{i}/{len(number_of_iterations)} / epsilon : {float(epsilon)} / alpha : {float(alpha)}--")
                # Save model with score
                save_model(
                    f"sarsa_on_line_world_it_{int(number_of_iteration)}_eps_{round(epsilon, 3)}_alp_{round(alpha, 3)}_s_{score}",
                    q=model.q, pi=model.pi)


def sarsa_on_grid_world() -> PolicyAndActionValueFunction:
    """
    Creates a Grid World environment (Single player versus Uniform Random Opponent)
    Launches a Monte Carlo ES (Exploring Starts) in order to find the optimal Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    """
    env = GridWorld()

    number_of_iterations = np.linspace(100, 100_000, 3).tolist()
    epsilons = np.linspace(0.001, 0.5, 3).tolist()
    alphas = np.linspace(0.01, 0.5, 3).tolist()

    for i, number_of_iteration in enumerate(number_of_iterations):
        print(f"-- Training ~~ Iteration n°{i + 1}/{len(number_of_iterations)} --")
        for epsilon in epsilons:
            for alpha in alphas:
                win_rate = 0.0
                game_played = 0
                model = sarsa(env, max_iter_count=int(number_of_iteration),
                              epsilon=float(epsilon), alpha=float(alpha))
                # Test a model and evaluate it
                print(
                    f"-- Testing start ~~ Iteration n°{i + 1}/{len(number_of_iterations)} -- with number_of_iteration : {int(number_of_iteration)} / epsilon : {int(epsilon)}  / alpha : {float(alpha)}--")
                for game in range(1_000):
                    env.reset()
                    while not env.is_game_over():
                        s = env.state_id()
                        aa = env.available_actions_ids()
                        if s not in model.pi:
                            a = np.random.choice(aa, p=[1 / len(aa) for _ in aa])
                        else:
                            a = np.random.choice(aa, p=[model.pi[s][a_p] for a_p in aa])
                        env.act_with_action_id(a)
                    game_played += 1
                    win_rate += env.score()
                score = round(win_rate / game_played, 4)
                print(
                    f"-- Testing done with a score of {score}% of win rate ~~ Iteration n°{i}/{len(number_of_iterations)} / epsilon : {float(epsilon)} / alpha : {float(alpha)}--")
                # Save model with score
                save_model(
                    f"sarsa_on_grid_world_it_{int(number_of_iteration)}_eps_{round(epsilon, 3)}_alp_{round(alpha, 3)}_s_{score}",
                    q=model.q, pi=model.pi)


def sarsa_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    env = TicTacToe()

    number_of_iterations = np.linspace(100, 100_000, 3).tolist()
    epsilons = np.linspace(0.001, 0.5, 3).tolist()
    alphas = np.linspace(0.01, 0.5, 3).tolist()

    for i, number_of_iteration in enumerate(number_of_iterations):
        print(f"-- Training ~~ Iteration n°{i + 1}/{len(number_of_iterations)} --")
        for epsilon in epsilons:
            for alpha in alphas:
                win_rate = 0.0
                game_played = 0
                model = sarsa(env, max_iter_count=int(number_of_iteration),
                              epsilon=float(epsilon), alpha=float(alpha))
                # Test a model and evaluate it
                print(
                    f"-- Testing start ~~ Iteration n°{i + 1}/{len(number_of_iterations)} -- with number_of_iteration : {int(number_of_iteration)} / epsilon : {int(epsilon)}  / alpha : {float(alpha)}--")
                for game in range(1_000):
                    env.reset()
                    while not env.is_game_over():
                        s = env.state_id()
                        aa = env.available_actions_ids()
                        if s not in model.pi:
                            a = np.random.choice(aa, p=[1 / len(aa) for _ in aa])
                        else:
                            a = np.random.choice(aa, p=[model.pi[s][a_p] for a_p in aa])
                        env.act_with_action_id(a)
                    game_played += 1
                    win_rate += env.score()
                score = round(win_rate / game_played, 4)
                print(
                    f"-- Testing done with a score of {score}% of win rate ~~ Iteration n°{i}/{len(number_of_iterations)} / epsilon : {float(epsilon)} / alpha : {float(alpha)}--")
                # Save model with score
                save_model(
                    f"sarsa_on_ttt_it_{int(number_of_iteration)}_eps_{round(epsilon, 3)}_alp_{round(alpha, 3)}_s_{score}",
                    q=model.q, pi=model.pi)

def dyna_q_on_line_world() -> PolicyAndActionValueFunction:
    env = LineWorld(7)

    number_of_iterations = np.linspace(100, 100_000, 3).tolist()
    epsilons = np.linspace(0.001, 0.5, 3).tolist()
    alphas = np.linspace(0.01, 0.5, 3).tolist()

    for i, number_of_iteration in enumerate(number_of_iterations):
        print(f"-- Training ~~ Iteration n°{i + 1}/{len(number_of_iterations)} --")
        for epsilon in epsilons:
            for alpha in alphas:
                win_rate = 0.0
                game_played = 0
                model = dyna_q(env, max_iter_count=int(number_of_iteration),
                              epsilon=float(epsilon), alpha=float(alpha))
                # Test a model and evaluate it
                print(
                    f"-- Testing start ~~ Iteration n°{i + 1}/{len(number_of_iterations)} -- with number_of_iteration : {int(number_of_iteration)} / epsilon : {int(epsilon)}  / alpha : {float(alpha)}--")
                for game in range(1_000):
                    env.reset()
                    while not env.is_game_over():
                        s = env.state_id()
                        aa = env.available_actions_ids()
                        if s not in model.pi:
                            a = np.random.choice(aa, p=[1 / len(aa) for _ in aa])
                        else:
                            a = np.random.choice(aa, p=[model.pi[s][a_p] for a_p in aa])
                        env.act_with_action_id(a)
                    game_played += 1
                    win_rate += env.score()
                score = round(win_rate / game_played, 4)
                print(
                    f"-- Testing done with a score of {score}% of win rate ~~ Iteration n°{i}/{len(number_of_iterations)} / epsilon : {float(epsilon)} / alpha : {float(alpha)}--")
                # Save model with score
                save_model(
                    f"dyna_q_on_line_world_it_{int(number_of_iteration)}_eps_{round(epsilon, 3)}_alp_{round(alpha, 3)}_s_{score}",
                    q=model.q, pi=model.pi)
def q_learning_on_line_world() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Q-Learning algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the optimal greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    env = LineWorld(7)

    number_of_iterations = np.linspace(100, 100_000, 3).tolist()
    epsilons = np.linspace(0.001, 0.5, 3).tolist()
    alphas = np.linspace(0.01, 0.5, 3).tolist()

    for i, number_of_iteration in enumerate(number_of_iterations):
        print(f"-- Training ~~ Iteration n°{i + 1}/{len(number_of_iterations)} --")
        for epsilon in epsilons:
            for alpha in alphas:
                win_rate = 0.0
                game_played = 0
                model = q_learning(env, max_iter_count=int(number_of_iteration),
                                   epsilon=float(epsilon), alpha=float(alpha))
                # Test a model and evaluate it
                print(
                    f"-- Testing start ~~ Iteration n°{i + 1}/{len(number_of_iterations)} -- with number_of_iteration : {int(number_of_iteration)} / epsilon : {int(epsilon)}  / alpha : {float(alpha)}--")
                for game in range(1_000):
                    env.reset()
                    while not env.is_game_over():
                        s = env.state_id()
                        aa = env.available_actions_ids()
                        if s not in model.pi:
                            a = np.random.choice(aa, p=[1 / len(aa) for _ in aa])
                        else:
                            a = np.random.choice(aa, p=[model.pi[s][a_p] for a_p in aa])
                        env.act_with_action_id(a)
                    game_played += 1
                    win_rate += env.score()
                score = round(win_rate / game_played, 4)
                print(
                    f"-- Testing done with a score of {score}% of win rate ~~ Iteration n°{i}/{len(number_of_iterations)} / epsilon : {float(epsilon)} / alpha : {float(alpha)}--")
                # Save model with score
                save_model(
                    f"q_learning_on_line_world_it_{int(number_of_iteration)}_eps_{round(epsilon, 3)}_alp_{round(alpha, 3)}_s_{score}",
                    q=model.q, pi=model.pi)

def dyna_q_on_grid_world() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Dyna-Q algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the optimal greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    env = GridWorld(7)

    number_of_iterations = np.linspace(100, 100_000, 3).tolist()
    epsilons = np.linspace(0.001, 0.5, 3).tolist()
    alphas = np.linspace(0.01, 0.5, 3).tolist()

    for i, number_of_iteration in enumerate(number_of_iterations):
        print(f"-- Training ~~ Iteration n°{i + 1}/{len(number_of_iterations)} --")
        for epsilon in epsilons:
            for alpha in alphas:
                win_rate = 0.0
                game_played = 0
                model = dyna_q(env, max_iter_count=int(number_of_iteration),
                               epsilon=float(epsilon), alpha=float(alpha))
                # Test a model and evaluate it
                print(
                    f"-- Testing start ~~ Iteration n°{i + 1}/{len(number_of_iterations)} -- with number_of_iteration : {int(number_of_iteration)} / epsilon : {int(epsilon)}  / alpha : {float(alpha)}--")
                for game in range(1_000):
                    env.reset()
                    while not env.is_game_over():
                        s = env.state_id()
                        aa = env.available_actions_ids()
                        if s not in model.pi:
                            a = np.random.choice(aa, p=[1 / len(aa) for _ in aa])
                        else:
                            a = np.random.choice(aa, p=[model.pi[s][a_p] for a_p in aa])
                        env.act_with_action_id(a)
                    game_played += 1
                    win_rate += env.score()
                score = round(win_rate / game_played, 4)
                print(
                    f"-- Testing done with a score of {score}% of win rate ~~ Iteration n°{i}/{len(number_of_iterations)} / epsilon : {float(epsilon)} / alpha : {float(alpha)}--")
                # Save model with score
                save_model(
                    f"dyna_q_on_grid_world_it_{int(number_of_iteration)}_eps_{round(epsilon, 3)}_alp_{round(alpha, 3)}_s_{score}",
                    q=model.q, pi=model.pi)

def q_learning_on_grid_world() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Q-Learning algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the optimal greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    env = GridWorld()

    number_of_iterations = np.linspace(100, 100_000, 3).tolist()
    epsilons = np.linspace(0.001, 0.5, 3).tolist()
    alphas = np.linspace(0.01, 0.5, 3).tolist()

    for i, number_of_iteration in enumerate(number_of_iterations):
        print(f"-- Training ~~ Iteration n°{i + 1}/{len(number_of_iterations)} --")
        for epsilon in epsilons:
            for alpha in alphas:
                win_rate = 0.0
                game_played = 0
                model = q_learning(env, max_iter_count=int(number_of_iteration),
                                   epsilon=float(epsilon), alpha=float(alpha))
                # Test a model and evaluate it
                print(
                    f"-- Testing start ~~ Iteration n°{i + 1}/{len(number_of_iterations)} -- with number_of_iteration : {int(number_of_iteration)} / epsilon : {int(epsilon)}  / alpha : {float(alpha)}--")
                for game in range(1_000):
                    env.reset()
                    while not env.is_game_over():
                        s = env.state_id()
                        aa = env.available_actions_ids()
                        if s not in model.pi:
                            a = np.random.choice(aa, p=[1 / len(aa) for _ in aa])
                        else:
                            a = np.random.choice(aa, p=[model.pi[s][a_p] for a_p in aa])
                        env.act_with_action_id(a)
                    game_played += 1
                    win_rate += env.score()
                score = round(win_rate / game_played, 4)
                print(
                    f"-- Testing done with a score of {score}% of win rate ~~ Iteration n°{i}/{len(number_of_iterations)} / epsilon : {float(epsilon)} / alpha : {float(alpha)}--")
                # Save model with score
                save_model(
                    f"q_learning_on_grid_world_it_{int(number_of_iteration)}_eps_{round(epsilon, 3)}_alp_{round(alpha, 3)}_s_{score}",
                    q=model.q, pi=model.pi)

def dyna_q_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Dyna-Q algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the optimal greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    env = TicTacToe()

    number_of_iterations = np.linspace(100, 100_000, 3).tolist()
    epsilons = np.linspace(0.001, 0.5, 3).tolist()
    alphas = np.linspace(0.01, 0.5, 3).tolist()

    for i, number_of_iteration in enumerate(number_of_iterations):
        print(f"-- Training ~~ Iteration n°{i + 1}/{len(number_of_iterations)} --")
        for epsilon in epsilons:
            for alpha in alphas:
                win_rate = 0.0
                game_played = 0
                model = dyna_q(env, max_iter_count=int(number_of_iteration),
                               epsilon=float(epsilon), alpha=float(alpha))
                # Test a model and evaluate it
                print(
                    f"-- Testing start ~~ Iteration n°{i + 1}/{len(number_of_iterations)} -- with number_of_iteration : {int(number_of_iteration)} / epsilon : {int(epsilon)}  / alpha : {float(alpha)}--")
                for game in range(1_000):
                    env.reset()
                    while not env.is_game_over():
                        s = env.state_id()
                        aa = env.available_actions_ids()
                        if s not in model.pi:
                            a = np.random.choice(aa, p=[1 / len(aa) for _ in aa])
                        else:
                            a = np.random.choice(aa, p=[model.pi[s][a_p] for a_p in aa])
                        env.act_with_action_id(a)
                    game_played += 1
                    win_rate += env.score()
                score = round(win_rate / game_played, 4)
                print(
                    f"-- Testing done with a score of {score}% of win rate ~~ Iteration n°{i}/{len(number_of_iterations)} / epsilon : {float(epsilon)} / alpha : {float(alpha)}--")
                # Save model with score
                save_model(
                    f"dyna_q_on_ttt_it_{int(number_of_iteration)}_eps_{round(epsilon, 3)}_alp_{round(alpha, 3)}_s_{score}",
                    q=model.q, pi=model.pi)

def q_learning_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Q-Learning algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the optimal greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    env = TicTacToe()

    number_of_iterations = np.linspace(100, 100_000, 3).tolist()
    epsilons = np.linspace(0.001, 0.5, 3).tolist()
    alphas = np.linspace(0.01, 0.5, 3).tolist()

    for i, number_of_iteration in enumerate(number_of_iterations):
        print(f"-- Training ~~ Iteration n°{i + 1}/{len(number_of_iterations)} --")
        for epsilon in epsilons:
            for alpha in alphas:
                win_rate = 0.0
                game_played = 0
                model = q_learning(env, max_iter_count=int(number_of_iteration),
                                   epsilon=float(epsilon), alpha=float(alpha))
                # Test a model and evaluate it
                print(
                    f"-- Testing start ~~ Iteration n°{i + 1}/{len(number_of_iterations)} -- with number_of_iteration : {int(number_of_iteration)} / epsilon : {int(epsilon)}  / alpha : {float(alpha)}--")
                for game in range(1_000):
                    env.reset()
                    while not env.is_game_over():
                        s = env.state_id()
                        aa = env.available_actions_ids()
                        if s not in model.pi:
                            a = np.random.choice(aa, p=[1 / len(aa) for _ in aa])
                        else:
                            a = np.random.choice(aa, p=[model.pi[s][a_p] for a_p in aa])
                        env.act_with_action_id(a)
                    game_played += 1
                    win_rate += env.score()
                score = round(win_rate / game_played, 4)
                print(
                    f"-- Testing done with a score of {score}% of win rate ~~ Iteration n°{i}/{len(number_of_iterations)} / epsilon : {float(epsilon)} / alpha : {float(alpha)}--")
                # Save model with score
                save_model(
                    f"q_learning_on_ttt_it_{int(number_of_iteration)}_eps_{round(epsilon, 3)}_alp_{round(alpha, 3)}_s_{score}",
                    q=model.q, pi=model.pi)


def expected_sarsa_on_line_world() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Expected SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = LineWorld(7)

    number_of_iterations = np.linspace(100, 100_000, 3).tolist()
    epsilons = np.linspace(0.001, 0.5, 3).tolist()
    alphas = np.linspace(0.01, 0.5, 3).tolist()

    for i, number_of_iteration in enumerate(number_of_iterations):
        print(f"-- Training ~~ Iteration n°{i + 1}/{len(number_of_iterations)} --")
        for epsilon in epsilons:
            for alpha in alphas:
                win_rate = 0.0
                game_played = 0
                model = expected_sarsa(env, max_iter_count=int(number_of_iteration),
                                   epsilon=float(epsilon), alpha=float(alpha))
                # Test a model and evaluate it
                print(
                    f"-- Testing start ~~ Iteration n°{i + 1}/{len(number_of_iterations)} -- with number_of_iteration : {int(number_of_iteration)} / epsilon : {int(epsilon)}  / alpha : {float(alpha)}--")
                for game in range(1_000):
                    env.reset()
                    while not env.is_game_over():
                        s = env.state_id()
                        aa = env.available_actions_ids()
                        if s not in model.pi:
                            a = np.random.choice(aa, p=[1 / len(aa) for _ in aa])
                        else:
                            a = np.random.choice(aa, p=[model.pi[s][a_p] for a_p in aa])
                        env.act_with_action_id(a)
                    game_played += 1
                    win_rate += env.score()
                score = round(win_rate / game_played, 4)
                print(
                    f"-- Testing done with a score of {score}% of win rate ~~ Iteration n°{i}/{len(number_of_iterations)} / epsilon : {float(epsilon)} / alpha : {float(alpha)}--")
                # Save model with score
                save_model(
                    f"expected_sarsa_on_line_world_it_{int(number_of_iteration)}_eps_{round(epsilon, 3)}_alp_{round(alpha, 3)}_s_{score}",
                    q=model.q, pi=model.pi)


def expected_sarsa_on_grid_world() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Expected SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    env = GridWorld()

    number_of_iterations = np.linspace(100, 100_000, 3).tolist()
    epsilons = np.linspace(0.001, 0.5, 3).tolist()
    alphas = np.linspace(0.01, 0.5, 3).tolist()

    for i, number_of_iteration in enumerate(number_of_iterations):
        print(f"-- Training ~~ Iteration n°{i + 1}/{len(number_of_iterations)} --")
        for epsilon in epsilons:
            for alpha in alphas:
                win_rate = 0.0
                game_played = 0
                model = expected_sarsa(env, max_iter_count=int(number_of_iteration),
                                   epsilon=float(epsilon), alpha=float(alpha))
                # Test a model and evaluate it
                print(
                    f"-- Testing start ~~ Iteration n°{i + 1}/{len(number_of_iterations)} -- with number_of_iteration : {int(number_of_iteration)} / epsilon : {int(epsilon)}  / alpha : {float(alpha)}--")
                for game in range(1_000):
                    env.reset()
                    while not env.is_game_over():
                        s = env.state_id()
                        aa = env.available_actions_ids()
                        if s not in model.pi:
                            a = np.random.choice(aa, p=[1 / len(aa) for _ in aa])
                        else:
                            a = np.random.choice(aa, p=[model.pi[s][a_p] for a_p in aa])
                        env.act_with_action_id(a)
                    game_played += 1
                    win_rate += env.score()
                score = round(win_rate / game_played, 4)
                print(
                    f"-- Testing done with a score of {score}% of win rate ~~ Iteration n°{i}/{len(number_of_iterations)} / epsilon : {float(epsilon)} / alpha : {float(alpha)}--")
                # Save model with score
                save_model(
                    f"expected_sarsa_on_grid_world_it_{int(number_of_iteration)}_eps_{round(epsilon, 3)}_alp_{round(alpha, 3)}_s_{score}",
                    q=model.q, pi=model.pi)


def expected_sarsa_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Expected SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    env = TicTacToe()

    number_of_iterations = np.linspace(100, 100_000, 3).tolist()
    epsilons = np.linspace(0.001, 0.5, 3).tolist()
    alphas = np.linspace(0.01, 0.5, 3).tolist()

    for i, number_of_iteration in enumerate(number_of_iterations):
        print(f"-- Training ~~ Iteration n°{i + 1}/{len(number_of_iterations)} --")
        for epsilon in epsilons:
            for alpha in alphas:
                win_rate = 0.0
                game_played = 0
                model = expected_sarsa(env, max_iter_count=int(number_of_iteration),
                                   epsilon=float(epsilon), alpha=float(alpha))
                # Test a model and evaluate it
                print(
                    f"-- Testing start ~~ Iteration n°{i + 1}/{len(number_of_iterations)} -- with number_of_iteration : {int(number_of_iteration)} / epsilon : {int(epsilon)}  / alpha : {float(alpha)}--")
                for game in range(1_000):
                    env.reset()
                    while not env.is_game_over():
                        s = env.state_id()
                        aa = env.available_actions_ids()
                        if s not in model.pi:
                            a = np.random.choice(aa, p=[1 / len(aa) for _ in aa])
                        else:
                            a = np.random.choice(aa, p=[model.pi[s][a_p] for a_p in aa])
                        env.act_with_action_id(a)
                    game_played += 1
                    win_rate += env.score()
                score = round(win_rate / game_played, 4)
                print(
                    f"-- Testing done with a score of {score}% of win rate ~~ Iteration n°{i}/{len(number_of_iterations)} / epsilon : {float(epsilon)} / alpha : {float(alpha)}--")
                # Save model with score
                save_model(
                    f"expected_sarsa_on_ttt_it_{int(number_of_iteration)}_eps_{round(epsilon, 3)}_alp_{round(alpha, 3)}_s_{score}",
                    q=model.q, pi=model.pi)


def sarsa_on_secret_env3() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env3
    Launches a SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env3()

    number_of_iterations = np.linspace(100, 100_000, 3).tolist()
    epsilons = np.linspace(0.001, 0.5, 3).tolist()
    alphas = np.linspace(0.01, 0.5, 3).tolist()

    for i, number_of_iteration in enumerate(number_of_iterations):
        print(f"-- Training ~~ Iteration n°{i + 1}/{len(number_of_iterations)} --")
        for epsilon in epsilons:
            for alpha in alphas:
                win_rate = 0.0
                game_played = 0
                model = sarsa(env, max_iter_count=int(number_of_iteration),
                              epsilon=float(epsilon), alpha=float(alpha))
                # Test a model and evaluate it
                print(
                    f"-- Testing start ~~ Iteration n°{i + 1}/{len(number_of_iterations)} -- with number_of_iteration : {int(number_of_iteration)} / epsilon : {int(epsilon)}  / alpha : {float(alpha)}--")
                for game in range(1_000):
                    env.reset()
                    while not env.is_game_over():
                        s = env.state_id()
                        aa = env.available_actions_ids()
                        if s not in model.pi:
                            a = np.random.choice(aa, p=[1 / len(aa) for _ in aa])
                        else:
                            a = np.random.choice(aa, p=[model.pi[s][a_p] for a_p in aa])
                        env.act_with_action_id(a)
                    game_played += 1
                    win_rate += env.score()
                score = round(win_rate / game_played, 4)
                print(
                    f"-- Testing done with a score of {score}% of win rate ~~ Iteration n°{i}/{len(number_of_iterations)} / epsilon : {float(epsilon)} / alpha : {float(alpha)}--")
                # Save model with score
                save_model(
                    f"sarsa_on_env3_it_{int(number_of_iteration)}_eps_{round(epsilon, 3)}_alp_{round(alpha, 3)}_s_{score}",
                    q=model.q, pi=model.pi)


def q_learning_on_secret_env3() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env3
    Launches a Q-Learning algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the optimal greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    env = Env3()

    number_of_iterations = np.linspace(100, 100_000, 3).tolist()
    epsilons = np.linspace(0.001, 0.5, 3).tolist()
    alphas = np.linspace(0.01, 0.5, 3).tolist()

    for i, number_of_iteration in enumerate(number_of_iterations):
        print(f"-- Training ~~ Iteration n°{i + 1}/{len(number_of_iterations)} --")
        for epsilon in epsilons:
            for alpha in alphas:
                win_rate = 0.0
                game_played = 0
                model = q_learning(env, max_iter_count=int(number_of_iteration),
                                   epsilon=float(epsilon), alpha=float(alpha))
                # Test a model and evaluate it
                print(
                    f"-- Testing start ~~ Iteration n°{i + 1}/{len(number_of_iterations)} -- with number_of_iteration : {int(number_of_iteration)} / epsilon : {int(epsilon)}  / alpha : {float(alpha)}--")
                for game in range(1_000):
                    env.reset()
                    while not env.is_game_over():
                        s = env.state_id()
                        aa = env.available_actions_ids()
                        if s not in model.pi:
                            a = np.random.choice(aa, p=[1 / len(aa) for _ in aa])
                        else:
                            a = np.random.choice(aa, p=[model.pi[s][a_p] for a_p in aa])
                        env.act_with_action_id(a)
                    game_played += 1
                    win_rate += env.score()
                score = round(win_rate / game_played, 4)
                print(
                    f"-- Testing done with a score of {score}% of win rate ~~ Iteration n°{i}/{len(number_of_iterations)} / epsilon : {float(epsilon)} / alpha : {float(alpha)}--")
                # Save model with score
                save_model(
                    f"q_learning_on_env3_it_{int(number_of_iteration)}_eps_{round(epsilon, 3)}_alp_{round(alpha, 3)}_s_{score}",
                    q=model.q, pi=model.pi)


def expected_sarsa_on_secret_env3() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env3
    Launches a Expected SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    env = Env3()

    number_of_iterations = np.linspace(100, 100_000, 3).tolist()
    epsilons = np.linspace(0.001, 0.5, 3).tolist()
    alphas = np.linspace(0.01, 0.5, 3).tolist()

    for i, number_of_iteration in enumerate(number_of_iterations):
        print(f"-- Training ~~ Iteration n°{i + 1}/{len(number_of_iterations)} --")
        for epsilon in epsilons:
            for alpha in alphas:
                win_rate = 0.0
                game_played = 0
                model = expected_sarsa(env, max_iter_count=int(number_of_iteration),
                                   epsilon=float(epsilon), alpha=float(alpha))
                # Test a model and evaluate it
                print(
                    f"-- Testing start ~~ Iteration n°{i + 1}/{len(number_of_iterations)} -- with number_of_iteration : {int(number_of_iteration)} / epsilon : {int(epsilon)}  / alpha : {float(alpha)}--")
                for game in range(1_000):
                    env.reset()
                    while not env.is_game_over():
                        s = env.state_id()
                        aa = env.available_actions_ids()
                        if s not in model.pi:
                            a = np.random.choice(aa, p=[1 / len(aa) for _ in aa])
                        else:
                            a = np.random.choice(aa, p=[model.pi[s][a_p] for a_p in aa])
                        env.act_with_action_id(a)
                    game_played += 1
                    win_rate += env.score()
                score = round(win_rate / game_played, 4)
                print(
                    f"-- Testing done with a score of {score}% of win rate ~~ Iteration n°{i}/{len(number_of_iterations)} / epsilon : {float(epsilon)} / alpha : {float(alpha)}--")
                # Save model with score
                save_model(
                    f"expected_sarsa_on_env3_it_{int(number_of_iteration)}_eps_{round(epsilon, 3)}_alp_{round(alpha, 3)}_s_{score}",
                    q=model.q, pi=model.pi)


def demo():
    # sarsa_on_line_world()
    # q_learning_on_line_world()
    # expected_sarsa_on_line_world()

    # sarsa_on_grid_world()
    # q_learning_on_grid_world()
    # expected_sarsa_on_grid_world()

    # sarsa_on_tic_tac_toe_solo()
    # q_learning_on_tic_tac_toe_solo()
    # expected_sarsa_on_tic_tac_toe_solo()

    # sarsa_on_secret_env3()
    # q_learning_on_secret_env3()
    # expected_sarsa_on_secret_env3()
    
    dyna_q_on_tic_tac_toe_solo()
    dyna_q_on_line_world()
    dyna_q_on_grid_world()

if __name__ == "__main__":
    demo()