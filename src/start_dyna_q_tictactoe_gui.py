if __name__ == '__main__':
    from envs import TicTacToe
    from temporal_difference_learning.dyna_Q import dyna_q

    print("ok")
    env = TicTacToe()
    model = dyna_q(env)
    env.reset()
    env.init_tkinter(model.pi)
    print("ok")