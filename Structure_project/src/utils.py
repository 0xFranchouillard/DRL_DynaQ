import pickle


def save_model(file_name, pi=None, q=None, v=None):
    if pi:
        with open(f'models/{file_name}_pi.pkl', 'wb') as pkl_file:
            pickle.dump(pi, pkl_file)
    if q:
        with open(f'models/{file_name}_q.pkl', 'wb') as pkl_file:
            pickle.dump(q, pkl_file)
    if v:
        with open(f'models/{file_name}_v.pkl', 'wb') as pkl_file:
            pickle.dump(v, pkl_file)


def load_model(file_name):
    with open(f'../drl_sample_project_python/drl_lib/to_do/models/{file_name}.pkl', 'rb') as openfile:
        pkl_file = pickle.load(openfile)
    return pkl_file
