import itertools
import subprocess

# specify a path to save girdsearch result
default_save_path = "./Solution2/models/girdsearch/"

param_gird = {
    'bert_model': ["microsoft/deberta-v3-base"],
    "epochs": [15],
    'lr': [1e-6, 1e-5, 5e-5],
    'batch_size': [5, 10, 15],
    'dropout': [0.1, 0.3, 0.5],
    'use_focus_loss': [True, False],
    "save_path": ["./Solution2/models/"],
    "device": ["cuda"],
}

def write_train_config(params_dict):
    """
    write params into train_config.py and let train.py load them.
    """
    with open("./Solution1/train_config.py", "w", encoding="utf-8") as f:
        f.write("train_configs = {\n")
        for k, v in params_dict.items():
            if type(v) is str:
                f.write(f'    "{k}": "{v}",\n')
            else:
                f.write(f'    "{k}": {v},\n')
        f.write("}\n")

if __name__ == "__main__":
    
    # generate all possible param combinations
    keys = list(param_gird.keys())
    values = list(param_gird.values())
    param_combinations = list(itertools.product(*values))
    print(f"Got {len(param_combinations)} param combinations.")

    # train model on each param comb
    for combination in param_combinations:
        params_dict = dict(zip(keys, combination))
        print("Running with params:", params_dict)

        # 1. write params into train_config.py
        write_train_config(params_dict)

        # 2. call train.py to load params and start training
        subprocess.run(["python", "./Solution2/train.py"])
        print()
