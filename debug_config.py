from config import load_config
config = load_config("experiment_config_paper.yaml")
print(type(config["data_dir"]))
print(config["data_dir"])
