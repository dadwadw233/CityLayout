import yaml
def cycle(dl):
    while True:
        for data in dl:
            yield data


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config