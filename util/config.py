import os

import yaml

CONFIG_FILE_PATH = '../resources/config.yaml'

props = None


def get_config():
    global props
    print(os.getcwd())
    if props is None:
        with open(CONFIG_FILE_PATH, 'r') as f:
            props = yaml.safe_load(f)
    return props


def get(scope, prop):
    global props
    if props is None:
        props = get_config()
    return props[scope][prop]

