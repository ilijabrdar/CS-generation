import json
from websocket import create_connection
from requests import post
import os
from util import config

HTTP_URL = config.get('REFACTORING_SERVICE', 'HTTP_URL')
WS_URL = config.get('REFACTORING_SERVICE', 'WS_URL')

CK_LOGGER = config.get('CK_SERVICE', 'LOGGER_CONFIG_PATH')
CK_JAR = config.get('CK_SERVICE', 'JAR_PATH')


def generate_smells(**kwargs):
    data = {
        'dataset': kwargs['samples'],
        'totalPositive': int(kwargs['smelly']),
        'totalNegative': int(kwargs['clean']),
        'samplingRatio': kwargs['sampling_ratio'],
        'destinationPath': kwargs['output'],
        'maxIterations': kwargs['max_iterations'],
        'smellType': kwargs['smell_type']
    }
    __send_http(data)


def __send_http(data):
    ws = create_connection(WS_URL)
    try:
        res = post(HTTP_URL, data=json.dumps(data), headers={'Content-Type': 'application/json'})
    except Exception:
        return False, 'Could not connect. Check if the target server is up and running.'

    print(res.text)

    msg = ws.recv()
    print(msg)
    ws.close()


def extract_metrics(**kwargs):
    original_metrics_path = os.path.join(kwargs['metrics_output'], 'original')
    generated_metrics_path = os.path.join(kwargs['metrics_output'], 'generated')

    if not os.path.exists(original_metrics_path):
        os.mkdir(original_metrics_path)
        os.chdir(original_metrics_path)
        os.system('java -jar ' + CK_JAR + ' ' + kwargs['original_dataset_path'] + ' false 0 False')
    if not os.path.exists(generated_metrics_path):
        os.mkdir(generated_metrics_path)
        os.chdir(generated_metrics_path)
        os.system('java -jar ' + CK_JAR + ' ' + kwargs['generated_dataset_path'] + ' false 0 False')
