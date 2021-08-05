import json
from websocket import create_connection
from requests import post
import os
from util import config

HTTP_URL = config.get('REFACTORING_SERVICE', 'HTTP_URL')
WS_URL = config.get('REFACTORING_SERVICE', 'WS_URL')

DATASET_PATH = config.get('DATASET', 'DIR_PATH')

GC_OUTPUT = config.get('GOD_CLASS_TRAIN', 'OUTPUT_DIR_PATH')
SAMPLING_RATIO = config.get('GOD_CLASS_TRAIN', 'SAMPLING_RATIO')
MAX_ITERATIONS = config.get('GOD_CLASS_TRAIN', 'MAX_ITERATIONS')

CK_OUTPUT = config.get('CK_SERVICE', 'OUTPUT_DIR_PATH')
CK_LOGGER = config.get('CK_SERVICE', 'LOGGER_CONFIG_PATH')
CK_JAR = config.get('CK_SERVICE', 'JAR_PATH')


def generate_god_class_smells(samples, clean, smelly):
    ws = create_connection(WS_URL)

    data = {
        'dataset': samples,
        'totalPositive': int(smelly),
        'totalNegative': int(clean),
        'samplingRatio': SAMPLING_RATIO,
        'destinationPath': GC_OUTPUT,
        'maxIterations': MAX_ITERATIONS
    }

    try:
        res = post(HTTP_URL, data=json.dumps(data), headers={'Content-Type': 'application/json'})
    except Exception:
        return False, 'Could not connect. Check if the target server is up and running.'

    print(res.text)

    msg = ws.recv()
    print(msg)
    ws.close()


def extract_metrics():
    original_path = os.path.join(CK_OUTPUT, 'original')
    generated_path = os.path.join(CK_OUTPUT, 'generated')
    if not os.path.exists(original_path):
        os.mkdir(original_path)
        os.chdir(original_path)
        os.system('java -jar ' + CK_JAR + ' ' + DATASET_PATH + ' false 0 False')
    if not os.path.exists(generated_path):
        os.mkdir(generated_path)
        os.chdir(generated_path)
        os.system('java -jar ' + CK_JAR + ' ' + GC_OUTPUT + ' false 0 False')

