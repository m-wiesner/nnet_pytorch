import os
import glob
import importlib


modules = glob.glob(
    os.path.sep.join(
        [os.path.dirname(__file__), '*.py']
    )
)

for f in modules:
    if os.path.isfile(f) and '__init__.py' not in f and 'norms.py' not in f:
        module_name, ext = os.path.splitext(f)
        if ext == '.py':
            module = importlib.import_module('models.' + os.path.basename(module_name))

MODELS = {
    'TDNN': TDNN.TDNN,
    'ChainTDNN': TDNN.ChainTDNN,
    'Resnet': Resnet.SpeechResnet,
    'ChainResnet': Resnet.ChainSpeechResnet,
    'WideResnet': WideResnet.SpeechResnet,
    'ChainWideResnet': WideResnet.ChainSpeechResnet,
}

def build_model(modelname, conf):
    return MODELS[modelname].build_model(conf)
