import os
import glob
import importlib


modules = glob.glob(
    os.path.sep.join(
        [os.path.dirname(__file__), '*.py']
    )
)

for f in modules:
    if os.path.isfile(f) and '__init__.py' not in f:
        module_name, ext = os.path.splitext(f)
        if ext == '.py':
            module = importlib.import_module('objectives.' + os.path.basename(module_name))

OBJECTIVES = {
    'CrossEntropy': CrossEntropy.CrossEntropy,
    'LFMMI': LFMMI.ChainLoss,
    'LFMMIOnly': LFMMIOnly.ChainLoss,
    'MultiLFMMI': MultiLFMMI.MultiChainLoss,
    'Energy': EnergyObjective.EnergyLoss,
    'InfoNCE': InfoNCEOnly.InfoNCELoss,
    'SemisupInfoNCE': SemisupInfoNCE.InfoNCELoss,
    'InfoNCE2pass': InfoNCE2pass.InfoNCELoss,
}

def build_objective(objectivename, conf):
    return OBJECTIVES[objectivename].build_objective(conf)
